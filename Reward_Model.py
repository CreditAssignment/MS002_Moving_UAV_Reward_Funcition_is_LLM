import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_MAX_DELTA = (20.0, 20.0, 20.0)


def _as_tensor(x, device, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def segment_to_tensors(segment: Sequence[Dict[str, Any]], device):
    if len(segment) == 0:
        raise ValueError("segment must not be empty.")

    states = torch.stack([
        _as_tensor(t["state"], device=device).reshape(-1)
        for t in segment
    ], dim=0)
    uav_slots = torch.as_tensor(
        [int(t["uav_slot"]) for t in segment],
        dtype=torch.long,
        device=device,
    )
    deltas = torch.stack([
        _as_tensor(t["delta"], device=device).reshape(-1)
        for t in segment
    ], dim=0)
    next_states = torch.stack([
        _as_tensor(t["next_state"], device=device).reshape(-1)
        for t in segment
    ], dim=0)
    return states, uav_slots, deltas, next_states


class RewardModel(nn.Module):
    """
    Transition-level dense reward model:
        R_phi(s_t, a_t, s_{t+1})

    action a_t = (uav_slot, delta_x, delta_y, delta_z)

    输入拼接：
        current state
        selected-UAV one-hot
        normalized delta
        next state

    输出是一个未约束标量。其绝对尺度不由偏好标签直接确定，因此 SAC 使用前
    还会经过 RunningRewardNormalizer 做尺度稳定化。
    """

    def __init__(
        self,
        state_dim: int,
        num_uav: int,
        hidden_dims=(512, 256, 128),
        max_delta=DEFAULT_MAX_DELTA,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_uav = int(num_uav)

        self.register_buffer(
            "delta_scale",
            torch.tensor(max_delta, dtype=torch.float32),
        )

        input_dim = self.state_dim * 2 + self.num_uav + 3
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(last_dim, int(hidden_dim)),
                nn.ReLU(),
            ])
            last_dim = int(hidden_dim)
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state, uav_slot, delta, next_state):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if next_state.ndim == 1:
            next_state = next_state.unsqueeze(0)
        if delta.ndim == 1:
            delta = delta.unsqueeze(0)
        if uav_slot.ndim == 0:
            uav_slot = uav_slot.unsqueeze(0)

        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, got {state.shape[-1]}."
            )
        if next_state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected next_state_dim={self.state_dim}, got {next_state.shape[-1]}."
            )

        uav_slot = uav_slot.long()
        one_hot = F.one_hot(uav_slot, num_classes=self.num_uav).float()
        delta_norm = delta / self.delta_scale

        x = torch.cat(
            [state, one_hot, delta_norm, next_state],
            dim=-1,
        )
        return self.net(x).squeeze(-1)

    def score_segment(self, segment: Sequence[Dict[str, Any]], device=None):
        if device is None:
            device = next(self.parameters()).device
        states, uav_slots, deltas, next_states = segment_to_tensors(
            segment,
            device=device,
        )
        return self(states, uav_slots, deltas, next_states).sum()


class RewardModelEnsemble(nn.Module):
    """
    小型 Reward Model ensemble。

    ensemble mean 用作 SAC reward；不同成员对 A/B 偏好的 disagreement 用于主动采样。
    """

    def __init__(
        self,
        state_dim: int,
        num_uav: int,
        ensemble_size: int = 3,
        hidden_dims=(512, 256, 128),
        max_delta=DEFAULT_MAX_DELTA,
    ):
        super().__init__()
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1.")

        self.state_dim = int(state_dim)
        self.num_uav = int(num_uav)
        self.ensemble_size = int(ensemble_size)
        self.version = 0

        self.models = nn.ModuleList([
            RewardModel(
                state_dim=self.state_dim,
                num_uav=self.num_uav,
                hidden_dims=hidden_dims,
                max_delta=max_delta,
            )
            for _ in range(self.ensemble_size)
        ])

    def predict_members(self, state, uav_slot, delta, next_state):
        outputs = [
            model(state, uav_slot, delta, next_state)
            for model in self.models
        ]
        return torch.stack(outputs, dim=0)

    def predict_mean_std(self, state, uav_slot, delta, next_state):
        members = self.predict_members(state, uav_slot, delta, next_state)
        mean = members.mean(dim=0)
        std = members.std(dim=0, unbiased=False)
        return mean, std

    def forward(self, state, uav_slot, delta, next_state):
        mean, _ = self.predict_mean_std(state, uav_slot, delta, next_state)
        return mean

    @torch.no_grad()
    def score_segment_members(self, segment, device=None):
        if device is None:
            device = next(self.parameters()).device
        states, uav_slots, deltas, next_states = segment_to_tensors(
            segment,
            device=device,
        )
        transition_scores = self.predict_members(
            states,
            uav_slots,
            deltas,
            next_states,
        )
        return transition_scores.sum(dim=1)

    @torch.no_grad()
    def preference_statistics(self, segment_a, segment_b, device=None):
        score_a = self.score_segment_members(segment_a, device=device)
        score_b = self.score_segment_members(segment_b, device=device)
        logits = score_a - score_b
        probs = torch.sigmoid(logits)
        mean_prob = probs.mean().item()
        prob_std = probs.std(unbiased=False).item()
        closeness = 1.0 - 2.0 * abs(mean_prob - 0.5)
        uncertainty = max(0.0, closeness) + 2.0 * prob_std
        return {
            "member_prob_a": probs.detach().cpu().tolist(),
            "mean_prob_a": float(mean_prob),
            "prob_std": float(prob_std),
            "uncertainty": float(uncertainty),
        }


class RunningRewardNormalizer:
    """在线 Welford 统计，用于稳定 Reward Model 输出尺度。"""

    def __init__(self, clip_value: float = 5.0, eps: float = 1e-6):
        self.clip_value = float(clip_value)
        self.eps = float(eps)
        self.count = 0.0
        self.mean = 0.0
        self.m2 = 0.0

    @property
    def variance(self):
        if self.count <= 1:
            return 1.0
        return max(self.m2 / (self.count - 1.0), self.eps)

    @property
    def std(self):
        return math.sqrt(self.variance)

    def update(self, values):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        for x in values:
            self.count += 1.0
            delta = float(x) - self.mean
            self.mean += delta / self.count
            delta2 = float(x) - self.mean
            self.m2 += delta * delta2

    def normalize(self, values, update: bool = False):
        if update:
            self.update(values)

        if isinstance(values, torch.Tensor):
            normalized = (values - float(self.mean)) / float(self.std + self.eps)
            return torch.clamp(
                normalized,
                -self.clip_value,
                self.clip_value,
            )

        values = np.asarray(values, dtype=np.float32)
        normalized = (values - self.mean) / (self.std + self.eps)
        return np.clip(normalized, -self.clip_value, self.clip_value)

    def reset(self):
        self.count = 0.0
        self.mean = 0.0
        self.m2 = 0.0

    def state_dict(self):
        return {
            "clip_value": self.clip_value,
            "eps": self.eps,
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
        }

    def load_state_dict(self, state_dict):
        self.clip_value = float(state_dict["clip_value"])
        self.eps = float(state_dict["eps"])
        self.count = float(state_dict["count"])
        self.mean = float(state_dict["mean"])
        self.m2 = float(state_dict["m2"])


def _record_label_value(record):
    label = record["label"]
    if isinstance(label, str):
        label = label.strip().lower()
        if label == "a":
            return 1.0
        if label == "b":
            return 0.0
        if label == "tie":
            return 0.5
        raise ValueError(f"Unsupported label string: {label}")
    return float(label)


def _preference_logits_for_model(model, records, device):
    logits = []
    targets = []
    weights = []

    for record in records:
        score_a = model.score_segment(record["segment_a"], device=device)
        score_b = model.score_segment(record["segment_b"], device=device)
        logits.append(score_a - score_b)
        targets.append(_record_label_value(record))
        weights.append(float(record.get("confidence", 1.0)))

    logits = torch.stack(logits)
    targets = torch.tensor(targets, dtype=torch.float32, device=device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = torch.clamp(weights, min=0.05, max=1.0)
    return logits, targets, weights


def train_reward_model_ensemble(
    ensemble: RewardModelEnsemble,
    preference_buffer,
    optimizers: Sequence[torch.optim.Optimizer],
    device,
    epochs: int = 20,
    batch_size: int = 16,
    grad_clip: float = 10.0,
):
    """
    使用 Bradley-Terry 风格 pairwise preference loss 训练 RM ensemble。

    对一个 trajectory segment：
        S(segment) = sum_t R_phi(s_t,a_t,s_{t+1})

    P(A > B) = sigmoid(S_A - S_B)

    tie 标签使用 target=0.5；LLM confidence 作为样本权重。
    """
    if len(optimizers) != ensemble.ensemble_size:
        raise ValueError("Need one optimizer for each reward model member.")
    if len(preference_buffer) == 0:
        return {"loss": None, "accuracy": None, "num_preferences": 0}

    ensemble.train()
    all_losses = []

    for _ in range(int(epochs)):
        for model_index, model in enumerate(ensemble.models):
            # 每个 ensemble member 独立 bootstrap，帮助形成有效 disagreement。
            records = preference_buffer.sample(
                min(int(batch_size), len(preference_buffer)),
                with_replacement=True,
            )

            logits, targets, weights = _preference_logits_for_model(
                model,
                records,
                device=device,
            )
            per_sample_loss = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
            loss = (per_sample_loss * weights).sum() / weights.sum()

            optimizers[model_index].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizers[model_index].step()
            all_losses.append(float(loss.detach().cpu()))

    ensemble.version += 1
    ensemble.eval()

    # 用当前 preference dataset 粗略报告 ensemble mean 的 pairwise accuracy。
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        evaluation_records = preference_buffer.sample(
            min(len(preference_buffer), 256),
            with_replacement=False,
        )
        for record in evaluation_records:
            target = _record_label_value(record)
            if target == 0.5:
                continue
            stats = ensemble.preference_statistics(
                record["segment_a"],
                record["segment_b"],
                device=device,
            )
            pred_a = stats["mean_prob_a"] >= 0.5
            true_a = target > 0.5
            correct += float(pred_a == true_a)
            total += 1.0

    accuracy = correct / total if total > 0 else None
    return {
        "loss": float(np.mean(all_losses)) if all_losses else None,
        "accuracy": accuracy,
        "num_preferences": len(preference_buffer),
        "rm_version": ensemble.version,
    }


def calibrate_reward_normalizer(
    ensemble: RewardModelEnsemble,
    reward_normalizer: RunningRewardNormalizer,
    replay_buffer,
    device,
    num_samples: int = 1024,
):
    if len(replay_buffer) == 0:
        return

    sample_size = min(int(num_samples), len(replay_buffer))
    batch = replay_buffer.sample(sample_size, device=device)
    with torch.no_grad():
        rewards = ensemble(
            batch["states"],
            batch["uav_slots"],
            batch["deltas"],
            batch["next_states"],
        )
    reward_normalizer.update(rewards)
