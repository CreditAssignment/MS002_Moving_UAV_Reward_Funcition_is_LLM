import copy
import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class PreferenceSampler:
    """
    主动偏好采样器。

    推荐路径：从同一个 Monitor 状态 deepcopy 出多个分支，生成 A/B 短轨迹。
    这样两段轨迹具有完全相同的起点，LLM 的比较更接近“动作/策略优劣”而不是
    “初始状态难度差异”。

    备用路径：从 Replay Buffer 的历史连续片段中寻找起始状态相近且 RM ensemble
    不确定性较高的 pair。
    """

    def __init__(
        self,
        segment_length: int = 5,
        candidate_pairs: int = 24,
        uncertainty_weight: float = 0.75,
        diversity_weight: float = 0.25,
    ):
        self.segment_length = int(segment_length)
        self.candidate_pairs = int(candidate_pairs)
        self.uncertainty_weight = float(uncertainty_weight)
        self.diversity_weight = float(diversity_weight)

        if self.segment_length <= 0:
            raise ValueError("segment_length must be positive.")
        if self.candidate_pairs <= 0:
            raise ValueError("candidate_pairs must be positive.")

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def _sample_actor_action(actor, state, select_able_mask):
        with torch.no_grad():
            _, uav_slot, delta, _ = actor.sample(
                state=state,
                select_able_mask=select_able_mask,
            )
        return int(uav_slot.reshape(-1)[0].item()), delta.reshape(-1, 3)[0].detach().clone()

    @staticmethod
    def _action_distance(action_a, action_b, num_uav: int, delta_scale=None):
        slot_a, delta_a = action_a
        slot_b, delta_b = action_b
        discrete = 1.0 if int(slot_a) != int(slot_b) else 0.0

        delta_a = PreferenceSampler._to_numpy(delta_a).reshape(-1).astype(np.float32)
        delta_b = PreferenceSampler._to_numpy(delta_b).reshape(-1).astype(np.float32)
        if delta_scale is None:
            delta_scale = np.ones(3, dtype=np.float32)
        else:
            delta_scale = PreferenceSampler._to_numpy(delta_scale).reshape(-1).astype(np.float32)
            delta_scale = np.maximum(delta_scale, 1e-6)

        continuous = float(np.linalg.norm((delta_a - delta_b) / delta_scale) / math.sqrt(3.0))
        return discrete + continuous

    def _rollout_branch(
        self,
        monitor,
        actor,
        max_num_uav: int,
        device,
        first_action=None,
    ):
        """在一个动作分支下连续采集segment_length个transition，然后这些transition全部装载到segment里，最后返回这个segment"""
        segment = []
        current_state_np = np.asarray(
            monitor.getState(max_num_uav=max_num_uav),
            dtype=np.float32,
        )

        for local_step in range(self.segment_length):
            select_able_mask = monitor.getSelectAbleMask()
            if not any(select_able_mask):
                break

            current_state = torch.tensor(
                current_state_np,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            if local_step == 0 and first_action is not None:
                uav_slot = int(first_action[0])
                delta = first_action[1]
                if isinstance(delta, torch.Tensor):
                    delta_tensor = delta.detach().to(device=device, dtype=torch.float32).reshape(1, 3)
                else:
                    delta_tensor = torch.tensor(delta, dtype=torch.float32, device=device).reshape(1, 3)
            else:
                uav_slot, sampled_delta = self._sample_actor_action(
                    actor,
                    current_state,
                    select_able_mask,
                )
                delta_tensor = sampled_delta.to(device=device).reshape(1, 3)

            next_state_np, env_done, info = monitor.step(
                uav_slot=uav_slot,
                delta=delta_tensor,
                max_num_uav=max_num_uav,
                terminate_on_connected=False,
                terminate_on_target=True,
                time_step_counter=local_step,
            )
            next_state_np = np.asarray(next_state_np, dtype=np.float32)
            next_select_able_mask = monitor.getSelectAbleMask()

            transition = {
                "state": current_state_np.copy(),
                "select_able_mask": np.asarray(select_able_mask, dtype=np.bool_).copy(),
                "uav_slot": int(uav_slot),
                "delta": delta_tensor.detach().cpu().numpy().reshape(-1).astype(np.float32),
                "next_state": next_state_np.copy(),
                "next_select_able_mask": np.asarray(next_select_able_mask, dtype=np.bool_).copy(),
                "done": float(env_done),
                "info": copy.deepcopy(info),
            }
            segment.append(transition)
            current_state_np = next_state_np

            if env_done:
                break

        return segment

    def generate_branch_candidates(
        self,
        monitor,
        actor,
        max_num_uav: int,
        device,
        reward_model_ensemble=None,
        num_candidates: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """从同一起始网络状态生成多个 A/B trajectory candidate pairs。"""
        if num_candidates is None:
            num_candidates = self.candidate_pairs
        num_candidates = int(num_candidates)

        start_state = torch.tensor(
            monitor.getState(max_num_uav=max_num_uav),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        select_able_mask = monitor.getSelectAbleMask()
        if not any(select_able_mask):
            return []

        delta_scale = getattr(actor, "delta_scale", None)
        candidates = []

        for _ in range(num_candidates):
            first_a = self._sample_actor_action(actor, start_state, select_able_mask)
            first_b = self._sample_actor_action(actor, start_state, select_able_mask)

            # 尽量确保两个候选在首动作上不是几乎相同。
            for _retry in range(8):
                distance = self._action_distance(
                    first_a,
                    first_b,
                    num_uav=max_num_uav,
                    delta_scale=delta_scale,
                )
                if distance >= 0.25:
                    break
                first_b = self._sample_actor_action(actor, start_state, select_able_mask)

            env_a = copy.deepcopy(monitor)
            env_b = copy.deepcopy(monitor)

            segment_a = self._rollout_branch(  # 沿着分支动作a采集一个segment_length长度的后续，存在segment_a里
                env_a,
                actor,
                max_num_uav=max_num_uav,
                device=device,
                first_action=first_a,
            )
            segment_b = self._rollout_branch(  # 沿着分支动作b采集一个segment_length长度的后续，存在segment_b里
                env_b,
                actor,
                max_num_uav=max_num_uav,
                device=device,
                first_action=first_b,
            )

            if not segment_a or not segment_b:
                continue

            first_action_distance = self._action_distance(
                first_a,
                first_b,
                num_uav=max_num_uav,
                delta_scale=delta_scale,
            )

            diagnostics = {
                "sampling_mode": "same_state_branching",
                "start_state_distance": 0.0,
                "first_action_distance": float(first_action_distance),
            }

            if reward_model_ensemble is not None:
                stats = reward_model_ensemble.preference_statistics(
                    segment_a,
                    segment_b,
                    device=device,
                )
                diagnostics.update(stats)
                uncertainty = stats["uncertainty"]
            else:
                uncertainty = 1.0

            # diversity 只作为次级因素，主要仍优先询问 ensemble 不确定的 pair。
            diversity = min(1.0, float(first_action_distance) / 2.0)
            selection_score = (
                self.uncertainty_weight * float(uncertainty)
                + self.diversity_weight * diversity
            )
            diagnostics["selection_score"] = float(selection_score)

            candidates.append({
                "segment_a": segment_a,
                "segment_b": segment_b,
                "diagnostics": diagnostics,
            })

        candidates.sort(
            key=lambda item: item["diagnostics"].get("selection_score", 0.0),
            reverse=True,
        )
        return candidates

    def select_branch_pairs(
        self,
        monitor,
        actor,
        max_num_uav: int,
        device,
        num_pairs: int = 8,
        reward_model_ensemble=None,
    ):
        candidates = self.generate_branch_candidates(
            monitor=monitor,
            actor=actor,
            max_num_uav=max_num_uav,
            device=device,
            reward_model_ensemble=reward_model_ensemble,
            num_candidates=max(self.candidate_pairs, int(num_pairs)),
        )
        return candidates[:int(num_pairs)]

    @staticmethod
    def _start_state_distance(segment_a, segment_b):
        a = np.asarray(segment_a[0]["state"], dtype=np.float32).reshape(-1)
        b = np.asarray(segment_b[0]["state"], dtype=np.float32).reshape(-1)
        if a.shape != b.shape:
            return float("inf")
        return float(np.linalg.norm(a - b) / math.sqrt(max(a.size, 1)))

    def select_pairs_from_replay(
        self,
        replay_buffer,
        num_pairs: int,
        reward_model_ensemble=None,
        device=None,
        history_segments: int = 128,
    ):
        """备用：从历史 replay 短轨迹中选择起点相近 + RM 不确定的 pair。"""
        segments = replay_buffer.sample_segments(
            num_segments=history_segments,
            segment_length=self.segment_length,
        )
        if len(segments) < 2:
            return []

        raw_candidates = []
        max_trials = min(2000, len(segments) * (len(segments) - 1) // 2)
        for _ in range(max_trials):
            segment_a, segment_b = random.sample(segments, 2)
            distance = self._start_state_distance(segment_a, segment_b)
            similarity = math.exp(-5.0 * distance) if math.isfinite(distance) else 0.0

            diagnostics = {
                "sampling_mode": "replay_nearest_state",
                "start_state_distance": float(distance),
                "state_similarity": float(similarity),
            }

            if reward_model_ensemble is not None:
                stats = reward_model_ensemble.preference_statistics(
                    segment_a,
                    segment_b,
                    device=device,
                )
                diagnostics.update(stats)
                uncertainty = stats["uncertainty"]
            else:
                uncertainty = 1.0

            score = (
                self.uncertainty_weight * float(uncertainty)
                + self.diversity_weight * float(similarity)
            )
            diagnostics["selection_score"] = float(score)
            raw_candidates.append({
                "segment_a": segment_a,
                "segment_b": segment_b,
                "diagnostics": diagnostics,
            })

        raw_candidates.sort(
            key=lambda item: item["diagnostics"]["selection_score"],
            reverse=True,
        )

        selected = []
        seen = set()
        for candidate in raw_candidates:
            ids_a = tuple(t.get("transition_id") for t in candidate["segment_a"])
            ids_b = tuple(t.get("transition_id") for t in candidate["segment_b"])
            key = tuple(sorted((ids_a, ids_b), key=str))
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
            if len(selected) >= int(num_pairs):
                break
        return selected
