import copy
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class SACReplayBuffer:
    """
    Replay Buffer for the LLM-assisted preference-reward Hybrid SAC pipeline.

    新版 buffer 不再把人工 reward 作为 transition 的固定字段。
    SAC 每次训练时都用“当前版本 Reward Model”重新计算 reward，天然实现 lazy relabeling。

    每条 transition 保存：
        transition_id
        episode_id
        time_step
        state
        select_able_mask
        uav_slot
        delta
        next_state
        next_select_able_mask
        done
        info

    其中 info 来自新版 Monitor.step()，包含客观网络变化和 LLM 偏好判断需要的结构化信息。
    """

    def __init__(self, capacity: int = 50000):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")

        self.capacity = int(capacity)
        self.buffer: List[Optional[Dict[str, Any]]] = [None] * self.capacity
        self.ptr = 0
        self.size = 0
        self.state_dim = None
        self.num_uav = None
        self.total_pushed = 0

    @staticmethod
    def _to_numpy_1d(x, dtype=np.float32):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=dtype).reshape(-1)

    @staticmethod
    def _to_int_scalar(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().item()
        return int(x)

    def push(
        self,
        state,
        select_able_mask,
        uav_slot,
        delta,
        next_state,
        next_select_able_mask,
        done,
        info: Optional[Dict[str, Any]] = None,
        episode_id: Optional[int] = None,
        time_step: Optional[int] = None,
    ) -> int:
        """存入一条原始 transition，并返回单调递增的 transition_id。"""

        state = self._to_numpy_1d(state, np.float32)
        next_state = self._to_numpy_1d(next_state, np.float32)
        select_able_mask = self._to_numpy_1d(select_able_mask, np.bool_)
        next_select_able_mask = self._to_numpy_1d(next_select_able_mask, np.bool_)
        delta = self._to_numpy_1d(delta, np.float32)
        uav_slot = self._to_int_scalar(uav_slot)
        done = float(done)

        if delta.shape[0] != 3:
            raise ValueError(
                f"delta should have shape [3], but got {delta.shape}."
            )
        if state.shape != next_state.shape:
            raise ValueError(
                f"state and next_state shapes differ: {state.shape} vs {next_state.shape}."
            )
        if select_able_mask.shape != next_select_able_mask.shape:
            raise ValueError(
                "select_able_mask and next_select_able_mask must have the same shape."
            )

        if self.state_dim is None:
            self.state_dim = int(state.shape[0])
        elif state.shape[0] != self.state_dim:
            raise ValueError(
                f"state_dim mismatch: expected {self.state_dim}, got {state.shape[0]}."
            )

        if self.num_uav is None:
            self.num_uav = int(select_able_mask.shape[0])
        elif select_able_mask.shape[0] != self.num_uav:
            raise ValueError(
                f"num_uav mismatch: expected {self.num_uav}, got {select_able_mask.shape[0]}."
            )

        if not 0 <= uav_slot < self.num_uav:
            raise ValueError(
                f"uav_slot should be in [0, {self.num_uav - 1}], got {uav_slot}."
            )

        transition_id = int(self.total_pushed)
        transition = {
            "transition_id": transition_id,
            "episode_id": None if episode_id is None else int(episode_id),
            "time_step": None if time_step is None else int(time_step),
            "state": state.copy(),
            "select_able_mask": select_able_mask.copy(),
            "uav_slot": uav_slot,
            "delta": delta.copy(),
            "next_state": next_state.copy(),
            "next_select_able_mask": next_select_able_mask.copy(),
            "done": done,
            "info": copy.deepcopy(info),
        }

        self.buffer[self.ptr] = transition
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_pushed += 1
        return transition_id

    def _valid_transitions(self) -> List[Dict[str, Any]]:
        items = [t for t in self.buffer if t is not None]
        items.sort(key=lambda t: t["transition_id"])
        return items

    def sample(
        self,
        batch_size: int,
        device=None,
        return_info: bool = False,
    ) -> Dict[str, Any]:
        if device is None:
            device = torch.device("cpu")
        if self.size < batch_size:
            raise ValueError(
                f"Not enough samples: current size={self.size}, batch_size={batch_size}."
            )

        valid = self._valid_transitions()
        transitions = random.sample(valid, int(batch_size))

        states = np.stack([t["state"] for t in transitions], axis=0)
        select_able_masks = np.stack([t["select_able_mask"] for t in transitions], axis=0)
        uav_slots = np.asarray([t["uav_slot"] for t in transitions], dtype=np.int64)
        deltas = np.stack([t["delta"] for t in transitions], axis=0)
        next_states = np.stack([t["next_state"] for t in transitions], axis=0)
        next_select_able_masks = np.stack(
            [t["next_select_able_mask"] for t in transitions],
            axis=0,
        )
        dones = np.asarray([t["done"] for t in transitions], dtype=np.float32)
        transition_ids = np.asarray(
            [t["transition_id"] for t in transitions],
            dtype=np.int64,
        )

        batch = {
            "states": torch.tensor(states, dtype=torch.float32, device=device),
            "select_able_masks": torch.tensor(select_able_masks, dtype=torch.bool, device=device),
            "uav_slots": torch.tensor(uav_slots, dtype=torch.long, device=device),
            "deltas": torch.tensor(deltas, dtype=torch.float32, device=device),
            "next_states": torch.tensor(next_states, dtype=torch.float32, device=device),
            "next_select_able_masks": torch.tensor(
                next_select_able_masks,
                dtype=torch.bool,
                device=device,
            ),
            "dones": torch.tensor(dones, dtype=torch.float32, device=device),
            "transition_ids": torch.tensor(
                transition_ids,
                dtype=torch.long,
                device=device,
            ),
        }

        if return_info:
            batch["infos"] = [copy.deepcopy(t["info"]) for t in transitions]
            batch["episode_ids"] = [t["episode_id"] for t in transitions]
            batch["time_steps"] = [t["time_step"] for t in transitions]

        return batch

    def sample_transition_dicts(self, batch_size: int) -> List[Dict[str, Any]]:
        if self.size < batch_size:
            raise ValueError("Not enough transitions in replay buffer.")
        return copy.deepcopy(random.sample(self._valid_transitions(), int(batch_size)))

    def recent(self, n: int) -> List[Dict[str, Any]]:
        valid = self._valid_transitions()
        return copy.deepcopy(valid[-int(n):])

    def get_by_transition_id(self, transition_id: int):
        for transition in self.buffer:
            if transition is not None and transition["transition_id"] == int(transition_id):
                return copy.deepcopy(transition)
        return None

    def build_contiguous_segments(
        self,
        segment_length: int = 5,
        max_segments: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        从 replay 中构造同一 episode 内、time_step 连续的短轨迹。
        该接口主要用于历史偏好采样备用路径。
        """
        segment_length = int(segment_length)
        if segment_length <= 0:
            raise ValueError("segment_length must be positive.")

        grouped = defaultdict(list)
        for transition in self._valid_transitions():
            episode_id = transition["episode_id"]
            if episode_id is None:
                continue
            grouped[episode_id].append(transition)

        segments = []
        for episode_id in sorted(grouped.keys()):
            episode = sorted(
                grouped[episode_id],
                key=lambda t: (
                    -1 if t["time_step"] is None else t["time_step"],
                    t["transition_id"],
                ),
            )
            for start in range(0, len(episode) - segment_length + 1):
                window = episode[start:start + segment_length]
                time_steps = [t["time_step"] for t in window]
                if all(ts is not None for ts in time_steps):
                    if any(
                        time_steps[k + 1] != time_steps[k] + 1
                        for k in range(len(time_steps) - 1)
                    ):
                        continue
                if any(window[k]["done"] > 0.5 for k in range(len(window) - 1)):
                    continue
                segments.append(copy.deepcopy(window))

        if max_segments is not None and len(segments) > max_segments:
            segments = random.sample(segments, int(max_segments))
        return segments

    def sample_segments(
        self,
        num_segments: int,
        segment_length: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        segments = self.build_contiguous_segments(segment_length=segment_length)
        if not segments:
            return []
        if len(segments) <= num_segments:
            return segments
        return random.sample(segments, int(num_segments))

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= int(batch_size)

    def clear(self):
        self.buffer = [None] * self.capacity
        self.ptr = 0
        self.size = 0
        self.state_dim = None
        self.num_uav = None
        self.total_pushed = 0

    def __len__(self):
        return self.size
