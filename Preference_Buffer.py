import copy
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class PreferenceBuffer:
    """
    保存 LLM 生成的 pairwise preference 数据。

    每条记录：
        segment_a: trajectory segment
        segment_b: trajectory segment
        label: 1.0(A preferred), 0.0(B preferred), 0.5(tie)
        confidence: [0, 1]
        source: openai / heuristic_fallback / manual / ...
        reason: LLM 的简短解释，仅用于审计，不参与 Reward Model 训练
        metadata: uncertainty / state similarity / query id 等调试信息
    """

    LABEL_MAP = {
        "a": 1.0,
        "b": 0.0,
        "tie": 0.5,
    }

    def __init__(self, capacity: int = 10000):
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.records: List[Dict[str, Any]] = []
        self.total_added = 0

    @classmethod
    def normalize_label(cls, label):
        if isinstance(label, str):
            key = label.strip().lower()
            if key not in cls.LABEL_MAP:
                raise ValueError(f"Unsupported preference label: {label}")
            return cls.LABEL_MAP[key]

        value = float(label)
        if value not in (0.0, 0.5, 1.0):
            raise ValueError("Numeric label must be 0.0, 0.5 or 1.0.")
        return value

    def add(
        self,
        segment_a,
        segment_b,
        label,
        confidence: float = 1.0,
        source: str = "unknown",
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not segment_a or not segment_b:
            raise ValueError("Both preference segments must be non-empty.")

        label_value = self.normalize_label(label)
        confidence = float(max(0.0, min(1.0, confidence)))

        record_id = int(self.total_added)
        record = {
            "preference_id": record_id,
            "segment_a": copy.deepcopy(segment_a),
            "segment_b": copy.deepcopy(segment_b),
            "label": label_value,
            "confidence": confidence,
            "source": str(source),
            "reason": str(reason),
            "metadata": copy.deepcopy(metadata) if metadata is not None else {},
        }

        self.records.append(record)
        self.total_added += 1

        if len(self.records) > self.capacity:
            overflow = len(self.records) - self.capacity
            del self.records[:overflow]

        return record_id

    def add_from_label_result(self, pair, label_result):
        preference = label_result["preference"]
        return self.add(
            segment_a=pair["segment_a"],
            segment_b=pair["segment_b"],
            label=preference,
            confidence=label_result.get("confidence", 1.0),
            source=label_result.get("source", "unknown"),
            reason=label_result.get("reason", ""),
            metadata={
                **copy.deepcopy(pair.get("diagnostics", {})),
                **copy.deepcopy(label_result.get("metadata", {})),
            },
        )

    def sample(self, batch_size: int, with_replacement: bool = False):
        if len(self.records) == 0:
            return []

        batch_size = int(batch_size)
        if with_replacement:
            return copy.deepcopy(
                random.choices(self.records, k=batch_size)
            )

        batch_size = min(batch_size, len(self.records))
        return copy.deepcopy(random.sample(self.records, batch_size))

    def latest(self, n: int = 10):
        return copy.deepcopy(self.records[-int(n):])

    def source_counts(self):
        counts = {}
        for record in self.records:
            source = record.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts

    def save_pickle(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "capacity": self.capacity,
                    "records": self.records,
                    "total_added": self.total_added,
                },
                f,
            )

    @classmethod
    def load_pickle(cls, path):
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        buffer = cls(capacity=payload["capacity"])
        buffer.records = payload["records"]
        buffer.total_added = payload["total_added"]
        return buffer

    def append_jsonl(self, path, record):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    def export_jsonl(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    def __len__(self):
        return len(self.records)
