import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional


SYSTEM_INSTRUCTIONS = """
You are a preference judge for reinforcement learning in a multi-UAV movement task.
You compare two short trajectory segments A and B that start from the same or nearly identical state.

The ONLY task objective is to move EVERY UAV toward the fixed target position:
    (x, y, z) = (0, 0, 0)

Use the supplied target metrics and UAV positions. Judge the trajectories according to this priority:
1. Prefer a trajectory in which all UAVs have reached the target tolerance.
2. Otherwise, prefer the trajectory with more UAVs inside the target tolerance.
3. Prefer the trajectory with a smaller maximum UAV-to-target distance. No UAV should be left far away.
4. Prefer the trajectory with smaller mean and total UAV-to-target distances.
5. When the end states are very close, prefer the trajectory that made more consistent progress toward the target.
6. If the two trajectories are essentially equivalent under these criteria, return tie.

Network connectivity, communication links, connected components, ground coverage, and coverage overlap are
IRRELEVANT to this sanity-check task and MUST NOT influence the preference.

Do not invent facts that are not present in the input. Do not output a numeric reward.

You MUST output ONLY a single JSON object, with NO extra text, markdown fences, or explanation outside the JSON.
The JSON schema is:
{
  "labels": [
    {
      "pair_index": <integer, matches the input pair_index>,
      "preference": "A" | "B" | "tie",
      "confidence": <float between 0.0 and 1.0>,
      "reason": <short string>
    },
    ...
  ]
}
Every input pair_index must appear exactly once in "labels".
""".strip()


class LLMPreferenceLabeler:
    """
    LLM pairwise preference labeler.

    mode:
        "qwen"      - 调用阿里云 DashScope 千问（Generation.call）。
        "openai"    - 调用 OpenAI Responses API。
        "heuristic" - 仅用于本地调试闭环，不是正式 LLM 实验。
        "auto"      - 有 DASHSCOPE_API_KEY 用 qwen；否则有 OPENAI_API_KEY 用 openai；
                      都没有则警告并退化到 heuristic。

    模型名可通过环境变量覆盖：
        QWEN_PREFERENCE_MODEL   (默认 "qwen-plus")
        OPENAI_PREFERENCE_MODEL (默认 "gpt-5.2")
    API Key 读取：
        DASHSCOPE_API_KEY  或 api_key=...
        OPENAI_API_KEY     或 api_key=...
    """

    def __init__(
        self,
        mode: str = "qwen",
        model: Optional[str] = None,
        api_key: Optional[str] = "接入千问API的key，需要自行注册申请",
        client=None,
    ):
        mode = str(mode).strip().lower()
        if mode not in {"auto", "qwen", "openai", "heuristic"}:
            raise ValueError("mode must be 'auto', 'qwen', 'openai' or 'heuristic'.")

        self.mode = mode
        # 两个模型名分开存，避免 auto 时互相污染
        self.qwen_model = model or os.getenv("QWEN_PREFERENCE_MODEL", "qwen-plus")
        self.openai_model = model or os.getenv("OPENAI_PREFERENCE_MODEL", "gpt-5.2")
        self.model = self.qwen_model  # 兼容旧字段

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
        self._client = client
        self._warned_fallback = False

    # ------------------------------------------------------------------
    # mode 解析
    # ------------------------------------------------------------------
    def _resolve_mode(self):
        if self.mode == "heuristic":
            return "heuristic"
        if self.mode == "qwen":
            return "qwen"
        if self.mode == "openai":
            return "openai"

        # auto
        if os.getenv("DASHSCOPE_API_KEY") or (self.api_key and str(self.api_key).startswith("sk-")):
            # 若显式给了 DASHSCOPE_API_KEY，优先 qwen
            if os.getenv("DASHSCOPE_API_KEY"):
                return "qwen"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        # 都读不到就走 heuristic
        return "heuristic"

    # ------------------------------------------------------------------
    # OpenAI 客户端
    # ------------------------------------------------------------------
    def _ensure_openai_client(self):
        if self._client is not None:
            return self._client

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI mode requires the 'openai' Python package. "
                "Install it in your training environment first."
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OpenAI mode requires OPENAI_API_KEY or api_key=... ."
            )

        self._client = OpenAI(api_key=api_key)
        return self._client

    # ------------------------------------------------------------------
    # 视图 / 摘要
    # ------------------------------------------------------------------
    @staticmethod
    def _target_view(snapshot):
        target = snapshot.get("target", {})
        distances = target.get("distances", {}) or {}
        return {
            "target_position": target.get("target_position", [0.0, 0.0, 0.0]),
            "tolerance": float(target.get("tolerance", 30.0)),
            "distances": {
                str(k): float(v) for k, v in distances.items()
            },
            "sum_distance": float(target.get("sum_distance", 0.0)),
            "mean_distance": float(target.get("mean_distance", 0.0)),
            "max_distance": float(target.get("max_distance", 0.0)),
            "num_within_tolerance": int(target.get("num_within_tolerance", 0)),
            "all_reached": bool(target.get("all_reached", False)),
        }

    @staticmethod
    def _uav_view(snapshot):
        result = []
        for uav in snapshot.get("uavs", []):
            result.append({
                "slot": uav.get("slot"),
                "uav_id": uav.get("uav_id"),
                "active": uav.get("active"),
                "position": uav.get("position"),
                "distance_to_target": uav.get("distance_to_target"),
            })
        return result

    @classmethod
    def summarize_segment(cls, segment):
        if not segment:
            raise ValueError("Cannot summarize an empty segment.")

        first_info = segment[0].get("info") or {}
        last_info = segment[-1].get("info") or {}
        start_snapshot = first_info.get("before", {})
        end_snapshot = last_info.get("after", {})

        actions = []
        total_move = 0.0
        improving_steps = 0
        worsening_steps = 0

        for index, transition in enumerate(segment):
            info = transition.get("info") or {}
            action = info.get("action", {})
            changes = info.get("changes", {})
            executed_delta = action.get(
                "executed_delta",
                transition.get("delta", [0, 0, 0]),
            )
            try:
                total_move += math.sqrt(sum(float(v) ** 2 for v in executed_delta))
            except Exception:
                pass

            distance_delta = float(
                changes.get("sum_distance_to_target_delta", 0.0)
            )
            if distance_delta < -1e-9:
                improving_steps += 1
            elif distance_delta > 1e-9:
                worsening_steps += 1

            actions.append({
                "step": index,
                "uav_slot": action.get("uav_slot", transition.get("uav_slot")),
                "uav_id": action.get("uav_id"),
                "executed_delta": executed_delta,
                "position_before": action.get("position_before"),
                "position_after": action.get("position_after"),
                "sum_distance_to_target_delta": distance_delta,
            })

        return {
            "length": len(segment),
            "start_target": cls._target_view(start_snapshot),
            "end_target": cls._target_view(end_snapshot),
            "start_uavs": cls._uav_view(start_snapshot),
            "end_uavs": cls._uav_view(end_snapshot),
            "trajectory_events": {
                "improving_steps": int(improving_steps),
                "worsening_steps": int(worsening_steps),
                "total_executed_move": float(total_move),
            },
            "actions": actions,
        }

    def build_batch_payload(self, pairs):
        payload = []
        for index, pair in enumerate(pairs):
            payload.append({
                "pair_index": index,
                "trajectory_A": self.summarize_segment(pair["segment_a"]),
                "trajectory_B": self.summarize_segment(pair["segment_b"]),
            })
        return payload

    # ------------------------------------------------------------------
    # JSON 解析工具
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_json(text: str) -> dict:
        """从模型输出里抠出 JSON。容忍 ```json ... ``` 包裹和前后废话。"""
        text = text.strip()
        if text.startswith("```"):
            # 去掉三反引号围栏
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        # 再兜底：从第一个 { 到最后一个 }
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Cannot locate JSON in model output: {text[:200]!r}")
        return json.loads(text[start:end + 1])

    def _pack_results(self, parsed, pairs, source, metadata):
        labels_by_index = {int(item["pair_index"]): item for item in parsed["labels"]}
        results = []
        for index in range(len(pairs)):
            if index not in labels_by_index:
                raise RuntimeError(f"LLM response is missing pair_index={index}.")
            item = labels_by_index[index]
            pref = str(item["preference"]).strip()
            if pref not in {"A", "B", "tie"}:
                # 兼容大小写 / TIE
                pl = pref.lower()
                if pl == "tie":
                    pref = "tie"
                elif pl in {"a", "b"}:
                    pref = pl.upper()
                else:
                    raise RuntimeError(f"Invalid preference value: {pref!r}")
            results.append({
                "preference": pref,
                "confidence": float(item.get("confidence", 0.5)),
                "reason": str(item.get("reason", "")),
                "source": source,
                "metadata": metadata,
            })
        return results

    # ------------------------------------------------------------------
    # 千问标注
    # ------------------------------------------------------------------
    def _label_pairs_qwen(self, pairs):
        try:
            import dashscope
            from dashscope import Generation
        except ImportError as exc:
            raise RuntimeError(
                "Qwen mode requires the 'dashscope' Python package. "
                "Install it via `pip install dashscope` first."
            ) from exc

        api_key = self.api_key or os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Qwen mode requires DASHSCOPE_API_KEY or api_key=... ."
            )
        dashscope.api_key = api_key

        payload = self.build_batch_payload(pairs)

        user_prompt = (
            "Compare every A/B pair below. The pair_index in your answer must match the input.\n"
            "Return ONLY a JSON object (no markdown fences, no explanation) matching the required schema.\n\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

        # 优先尝试 json_object（部分 qwen 版本支持）；如报错则回退纯文本
        try:
            response = Generation.call(
                model=self.qwen_model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                result_format="message",
                response_format={"type": "json_object"},
            )
        except TypeError:
            # dashscope 老版本不支持 response_format 参数
            response = Generation.call(
                model=self.qwen_model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                result_format="message",
            )

        # 若 http 层返回错误，视情况不带 response_format 再试一次
        try:
            from http import HTTPStatus
            ok = response.status_code == HTTPStatus.OK
        except Exception:
            ok = False

        if not ok:
            # 可能是 response_format 不被支持，退回纯文本调用
            response = Generation.call(
                model=self.qwen_model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": user_prompt},
                ],
                result_format="message",
            )
            from http import HTTPStatus
            if response.status_code != HTTPStatus.OK:
                raise RuntimeError(
                    f"Qwen API error: {response.code} - {response.message}"
                )

        raw_text = response.output.choices[0].message.content
        parsed = self._extract_json(raw_text)

        return self._pack_results(
            parsed,
            pairs,
            source="qwen",
            metadata={
                "model": self.qwen_model,
                "request_id": getattr(response, "request_id", None),
            },
        )

    # ------------------------------------------------------------------
    # OpenAI 标注（原逻辑，保持不变）
    # ------------------------------------------------------------------
    def _label_pairs_openai(self, pairs):
        client = self._ensure_openai_client()
        payload = self.build_batch_payload(pairs)

        schema = {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair_index": {"type": "integer"},
                            "preference": {
                                "type": "string",
                                "enum": ["A", "B", "tie"],
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "reason": {"type": "string"},
                        },
                        "required": [
                            "pair_index",
                            "preference",
                            "confidence",
                            "reason",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["labels"],
            "additionalProperties": False,
        }

        response = client.responses.create(
            model=self.openai_model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=(
                "Compare every A/B pair below. The pair_index in your answer must match the input.\n\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "uav_preference_labels",
                    "strict": True,
                    "schema": schema,
                },
                "verbosity": "low",
            },
            store=False,
        )

        raw_text = response.output_text
        parsed = json.loads(raw_text)

        return self._pack_results(
            parsed,
            pairs,
            source="openai",
            metadata={
                "model": self.openai_model,
                "response_id": getattr(response, "id", None),
            },
        )

    # ------------------------------------------------------------------
    # heuristic（调试用，保持原样）
    # ------------------------------------------------------------------
    @staticmethod
    def _heuristic_compare(summary_a, summary_b):
        a = summary_a["end_target"]
        b = summary_b["end_target"]
        ea = summary_a["trajectory_events"]
        eb = summary_b["trajectory_events"]

        # 与 LLM prompt 保持同一目标排序。最后两个项只在前面的目标指标
        # 基本等价时用于偏向更稳定、少恶化的短轨迹。
        key_a = (
            int(a["all_reached"]),
            int(a["num_within_tolerance"]),
            -float(a["max_distance"]),
            -float(a["mean_distance"]),
            -float(a["sum_distance"]),
            int(ea["improving_steps"]),
            -int(ea["worsening_steps"]),
        )
        key_b = (
            int(b["all_reached"]),
            int(b["num_within_tolerance"]),
            -float(b["max_distance"]),
            -float(b["mean_distance"]),
            -float(b["sum_distance"]),
            int(eb["improving_steps"]),
            -int(eb["worsening_steps"]),
        )

        if key_a == key_b:
            return (
                "tie",
                0.55,
                "Both trajectories are essentially equivalent for the target-position task.",
            )

        preference = "A" if key_a > key_b else "B"
        return (
            preference,
            0.90,
            "Debug heuristic preference based only on progress of all UAVs toward (0,0,0).",
        )

    def _label_pairs_heuristic(self, pairs):
        if not self._warned_fallback:
            warnings.warn(
                "LLMPreferenceLabeler is using heuristic_fallback. "
                "This is only for debugging the training pipeline and is NOT equivalent to LLM preference labels. "
                "Set PREFERENCE_LABELER_MODE=qwen and DASHSCOPE_API_KEY for the intended experiment.",
                RuntimeWarning,
            )
            self._warned_fallback = True

        results = []
        for pair in pairs:
            summary_a = self.summarize_segment(pair["segment_a"])
            summary_b = self.summarize_segment(pair["segment_b"])
            preference, confidence, reason = self._heuristic_compare(summary_a, summary_b)
            results.append({
                "preference": preference,
                "confidence": confidence,
                "reason": reason,
                "source": "heuristic_fallback",
                "metadata": {},
            })
        return results

    # ------------------------------------------------------------------
    # 统一入口
    # ------------------------------------------------------------------
    def label_pairs(self, pairs):
        if not pairs:
            return []

        resolved_mode = self._resolve_mode()

        if resolved_mode == "heuristic":
            return self._label_pairs_heuristic(pairs)

        if resolved_mode == "qwen":
            try:
                return self._label_pairs_qwen(pairs)
            except Exception:
                if self.mode == "qwen":
                    raise
                warnings.warn(
                    "Qwen preference labeling failed in auto mode; falling back to debug heuristic labels.",
                    RuntimeWarning,
                )
                return self._label_pairs_heuristic(pairs)

        # openai
        try:
            return self._label_pairs_openai(pairs)
        except Exception:
            if self.mode == "openai":
                raise
            warnings.warn(
                "OpenAI preference labeling failed in auto mode; falling back to debug heuristic labels.",
                RuntimeWarning,
            )
            return self._label_pairs_heuristic(pairs)