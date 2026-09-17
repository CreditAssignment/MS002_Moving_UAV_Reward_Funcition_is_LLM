import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from E_EnvController import EnvController
from LLM_Preference_Labeler import LLMPreferenceLabeler
from Preference_Buffer import PreferenceBuffer
from Preference_Sampler import PreferenceSampler
from Reward_Model import (
    RewardModelEnsemble,
    RunningRewardNormalizer,
    calibrate_reward_normalizer,
    train_reward_model_ensemble,
)
from SAC_Architecture01 import HybridSACActor, HybridSACCritic, DEFAULT_MAX_DELTA
from SAC_ReplayBuffer import SACReplayBuffer


# ============================================================================
# 0. 实验配置
# ============================================================================
SEED = 42
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

MAX_NUM_UAV = 8
NUM_ITERATIONS = 100
TOTAL_TIME_STEPS = 500

# Hybrid SAC
REPLAY_CAPACITY = 50000
BATCH_SIZE = 64
UPDATE_EVERY = 10
EPOCHS_EVERY_TRAINING = 4
GAMMA = 0.99
ALPHA = 0.1
TAU = 0.01
MAX_GRAD_NORM = 10.0
ACTOR_LR = 5e-5
CRITIC_LR = 5e-5
MAX_DELTA = DEFAULT_MAX_DELTA

# Preference / Reward Model
WARMUP_STEPS = 200
PREFERENCE_QUERY_EVERY = 500
PREFERENCE_PAIRS_PER_QUERY = 4
PREFERENCE_CANDIDATE_PAIRS = 24
PREFERENCE_SEGMENT_LENGTH = 5
PREFERENCE_BUFFER_CAPACITY = 10000
MIN_PREFERENCE_RECORDS = 8

RM_ENSEMBLE_SIZE = 3
RM_LR = 1e-4
RM_TRAIN_EPOCHS = 25
RM_BATCH_SIZE = 16
RM_REWARD_CLIP = 5.0
RM_CALIBRATION_SAMPLES = 1024

# LLMPreferenceLabeler:
#   auto     -> 有 OPENAI_API_KEY 时调用 LLM，否则退化为明确标记的调试 heuristic
#   openai   -> 强制实际调用 OpenAI；失败即终止
#   heuristic-> 仅用于本地调试闭环
PREFERENCE_LABELER_MODE = "qwen"

# 固定 validation，不使用 Reward Model reward 选最优模型。
VALIDATE_EVERY_EPISODES = 5
VALIDATION_STEPS = 500
VALIDATION_RANDOM_SEEDS = [101, 202, 303, 404]

# 输出目录
MODEL_DIR = Path("./R_models")
RECORD_DIR = Path("./R_records")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RECORD_DIR.mkdir(parents=True, exist_ok=True)

EPISODE_CSV = RECORD_DIR / "episode_metrics.csv"
PREFERENCE_JSONL = RECORD_DIR / "preference_labels.jsonl"
PREFERENCE_PICKLE = RECORD_DIR / "preference_buffer.pkl"


# ============================================================================
# 1. 工具函数
# ============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.lerp_(source_param, tau)


def save_checkpoint(
    path,
    actor,
    critic_1,
    critic_2,
    target_critic_1,
    target_critic_2,
    reward_ensemble,
    reward_normalizer,
    global_step,
    episode,
    validation_metrics=None,
):
    torch.save(
        {
            "actor": actor.state_dict(),
            "critic_1": critic_1.state_dict(),
            "critic_2": critic_2.state_dict(),
            "target_critic_1": target_critic_1.state_dict(),
            "target_critic_2": target_critic_2.state_dict(),
            "reward_ensemble": reward_ensemble.state_dict(),
            "reward_model_version": reward_ensemble.version,
            "reward_normalizer": reward_normalizer.state_dict(),
            "global_step": int(global_step),
            "episode": int(episode),
            "validation_metrics": validation_metrics,
            "hyperparameters": {
                "GAMMA": GAMMA,
                "ALPHA": ALPHA,
                "TAU": TAU,
                "MAX_GRAD_NORM": MAX_GRAD_NORM,
                "MAX_DELTA": tuple(float(v) for v in MAX_DELTA),
                "PREFERENCE_SEGMENT_LENGTH": PREFERENCE_SEGMENT_LENGTH,
                "RM_ENSEMBLE_SIZE": RM_ENSEMBLE_SIZE,
            },
        },
        path,
    )


def append_episode_csv(row):
    file_exists = EPISODE_CSV.exists()
    with EPISODE_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def compact_target_metrics(metrics):
    return {
        "target_position": list(metrics["target_position"]),
        "tolerance": float(metrics["tolerance"]),
        "sum_distance": float(metrics["sum_distance"]),
        "mean_distance": float(metrics["mean_distance"]),
        "max_distance": float(metrics["max_distance"]),
        "num_within_tolerance": int(metrics["num_within_tolerance"]),
        "all_reached": bool(metrics["all_reached"]),
    }


def run_validation(actor, env_controller):
    """
    用固定 sanity-check 场景和 deterministic Actor 评估。

    验证完全不使用 Reward Model reward，也不使用网络连通/覆盖指标，
    只检查所有 UAV 是否真正趋近目标 (0,0,0)。
    """
    actor_was_training = actor.training
    actor.eval()

    monitor = env_controller.build_monitor_scene()
    state_np = np.asarray(
        monitor.getState(max_num_uav=MAX_NUM_UAV),
        dtype=np.float32,
    )

    first_success_step = None
    steps_executed = 0

    with torch.no_grad():
        for step in range(VALIDATION_STEPS):
            select_able_mask = monitor.getSelectAbleMask()
            if not any(select_able_mask):
                break

            state = torch.tensor(
                state_np,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(0)

            _, uav_slot, delta = actor.deterministic_action(
                state,
                select_able_mask=select_able_mask,
            )

            next_state_np, env_done, info = monitor.step(
                uav_slot=int(uav_slot.item()),
                delta=delta,
                max_num_uav=MAX_NUM_UAV,
                terminate_on_connected=False,
                terminate_on_target=True,
                time_step_counter=step,
            )
            state_np = np.asarray(next_state_np, dtype=np.float32)
            steps_executed = step + 1

            if info["after"]["target"]["all_reached"]:
                if first_success_step is None:
                    first_success_step = step + 1

            if env_done:
                break

    target_metrics = compact_target_metrics(monitor.getTargetMetrics())
    summary = {
        **target_metrics,
        "steps_executed": int(steps_executed),
        "first_success_step": (
            VALIDATION_STEPS + 1
            if first_success_step is None
            else int(first_success_step)
        ),
    }

    if actor_was_training:
        actor.train()

    # 字典序和 LLM 的目标保持一致：全部到达 > 到达数量 > 最远 UAV 距离
    # > 平均/总距离 > 完成速度。
    key = (
        int(summary["all_reached"]),
        int(summary["num_within_tolerance"]),
        -float(summary["max_distance"]),
        -float(summary["mean_distance"]),
        -float(summary["sum_distance"]),
        -float(summary["first_success_step"]),
    )
    return summary, key


def update_hybrid_sac(
    actor,
    critic_1,
    critic_2,
    target_critic_1,
    target_critic_2,
    optimizer_actor,
    optimizer_critic_1,
    optimizer_critic_2,
    reward_ensemble,
    reward_normalizer,
    replay_buffer,
):
    metrics = {
        "actor_loss": None,
        "critic_1_loss": None,
        "critic_2_loss": None,
        "mean_rm_reward": None,
    }

    for epoch in range(EPOCHS_EVERY_TRAINING):
        batch = replay_buffer.sample(BATCH_SIZE, device=DEVICE)
        states = batch["states"]
        select_able_masks = batch["select_able_masks"]
        uav_slots = batch["uav_slots"]
        deltas = batch["deltas"]
        next_states = batch["next_states"]
        next_select_able_masks = batch["next_select_able_masks"]
        dones = batch["dones"]

        # ------------------------------------------------------------------
        # Lazy reward relabeling: 每次采样都用最新 Reward Model 重新打分。
        # ------------------------------------------------------------------
        reward_ensemble.eval()

        with torch.no_grad():

            # --------------------------------------------------------------
            # 1. 使用当前 Reward Model 对 transition 重新计算 reward
            # --------------------------------------------------------------
            raw_rewards = reward_ensemble(
                states,
                uav_slots,
                deltas,
                next_states,
            )

            rewards = reward_normalizer.normalize(
                raw_rewards,
                update=True,
            )

            # --------------------------------------------------------------
            # 2. 初始化 next_soft_value
            #
            # terminal transition 的 next_soft_value 永远保持为 0。
            # 只有 non-terminal transition 才需要计算 V(s_{t+1})。
            # --------------------------------------------------------------
            next_soft_value = torch.zeros_like(rewards)

            # dones:
            #   0.0 -> non-terminal
            #   1.0 -> terminal
            nonterminal_index = dones < 0.5

            # --------------------------------------------------------------
            # 3. 只处理 non-terminal transition
            # --------------------------------------------------------------
            if nonterminal_index.any():

                nonterminal_next_states = next_states[nonterminal_index]

                nonterminal_next_select_able_masks = (
                    next_select_able_masks[nonterminal_index]
                )

                # ----------------------------------------------------------
                # 安全检查：
                # non-terminal state 理论上必须至少存在一个合法 UAV。
                #
                # 如果这里出现全 False，说明环境的 done / action mask
                # 定义存在不一致，而不是正常 terminal 情况。
                # ----------------------------------------------------------
                if (~nonterminal_next_select_able_masks).all(dim=-1).any():
                    raise RuntimeError(
                        "A non-terminal transition has no selectable UAV."
                    )

                # ----------------------------------------------------------
                # 4. 只让 Actor 为 non-terminal next states 产生动作
                # ----------------------------------------------------------
                next_uav_probs, next_log_probs, next_deltas_all = (
                    actor.sample_all(
                        state=nonterminal_next_states,
                        select_able_mask=nonterminal_next_select_able_masks,
                    )
                )

                # ----------------------------------------------------------
                # 5. Target Critics 也只计算 non-terminal next states
                # ----------------------------------------------------------
                target_q1_all = target_critic_1.q_all_uavs(
                    state=nonterminal_next_states,
                    delta_all=next_deltas_all,
                )

                target_q2_all = target_critic_2.q_all_uavs(
                    state=nonterminal_next_states,
                    delta_all=next_deltas_all,
                )

                target_min_q_all = torch.minimum(
                    target_q1_all,
                    target_q2_all,
                )

                # ----------------------------------------------------------
                # 6. Hybrid SAC 的 next-state soft value
                # ----------------------------------------------------------
                nonterminal_soft_value = (
                        next_uav_probs
                        * (
                                target_min_q_all
                                - ALPHA * next_log_probs
                        )
                ).sum(dim=-1)

                # 只写回 non-terminal 位置。
                # terminal 位置继续保持 0。
                next_soft_value[nonterminal_index] = (
                    nonterminal_soft_value
                )

            # --------------------------------------------------------------
            # 7. TD target
            # --------------------------------------------------------------
            q_target = (
                    rewards
                    + GAMMA
                    * (1.0 - dones)
                    * next_soft_value
            )

        # ------------------------------------------------------------------
        # Twin Critic update
        # ------------------------------------------------------------------
        current_q1 = critic_1(states, uav_slots, deltas)
        current_q2 = critic_2(states, uav_slots, deltas)
        critic_1_loss = F.mse_loss(current_q1, q_target)
        critic_2_loss = F.mse_loss(current_q2, q_target)

        optimizer_critic_1.zero_grad(set_to_none=True)
        optimizer_critic_2.zero_grad(set_to_none=True)
        (critic_1_loss + critic_2_loss).backward()
        torch.nn.utils.clip_grad_norm_(critic_1.parameters(), MAX_GRAD_NORM)
        torch.nn.utils.clip_grad_norm_(critic_2.parameters(), MAX_GRAD_NORM)
        optimizer_critic_1.step()
        optimizer_critic_2.step()

        # ------------------------------------------------------------------
        # Actor update
        # ------------------------------------------------------------------
        critic_1.requires_grad_(False)
        critic_2.requires_grad_(False)

        uav_probs, total_log_probs, deltas_all = actor.sample_all(
            state=states,
            select_able_mask=select_able_masks,
        )
        q1_policy_all = critic_1.q_all_uavs(states, deltas_all)
        q2_policy_all = critic_2.q_all_uavs(states, deltas_all)
        min_q_policy_all = torch.minimum(q1_policy_all, q2_policy_all)

        actor_loss = (
            uav_probs * (ALPHA * total_log_probs - min_q_policy_all)
        ).sum(dim=-1).mean()

        optimizer_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
        optimizer_actor.step()

        critic_1.requires_grad_(True)
        critic_2.requires_grad_(True)

        if epoch + 1 == EPOCHS_EVERY_TRAINING:
            soft_update(target_critic_1, critic_1, TAU)
            soft_update(target_critic_2, critic_2, TAU)

        metrics = {
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_1_loss": float(critic_1_loss.detach().cpu()),
            "critic_2_loss": float(critic_2_loss.detach().cpu()),
            "mean_rm_reward": float(rewards.mean().detach().cpu()),
        }

    return metrics


# ============================================================================
# 2. 初始化
# ============================================================================
set_seed(SEED)
print(f"Training device: {DEVICE}")

env_controller = EnvController()
probe_monitor = env_controller.build_monitor_scene()
probe_state = probe_monitor.getState(max_num_uav=MAX_NUM_UAV)
state_dim = len(probe_state)
print(f"state_dim = {state_dim}")

actor = HybridSACActor(
    state_dim=state_dim,
    num_uav=MAX_NUM_UAV,
    max_delta=MAX_DELTA,
).to(DEVICE)
critic_1 = HybridSACCritic(
    state_dim=state_dim,
    num_uav=MAX_NUM_UAV,
    max_delta=MAX_DELTA,
).to(DEVICE)
critic_2 = HybridSACCritic(
    state_dim=state_dim,
    num_uav=MAX_NUM_UAV,
    max_delta=MAX_DELTA,
).to(DEVICE)
target_critic_1 = HybridSACCritic(
    state_dim=state_dim,
    num_uav=MAX_NUM_UAV,
    max_delta=MAX_DELTA,
).to(DEVICE)
target_critic_2 = HybridSACCritic(
    state_dim=state_dim,
    num_uav=MAX_NUM_UAV,
    max_delta=MAX_DELTA,
).to(DEVICE)

# 重要：Target Critic 必须从对应 Critic 的同一初始参数开始。
target_critic_1.load_state_dict(critic_1.state_dict())
target_critic_2.load_state_dict(critic_2.state_dict())
target_critic_1.requires_grad_(False)
target_critic_2.requires_grad_(False)

optimizer_actor = optim.Adam(actor.parameters(), lr=ACTOR_LR)
optimizer_critic_1 = optim.Adam(critic_1.parameters(), lr=CRITIC_LR)
optimizer_critic_2 = optim.Adam(critic_2.parameters(), lr=CRITIC_LR)

reward_ensemble = RewardModelEnsemble(  # 一次创建多个独立的奖励模型，而不是只有一个 Reward Model。
    state_dim=state_dim,
    num_uav=MAX_NUM_UAV,
    ensemble_size=RM_ENSEMBLE_SIZE,
    max_delta=MAX_DELTA,
).to(DEVICE)
reward_optimizers = [
    optim.Adam(model.parameters(), lr=RM_LR)
    for model in reward_ensemble.models
]
reward_normalizer = RunningRewardNormalizer(clip_value=RM_REWARD_CLIP)

replay_buffer = SACReplayBuffer(capacity=REPLAY_CAPACITY)
preference_buffer = PreferenceBuffer(capacity=PREFERENCE_BUFFER_CAPACITY)
preference_sampler = PreferenceSampler(
    segment_length=PREFERENCE_SEGMENT_LENGTH,
    candidate_pairs=PREFERENCE_CANDIDATE_PAIRS,
)
llm_labeler = LLMPreferenceLabeler(mode=PREFERENCE_LABELER_MODE)

reward_model_ready = False
global_step = 0
best_validation_key = None
best_validation_metrics = None
last_sac_metrics = {}
last_rm_metrics = {}


# ============================================================================
# 3. 开始闭环训练
# ============================================================================
for episode in range(NUM_ITERATIONS):
    monitor = env_controller.build_monitor_scene()
    state_np = np.asarray(
        monitor.getState(max_num_uav=MAX_NUM_UAV),
        dtype=np.float32,
    )

    first_target_reached_step = None

    for time_step in range(TOTAL_TIME_STEPS):
        # if time_step % 100 == 0:
        #     for uavID, uav in monitor.uav_set.items():
        #         print(f"\033[91mUAV{uavID} {uav.uav_position}\033[0m")

        select_able_mask = monitor.getSelectAbleMask()
        if not any(select_able_mask):
            print(f"Episode {episode}: no active UAV is available.")
            break

        state = torch.tensor(
            state_np,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        # --------------------------------------------------------------
        # Actor exploration
        # --------------------------------------------------------------
        with torch.no_grad():
            _, uav_slot, delta, _ = actor.sample(
                state=state,
                select_able_mask=select_able_mask,
            )

        # --------------------------------------------------------------
        # Environment: 不再返回人工 reward，只返回 transition 结果
        # --------------------------------------------------------------
        next_state_np, env_done, info = monitor.step(
            uav_slot=int(uav_slot.item()),
            delta=delta,
            max_num_uav=MAX_NUM_UAV,
            terminate_on_connected=False,
            terminate_on_target=True,
            time_step_counter=time_step,
        )
        next_state_np = np.asarray(next_state_np, dtype=np.float32)
        next_select_able_mask = monitor.getSelectAbleMask()

        reached_time_limit = time_step == TOTAL_TIME_STEPS - 1
        done = bool(env_done or reached_time_limit)

        replay_buffer.push(
            state=state_np,
            select_able_mask=select_able_mask,
            uav_slot=uav_slot,
            delta=delta,
            next_state=next_state_np,
            next_select_able_mask=next_select_able_mask,
            done=done,
            info=info,
            episode_id=episode,
            time_step=time_step,
        )

        global_step += 1

        target_after = info["after"]["target"]
        if target_after["all_reached"] and first_target_reached_step is None:  # 记录第一次到达的时间步
            first_target_reached_step = time_step + 1

        # --------------------------------------------------------------
        # Active preference learning:
        # 同一状态分支 -> trajectory pair -> LLM -> Preference Buffer -> RM
        # --------------------------------------------------------------
        should_query_preference = (
            global_step >= WARMUP_STEPS
            and global_step % PREFERENCE_QUERY_EVERY == 0
        )

        if should_query_preference:
            print(
                f"[Preference Query] global_step={global_step}, "
                f"current preferences={len(preference_buffer)}"
            )

            pairs = preference_sampler.select_branch_pairs(
                monitor=monitor,
                actor=actor,
                max_num_uav=MAX_NUM_UAV,
                device=DEVICE,
                num_pairs=PREFERENCE_PAIRS_PER_QUERY,
                reward_model_ensemble=(reward_ensemble if reward_model_ready else None),
            )

            if pairs:
                label_results = llm_labeler.label_pairs(pairs)
                for pair, label_result in zip(pairs, label_results):
                    preference_id = preference_buffer.add_from_label_result(
                        pair,
                        label_result,
                    )
                    record = preference_buffer.records[-1]
                    preference_buffer.append_jsonl(PREFERENCE_JSONL, record)
                    print(
                        f"  preference#{preference_id}: "
                        f"{label_result['preference']} "
                        f"conf={label_result['confidence']:.2f} "
                        f"source={label_result['source']}"
                    )

                preference_buffer.save_pickle(PREFERENCE_PICKLE)

            if len(preference_buffer) >= MIN_PREFERENCE_RECORDS:
                last_rm_metrics = train_reward_model_ensemble(
                    ensemble=reward_ensemble,
                    preference_buffer=preference_buffer,
                    optimizers=reward_optimizers,
                    device=DEVICE,
                    epochs=RM_TRAIN_EPOCHS,
                    batch_size=RM_BATCH_SIZE,
                    grad_clip=MAX_GRAD_NORM,
                )
                reward_model_ready = True

                # RM 更新后旧尺度统计已经失效，用当前 RM 对历史 replay 重新校准。
                reward_normalizer.reset()
                calibrate_reward_normalizer(
                    ensemble=reward_ensemble,
                    reward_normalizer=reward_normalizer,
                    replay_buffer=replay_buffer,
                    device=DEVICE,
                    num_samples=RM_CALIBRATION_SAMPLES,
                )
                print(f"[Reward Model] {last_rm_metrics}")

        # --------------------------------------------------------------
        # Hybrid SAC update：仅在 RM 已有足够偏好监督后启动
        # --------------------------------------------------------------
        if (
            reward_model_ready
            and replay_buffer.can_sample(BATCH_SIZE)
            and global_step % UPDATE_EVERY == 0
        ):
            last_sac_metrics = update_hybrid_sac(
                actor=actor,
                critic_1=critic_1,
                critic_2=critic_2,
                target_critic_1=target_critic_1,
                target_critic_2=target_critic_2,
                optimizer_actor=optimizer_actor,
                optimizer_critic_1=optimizer_critic_1,
                optimizer_critic_2=optimizer_critic_2,
                reward_ensemble=reward_ensemble,
                reward_normalizer=reward_normalizer,
                replay_buffer=replay_buffer,
            )

        state_np = next_state_np
        if done:
            break

    # ========================================================================
    # 4. Episode objective metrics（不使用 RM reward）
    # ========================================================================
    final_target = compact_target_metrics(monitor.getTargetMetrics())
    episode_row = {
        "episode": episode,
        "global_step": global_step,
        "rm_ready": int(reward_model_ready),
        "rm_version": reward_ensemble.version,
        "num_preferences": len(preference_buffer),
        "preference_sources": json.dumps(
            preference_buffer.source_counts(),
            ensure_ascii=False,
        ),
        "all_reached": int(final_target["all_reached"]),
        "num_within_tolerance": final_target["num_within_tolerance"],
        "sum_distance": final_target["sum_distance"],
        "mean_distance": final_target["mean_distance"],
        "max_distance": final_target["max_distance"],
        "first_target_reached_step": (
            -1
            if first_target_reached_step is None
            else first_target_reached_step
        ),
        "actor_loss": last_sac_metrics.get("actor_loss"),
        "critic_1_loss": last_sac_metrics.get("critic_1_loss"),
        "critic_2_loss": last_sac_metrics.get("critic_2_loss"),
        "mean_rm_reward": last_sac_metrics.get("mean_rm_reward"),
        "rm_loss": last_rm_metrics.get("loss"),
        "rm_pair_accuracy": last_rm_metrics.get("accuracy"),
    }
    append_episode_csv(episode_row)

    print(
        f"Episode {episode}/{NUM_ITERATIONS} | step={global_step} | "
        f"reached={final_target['num_within_tolerance']}/{MAX_NUM_UAV} | "
        f"mean_dist={final_target['mean_distance']:.2f} | "
        f"max_dist={final_target['max_distance']:.2f} | "
        f"preferences={len(preference_buffer)} | "
        f"rm_v={reward_ensemble.version}"
    )

    # ========================================================================
    # 5. 固定验证集选最优模型
    # ========================================================================
    if (episode + 1) % VALIDATE_EVERY_EPISODES == 0:
        validation_metrics, validation_key = run_validation(
            actor,
            env_controller,
        )
        print(
            "[Validation] "
            f"success={validation_metrics['all_reached']} | "
            f"reached={validation_metrics['num_within_tolerance']}/{MAX_NUM_UAV} | "
            f"mean_dist={validation_metrics['mean_distance']:.2f} | "
            f"max_dist={validation_metrics['max_distance']:.2f} | "
            f"first_success={validation_metrics['first_success_step']}"
        )

        if best_validation_key is None or validation_key > best_validation_key:
            best_validation_key = validation_key
            best_validation_metrics = validation_metrics
            save_checkpoint(
                MODEL_DIR / "best_model_by_fixed_validation.pth",
                actor,
                critic_1,
                critic_2,
                target_critic_1,
                target_critic_2,
                reward_ensemble,
                reward_normalizer,
                global_step,
                episode,
                validation_metrics=validation_metrics,
            )
            with (RECORD_DIR / "best_validation_metrics.json").open(
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(validation_metrics, f, ensure_ascii=False, indent=2)

    # 周期性保存恢复点。
    if (episode + 1) % 25 == 0:
        save_checkpoint(
            MODEL_DIR / "latest_training_checkpoint.pth",
            actor,
            critic_1,
            critic_2,
            target_critic_1,
            target_critic_2,
            reward_ensemble,
            reward_normalizer,
            global_step,
            episode,
            validation_metrics=best_validation_metrics,
        )


# ============================================================================
# 6. 训练结束保存
# ============================================================================
save_checkpoint(
    MODEL_DIR / "final_training_checkpoint.pth",
    actor,
    critic_1,
    critic_2,
    target_critic_1,
    target_critic_2,
    reward_ensemble,
    reward_normalizer,
    global_step,
    NUM_ITERATIONS - 1,
    validation_metrics=best_validation_metrics,
)
preference_buffer.save_pickle(PREFERENCE_PICKLE)
print("END")
