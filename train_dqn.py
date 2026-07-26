"""
train_dqn.py — Huấn luyện DQN thuần (Stable-Baselines3) cho UAVMapEnv.

Đặc điểm:
  - Curriculum Learning 3 giai đoạn: map_easy → map_medium → map_hard
    (chuyển khi rolling success rate 200 episode >= 70%)
  - Early Stopping: dừng khi success rate >= 85% và cải thiện < 2%
    trong 500 episode liên tiếp (patience), hoặc đạt 300,000 timesteps.
  - EvalCallback: checkpoint model tốt nhất (success rate cao nhất) → models/dqn_best.zip
  - EpisodeLoggerCallback: ghi CSV logs/train_log.csv đủ dữ liệu để vẽ 4 biểu đồ.
  - Reproducible: seed=42 cho numpy, torch, env.

Cấu trúc thư mục đầu ra:
  logs/train_log.csv      — log mỗi episode
  models/dqn_best.zip     — checkpoint tốt nhất
  models/dqn_final.zip    — model cuối training
"""

import os
import csv
import random
import numpy as np
import torch
from collections import deque

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from uav_env.uav_map_env import UAVMapEnv

# ───────────────────────────────────────────────────────────────
# Cấu hình chung
# ───────────────────────────────────────────────────────────────
SEED = 42
MAX_TIMESTEPS = 300_000

TRAIN_MAPS = [
    "maps/map_easy.png",
    "maps/map_medium.png",
    "maps/map_hard.png",
]
EVAL_MAP = "maps/map_easy.png"   # eval trên map dễ để ổn định benchmark

CURRICULUM_WINDOW = 200          # cửa sổ rolling để đánh giá chuyển stage
CURRICULUM_THRESHOLD = 0.70      # success rate để chuyển stage

EARLY_STOP_WINDOW = 200
EARLY_STOP_THRESHOLD = 0.85      # success rate tối thiểu để xét early stop
EARLY_STOP_MIN_IMPROVEMENT = 0.02
EARLY_STOP_PATIENCE = 500        # số episode liên tiếp không cải thiện để dừng

EVAL_FREQ = 5_000                # timesteps giữa các lần eval
N_EVAL_EPISODES = 20

LOG_DIR  = "logs"
MODEL_DIR = "models"
LOG_CSV  = os.path.join(LOG_DIR, "train_log.csv")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_best")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_final")

# Hyperparameters (yêu cầu đề bài)
HYPERPARAMS = dict(
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,
    exploration_fraction=0.3,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    max_grad_norm=10,
    policy_kwargs=dict(net_arch=[256, 256]),
)


def set_global_seed(seed: int):
    """Cố định seed cho numpy, Python random, và PyTorch để đảm bảo reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(map_path: str, fixed_start_goal: bool = False, seed: int = SEED):
    """
    Tạo UAVMapEnv đã bọc trong Monitor, render_mode=None (bắt buộc khi train).
    Monitor wrapper giúp SB3 tự động log episode reward/length,
    đồng thời info dict được giữ nguyên để callback đọc 'outcome'.
    """
    env = UAVMapEnv(
        map_path=map_path,
        fixed_start_goal=fixed_start_goal,
        render_mode=None,
    )
    env.reset(seed=seed)
    env = Monitor(env)
    return env


# ───────────────────────────────────────────────────────────────
# Callback 1: EpisodeLoggerCallback — ghi CSV mỗi episode
# ───────────────────────────────────────────────────────────────
class EpisodeLoggerCallback(BaseCallback):
    """
    Ghi log mỗi episode vào file CSV với đầy đủ 7 cột:
      episode, timestep, total_reward, episode_length,
      outcome, current_map_stage, rolling_success_rate_200

    Dữ liệu này phục vụ trực tiếp việc vẽ:
      (a) Reward/episode (raw + moving average)
      (e) Rolling success rate
      (g) Outcome distribution
    """

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_num = 0
        self.outcome_history: deque = deque(maxlen=EARLY_STOP_WINDOW)

        # Header CSV
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "timestep", "total_reward", "episode_length",
                "outcome", "current_map_stage", "rolling_success_rate_200"
            ])

        # Tham chiếu tới CurriculumCallback để lấy stage hiện tại
        self._curriculum_cb = None

    def _on_step(self) -> bool:
        # SB3 gọi _on_step() sau mỗi env step. Kiểm tra episode đã kết thúc chưa.
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            self.episode_num += 1

            # Lấy dữ liệu từ Monitor wrapper (episode_reward, episode_length)
            ep_info = info.get("episode", {})
            total_reward   = ep_info.get("r", 0.0)
            episode_length = ep_info.get("l", 0)

            # outcome đến từ UAVMapEnv._get_info()
            # Monitor wrapper đưa info gốc vào key "terminal_observation"
            # nhưng outcome sẽ có trong info trực tiếp
            outcome = info.get("outcome", "timeout")

            self.outcome_history.append(1 if outcome == "success" else 0)
            rolling_sr = (sum(self.outcome_history) / len(self.outcome_history)
                          if self.outcome_history else 0.0)

            map_stage = (self._curriculum_cb.current_stage_idx
                         if self._curriculum_cb else 0)

            # Ghi CSV
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.episode_num,
                    self.num_timesteps,
                    round(total_reward, 3),
                    episode_length,
                    outcome,
                    map_stage,
                    round(rolling_sr, 4),
                ])

            # In tiến độ mỗi 100 episode
            if self.episode_num % 100 == 0:
                stage_name = TRAIN_MAPS[map_stage] if map_stage < len(TRAIN_MAPS) else "N/A"
                print(
                    f"[EpisodeLogger] Ep {self.episode_num:>5} | "
                    f"Steps {self.num_timesteps:>7} | "
                    f"Rolling SR-200: {rolling_sr:.1%} | "
                    f"Stage: {os.path.basename(stage_name)}"
                )

        return True


# ───────────────────────────────────────────────────────────────
# Callback 2: CurriculumCallback — chuyển map khi đủ điều kiện
# ───────────────────────────────────────────────────────────────
class CurriculumCallback(BaseCallback):
    """
    Curriculum Learning qua 3 giai đoạn bản đồ (dễ → trung bình → khó).

    Điều kiện chuyển stage:
      Rolling success rate trên 200 episode gần nhất >= 70%.

    Cách chuyển map:
      Gọi env.env_method("_load_map", new_map_path) để nạp map mới
      mà không cần tắt / khởi tạo lại toàn bộ env wrapper.
    """

    def __init__(self, train_env, verbose: int = 1):
        super().__init__(verbose)
        self.train_env = train_env
        self.current_stage_idx = 0
        self.outcome_history: deque = deque(maxlen=CURRICULUM_WINDOW)

    def _on_step(self) -> bool:
        if self.current_stage_idx >= len(TRAIN_MAPS) - 1:
            return True  # Đã ở stage cuối, không cần chuyển nữa

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if not done:
                continue
            outcome = info.get("outcome", "timeout")
            self.outcome_history.append(1 if outcome == "success" else 0)

        # Kiểm tra điều kiện chuyển stage sau khi có đủ 200 episode
        if len(self.outcome_history) < CURRICULUM_WINDOW:
            return True

        rolling_sr = sum(self.outcome_history) / len(self.outcome_history)
        if rolling_sr >= CURRICULUM_THRESHOLD:
            self.current_stage_idx += 1
            new_map = TRAIN_MAPS[self.current_stage_idx]
            self.outcome_history.clear()

            # Đổi bản đồ cho training env
            try:
                self.train_env.env_method("_load_map", new_map)
                self.train_env.env_method("reset")
            except Exception as e:
                # Fallback: in cảnh báo nếu không đổi được ngay
                print(f"[Curriculum] WARNING: Could not hot-swap map: {e}")

            print(
                f"\n[Curriculum] >>> Stage {self.current_stage_idx}: "
                f"Chuyen sang {os.path.basename(new_map)} "
                f"(SR-200={rolling_sr:.1%}) <<<\n"
            )

        return True


# ───────────────────────────────────────────────────────────────
# Callback 3: EarlyStoppingCallback — dừng khi hội tụ
# ───────────────────────────────────────────────────────────────
class EarlyStoppingCallback(BaseCallback):
    """
    Dừng sớm khi mô hình hội tụ theo 2 điều kiện kết hợp:
      1. Rolling success rate (200 ep) >= 85%
      2. Cải thiện so với 200 episode trước đó < 2%
         liên tục trong 500 episode (patience)

    Ngoài ra có giới hạn an toàn: dừng tối đa sau MAX_TIMESTEPS.
    """

    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.outcome_history: deque = deque(maxlen=EARLY_STOP_WINDOW * 2)
        self.patience_counter = 0
        self.prev_rolling_sr = 0.0
        self.episode_num = 0

    def _on_step(self) -> bool:
        # Kiểm tra giới hạn timesteps tuyệt đối
        if self.num_timesteps >= MAX_TIMESTEPS:
            print(
                f"\n[EarlyStopping] Dừng training: đã đạt giới hạn tối đa "
                f"{MAX_TIMESTEPS:,} timesteps.\n"
                f"  Episode: {self.episode_num} | "
                f"Rolling SR-200: {self._get_rolling_sr():.1%}"
            )
            return False

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if not done:
                continue
            self.episode_num += 1
            outcome = info.get("outcome", "timeout")
            self.outcome_history.append(1 if outcome == "success" else 0)

            # Cần ít nhất đủ 2 cửa sổ để tính cải thiện
            if len(self.outcome_history) < EARLY_STOP_WINDOW * 2:
                continue

            rolling_sr = self._get_rolling_sr()

            # Tính cải thiện: so sánh 200 ep gần nhất với 200 ep trước đó
            recent   = list(self.outcome_history)[-EARLY_STOP_WINDOW:]
            previous = list(self.outcome_history)[:EARLY_STOP_WINDOW]
            sr_recent   = sum(recent)   / EARLY_STOP_WINDOW
            sr_previous = sum(previous) / EARLY_STOP_WINDOW
            improvement = sr_recent - sr_previous

            if rolling_sr >= EARLY_STOP_THRESHOLD and improvement < EARLY_STOP_MIN_IMPROVEMENT:
                self.patience_counter += 1
            else:
                self.patience_counter = 0

            if self.patience_counter >= EARLY_STOP_PATIENCE:
                print(
                    f"\n[EarlyStopping] Hội tụ! Dừng training sớm.\n"
                    f"  Lý do: SR={rolling_sr:.1%} >= {EARLY_STOP_THRESHOLD:.0%} "
                    f"và cải thiện={improvement:.2%} < {EARLY_STOP_MIN_IMPROVEMENT:.0%} "
                    f"trong {EARLY_STOP_PATIENCE} episode liên tiếp.\n"
                    f"  Episode: {self.episode_num} | Timestep: {self.num_timesteps:,}"
                )
                return False

        return True

    def _get_rolling_sr(self) -> float:
        if not self.outcome_history:
            return 0.0
        recent = list(self.outcome_history)[-EARLY_STOP_WINDOW:]
        return sum(recent) / len(recent)


# ───────────────────────────────────────────────────────────────
# Custom EvalCallback — success rate làm tiêu chí lưu model tốt nhất
# ───────────────────────────────────────────────────────────────
class SuccessRateEvalCallback(EvalCallback):
    """
    Mở rộng EvalCallback của SB3 để sử dụng success rate (thay vì mean_reward)
    làm tiêu chí chọn model tốt nhất.

    EvalCallback tiêu chuẩn chỉ lưu model khi mean_reward cao hơn, nhưng với
    UAV env reward rất nhiễu (vị trí random), success rate ổn định hơn.

    Sử dụng evaluate_policy của SB3 với return_episode_rewards=True để lấy
    episode_rewards và episode_lengths, sau đó đọc is_success từ info trực tiếp.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_success_rate = -1.0

    def _on_step(self) -> bool:
        # Gọi _on_step của EvalCallback gốc (tính mean_reward, render, v.v.)
        result = super()._on_step()

        # Chỉ đánh giá success rate tại các bước eval định kỳ
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sr = self._compute_success_rate()
            print(f"[EvalCallback] Eval SR: {sr:.1%} (best: {self.best_success_rate:.1%})")

            if sr > self.best_success_rate:
                self.best_success_rate = sr
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "dqn_best"))
                    print(f"[EvalCallback] Lưu model tốt nhất: SR={sr:.1%}")

        return result

    def _compute_success_rate(self) -> float:
        """
        Chạy N_EVAL_EPISODES trên eval_env (VecEnv) và tính success rate.
        Dùng API VecEnv chuẩn:
          - reset() → obs (numpy array shape (n_envs, obs_dim))
          - step()  → (obs, rewards, dones, infos) — 4-tuple
        """
        n_success = 0
        n_episodes = 0

        # VecEnv reset trả về obs trực tiếp (không có info)
        obs = self.eval_env.reset()

        ep_count = 0
        while ep_count < N_EVAL_EPISODES:
            action, _ = self.model.predict(obs, deterministic=True)
            # VecEnv step → 4-tuple (obs, rewards, dones, infos)
            obs, rewards, dones, infos = self.eval_env.step(action)

            for done, info in zip(dones, infos):
                if done:
                    ep_count += 1
                    # SB3 VecEnv đưa terminal info vào key "terminal_observation"
                    # outcome nằm trong info gốc của env
                    outcome = info.get("outcome", "timeout")
                    if outcome == "success" or info.get("is_success", False):
                        n_success += 1

        return n_success / N_EVAL_EPISODES if N_EVAL_EPISODES > 0 else 0.0


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  UAV PATH PLANNING — DQN Training with Curriculum Learning")
    print("=" * 70)

    # Tạo thư mục đầu ra (bắt buộc trước khi SB3 dùng)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "eval_results"), exist_ok=True)

    # Cố định seed toàn cục
    set_global_seed(SEED)

    # ── Khởi tạo môi trường ──────────────────────────────────────
    print(f"\n[Setup] Khởi tạo training env: {TRAIN_MAPS[0]}")
    train_env = make_env(TRAIN_MAPS[0], fixed_start_goal=False, seed=SEED)

    print(f"[Setup] Khởi tạo eval env: {EVAL_MAP}")
    eval_env = make_env(EVAL_MAP, fixed_start_goal=False, seed=SEED + 1)

    # ── Khởi tạo model DQN ──────────────────────────────────────
    print("\n[Setup] Khởi tạo DQN model...")
    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=0,
        tensorboard_log=None,   # Tắt TensorBoard, dùng CSV logger tự viết
        **HYPERPARAMS,
    )

    print(f"  Policy net: {HYPERPARAMS['policy_kwargs']['net_arch']}")
    print(f"  Buffer size: {HYPERPARAMS['buffer_size']:,}")
    print(f"  Learning rate: {HYPERPARAMS['learning_rate']}")
    print(f"  Exploration: {HYPERPARAMS['exploration_initial_eps']} → "
          f"{HYPERPARAMS['exploration_final_eps']} "
          f"(over {HYPERPARAMS['exploration_fraction'] * 100:.0f}% timesteps)")

    # ── Thiết lập Callbacks ──────────────────────────────────────
    logger_cb    = EpisodeLoggerCallback(log_path=LOG_CSV, verbose=0)
    curriculum_cb = CurriculumCallback(train_env=train_env, verbose=1)
    early_stop_cb = EarlyStoppingCallback(verbose=1)

    eval_cb = SuccessRateEvalCallback(
        eval_env=eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=os.path.join(LOG_DIR, "eval_results"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=0,
    )

    # Chia sẻ tham chiếu curriculum callback cho logger để ghi stage
    logger_cb._curriculum_cb = curriculum_cb

    callback_list = CallbackList([
        logger_cb,
        curriculum_cb,
        early_stop_cb,
        eval_cb,
    ])

    # ── Bắt đầu Training ─────────────────────────────────────────
    print(f"\n[Train] Bắt đầu training | Max timesteps: {MAX_TIMESTEPS:,}\n")
    model.learn(
        total_timesteps=MAX_TIMESTEPS,
        callback=callback_list,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    # ── Lưu model cuối ───────────────────────────────────────────
    model.save(FINAL_MODEL_PATH)
    print(f"\n[Done] Model cuối đã lưu: {FINAL_MODEL_PATH}.zip")
    print(f"[Done] Log CSV: {LOG_CSV}")

    train_env.close()
    eval_env.close()
    print("\n[Done] Training hoàn tất!")


if __name__ == "__main__":
    main()
