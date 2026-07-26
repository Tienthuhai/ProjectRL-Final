"""
train_a2c.py — Huấn luyện A2C với đa tiến trình (SubprocVecEnv) cho UAVMapEnv.

Tối ưu phần cứng:
- Sử dụng 12 processes CPU để mô phỏng môi trường song song.
- Sử dụng GPU để train A2C.

Các cấu trúc callback giữ nguyên như cũ, chỉ thay đổi thư mục lưu log.
"""

import os
import csv
import random
import time
import numpy as np
import torch
from collections import deque

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

from uav_env.uav_map_env import UAVMapEnv

# ───────────────────────────────────────────────────────────────
# Cấu hình chung
# ───────────────────────────────────────────────────────────────
SEED = 42
MAX_TIMESTEPS = 300_000
N_ENVS = 10  # Dùng 10 luồng theo yêu cầu

CLEAR_PREVIOUS_RESULTS = 1  # 1 (Bật) hoặc 0 (Tắt): Xóa toàn bộ kết quả cũ trước khi chạy lại

TRAIN_MAPS = [
    "maps/map_easy.png",
    "maps/map_medium.png",
    "maps/map_hard.png",
]
EVAL_MAP = "maps/map_easy.png"

CURRICULUM_WINDOW = 200
CURRICULUM_THRESHOLD = 0.75

EARLY_STOP_WINDOW = 200
EARLY_STOP_THRESHOLD = 0.85
EARLY_STOP_MIN_IMPROVEMENT = 0.02
EARLY_STOP_PATIENCE = 500

EVAL_FREQ = max(5_000 // N_ENVS, 1)  # Cập nhật eval_freq vì mỗi step của mô hình = N_ENVS steps của môi trường
N_EVAL_EPISODES = 20

# THAY ĐỔI ĐƯỜNG DẪN LƯU KẾT QUẢ CHO A2C
RESULTS_DIR = "results_a2c"
LOG_DIR     = os.path.join(RESULTS_DIR, "logs")
MODEL_DIR   = os.path.join(RESULTS_DIR, "models")
LOG_CSV     = os.path.join(RESULTS_DIR, "train_log.csv")
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "a2c_best")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "a2c_final")

# Hyperparameters cho A2C
HYPERPARAMS = dict(
    learning_rate=3e-4,
    n_steps=100,          # Tăng n_steps lên 100 -> Batch size = 100 * 10 = 1000 để bắt GPU tính toán cục lớn hơn
    gamma=0.99,
    ent_coef=0.01,        # Entropy coefficient giúp exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[1024, 1024]), # Tăng mạng Nơ-ron lên gấp đôi để "vắt" GPU
)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_env(map_path: str, fixed_start_goal: bool = False, seed: int = SEED, rank: int = 0):
    """
    Trả về một hàm khởi tạo môi trường để truyền vào SubprocVecEnv.
    """
    def _init():
        env = UAVMapEnv(
            map_path=map_path,
            fixed_start_goal=fixed_start_goal,
            render_mode=None,
            max_steps=500,  # Giới hạn 300 bước mỗi màn chơi theo yêu cầu
        )
        env.reset(seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


# ───────────────────────────────────────────────────────────────
# Callback 1: EpisodeLoggerCallback
# ───────────────────────────────────────────────────────────────
class EpisodeLoggerCallback(BaseCallback):
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_num = 0
        self.outcome_history: deque = deque(maxlen=EARLY_STOP_WINDOW)
        self.start_time = time.time()

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        # Chỉ ghi header nếu file chưa tồn tại hoặc trống
        if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "timestep", "total_reward", "episode_length",
                    "outcome", "current_map_stage", "rolling_success_rate_200"
                ])

        self._curriculum_cb = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            self.episode_num += 1

            ep_info = info.get("episode", {})
            total_reward   = ep_info.get("r", 0.0)
            episode_length = ep_info.get("l", 0)

            outcome = info.get("outcome", None)
            if outcome is None:
                outcome = info.get("terminal_info", {}).get("outcome", "timeout")

            self.outcome_history.append(1 if outcome == "success" else 0)
            rolling_sr = (sum(self.outcome_history) / len(self.outcome_history)
                          if self.outcome_history else 0.0)

            map_stage = (self._curriculum_cb.current_stage_idx
                         if self._curriculum_cb else 0)

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

            if self.episode_num % 50 == 0:
                stage_name = TRAIN_MAPS[map_stage] if map_stage < len(TRAIN_MAPS) else "N/A"
                
                # Tính thời gian còn lại
                elapsed = time.time() - self.start_time
                steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
                rem_steps = MAX_TIMESTEPS - self.num_timesteps
                rem_time = rem_steps / steps_per_sec if steps_per_sec > 0 else 0
                rem_m, rem_s = divmod(int(rem_time), 60)
                rem_h, rem_m = divmod(rem_m, 60)
                time_str = f"{rem_h:02d}:{rem_m:02d}:{rem_s:02d}"

                print(
                    f"[Logger] Epoch/Ep {self.episode_num:>5} | "
                    f"Steps {self.num_timesteps:>7} | "
                    f"ETA: {time_str} | "
                    f"SR-200: {rolling_sr:.1%} | "
                    f"Stage: {os.path.basename(stage_name)}"
                )

        return True


# ───────────────────────────────────────────────────────────────
# Callback 2: CurriculumCallback
# ───────────────────────────────────────────────────────────────
class CurriculumCallback(BaseCallback):
    def __init__(self, train_env, verbose: int = 1):
        super().__init__(verbose)
        self.train_env = train_env
        self.current_stage_idx = 0
        self.outcome_history: deque = deque(maxlen=CURRICULUM_WINDOW)
        self._early_stop_cb = None
        self._just_switched = False

    def _on_step(self) -> bool:
        if self.current_stage_idx >= len(TRAIN_MAPS) - 1:
            return True

        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        self._just_switched = False

        for done, info in zip(dones, infos):
            if not done:
                continue
            outcome = info.get("outcome") or info.get("terminal_info", {}).get("outcome", "timeout")
            self.outcome_history.append(1 if outcome == "success" else 0)

        if len(self.outcome_history) < CURRICULUM_WINDOW or self._just_switched:
            return True

        rolling_sr = sum(self.outcome_history) / len(self.outcome_history)
        if rolling_sr >= CURRICULUM_THRESHOLD:
            self._just_switched = True
            self.current_stage_idx += 1
            new_map = TRAIN_MAPS[self.current_stage_idx]

            self.outcome_history.clear()

            if self._early_stop_cb is not None:
                self._early_stop_cb.on_curriculum_switch()

            try:
                self.train_env.env_method("_load_map", new_map)
                self.train_env.env_method("reset")
            except Exception as e:
                print(f"[Curriculum] WARNING: Could not hot-swap map: {e}")

            print(
                f"\n[Curriculum] >>> Stage {self.current_stage_idx}: "
                f"Chuyen sang {os.path.basename(new_map)} "
                f"(SR-200={rolling_sr:.1%}) <<<\n"
            )

        return True


# ───────────────────────────────────────────────────────────────
# Callback 3: EarlyStoppingCallback
# ───────────────────────────────────────────────────────────────
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, verbose: int = 1):
        super().__init__(verbose)
        self.outcome_history: deque = deque(maxlen=EARLY_STOP_WINDOW * 2)
        self.patience_counter = 0
        self.prev_rolling_sr = 0.0
        self.episode_num = 0

    def on_curriculum_switch(self):
        self.outcome_history.clear()
        self.patience_counter = 0
        self.prev_rolling_sr = 0.0
        if self.verbose >= 1:
            print("[EarlyStopping] Reset patience counter do Curriculum chuyển stage.")

    def _on_step(self) -> bool:
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
            outcome = info.get("outcome") or info.get("terminal_info", {}).get("outcome", "timeout")
            self.outcome_history.append(1 if outcome == "success" else 0)

            if len(self.outcome_history) < EARLY_STOP_WINDOW * 2:
                continue

            rolling_sr = self._get_rolling_sr()

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
# Custom EvalCallback
# ───────────────────────────────────────────────────────────────
class SuccessRateEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_success_rate = -1.0

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Với VecEnv, n_calls tăng lên 1 mỗi lần step(). Mỗi step() xử lý N_ENVS steps của môi trường
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sr = self._compute_success_rate()
            print(f"[EvalCallback] Eval SR: {sr:.1%} (best: {self.best_success_rate:.1%})")

            if sr > self.best_success_rate:
                self.best_success_rate = sr
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "a2c_best"))
                    print(f"[EvalCallback] Lưu model tốt nhất: SR={sr:.1%}")

        return result

    def _compute_success_rate(self) -> float:
        successes = []

        def _count_success(_locals, _globals):
            dones = _locals.get("dones", [])
            infos = _locals.get("infos", [])
            for done, info in zip(dones, infos):
                if done:
                    outcome = info.get("outcome") or info.get("terminal_info", {}).get("outcome", "timeout")
                    successes.append(1 if (outcome == "success" or info.get("is_success", False)) else 0)

        evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            callback=_count_success,
            warn=False,
        )

        if not successes:
            return 0.0
        return sum(successes) / len(successes)


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  UAV PATH PLANNING — A2C Training ({N_ENVS} processes, GPU)")
    print("=" * 70)

    import shutil
    if CLEAR_PREVIOUS_RESULTS == 1 and os.path.exists(RESULTS_DIR):
        print(f"\n[Setup] Đang xóa kết quả cũ trong thư mục '{RESULTS_DIR}'...")
        shutil.rmtree(RESULTS_DIR)

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "eval_results"), exist_ok=True)

    set_global_seed(SEED)

    print(f"\n[Setup] Khởi tạo {N_ENVS} training envs song song: {TRAIN_MAPS[0]}")
    # Chạy N_ENVS process song song
    train_env = SubprocVecEnv([make_env(TRAIN_MAPS[0], fixed_start_goal=False, seed=SEED, rank=i) for i in range(N_ENVS)])

    print(f"[Setup] Khởi tạo eval env: {EVAL_MAP}")
    # Eval_env chỉ cần DummyVecEnv với 1 env
    eval_env = DummyVecEnv([make_env(EVAL_MAP, fixed_start_goal=True, seed=SEED + 100, rank=0)])

    print("\n[Setup] Khởi tạo A2C model...")
    # Khởi tạo A2C
    model = A2C(
        policy="MlpPolicy",
        env=train_env,
        seed=SEED,
        verbose=1, # Bật verbose = 1 để xem fps
        tensorboard_log=None,
        device="cuda", # Bắt buộc xài GPU nếu có
        **HYPERPARAMS,
    )

    print(f"  Policy net: {HYPERPARAMS['policy_kwargs']['net_arch']}")
    print(f"  Learning rate: {HYPERPARAMS['learning_rate']}")
    print(f"  n_steps: {HYPERPARAMS['n_steps']} (Batch update size: {HYPERPARAMS['n_steps'] * N_ENVS})")
    
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

    logger_cb._curriculum_cb = curriculum_cb
    curriculum_cb._early_stop_cb = early_stop_cb

    callback_list = CallbackList([
        logger_cb,
        curriculum_cb,
        early_stop_cb,
        eval_cb,
    ])

    print(f"\n[Train] Bắt đầu training | Max timesteps: {MAX_TIMESTEPS:,}\n")
    model.learn(
        total_timesteps=MAX_TIMESTEPS,
        callback=callback_list,
        reset_num_timesteps=True,
        progress_bar=True,
    )

    model.save(FINAL_MODEL_PATH)
    print(f"\n[Done] Model cuối đã lưu: {FINAL_MODEL_PATH}.zip")
    print(f"[Done] Log CSV: {LOG_CSV}")

    train_env.close()
    eval_env.close()
    print("\n[Done] Training hoàn tất!")


if __name__ == "__main__":
    main()
