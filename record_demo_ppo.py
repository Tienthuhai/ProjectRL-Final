"""
record_demo_ppo.py — Quay video demo agent PPO đã train trên cả 6 map.

Chạy sau khi train_ppo.py hoàn tất và có results_ppo/models/ppo_best.zip.

Output:
  results_ppo/videos/demo_<map_name>.mp4  — video riêng cho từng map
  results_ppo/videos/demo_all_maps.mp4    — video ghép cả 6 map (nếu có opencv-python)

Yêu cầu thêm: opencv-python (pip install opencv-python)
               imageio[ffmpeg] hoặc ffmpeg trên PATH

Cách dùng:
  python record_demo_ppo.py
  python record_demo_ppo.py --model results_ppo/models/ppo_final.zip   # dùng model khác
"""

import os
import sys
import math
import argparse
import numpy as np
import pygame
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uav_env.uav_map_env import UAVMapEnv

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("[Warning] stable-baselines3 không tìm thấy. Sẽ chạy agent random.")

# ── Cấu hình ──────────────────────────────────────────────────
DEFAULT_MODEL = "results_ppo/models/ppo_best.zip"

VIDEO_DIR     = "results_ppo/videos"
FPS           = 30
N_EPISODES    = 5          # Số episode quay cho mỗi map

ALL_MAPS = [
    ("maps/map_easy.png",    "map_easy",    "Train Stage 1 (Easy)"),
    ("maps/map_medium.png",  "map_medium",  "Train Stage 2 (Medium)"),
    ("maps/map_hard.png",    "map_hard",    "Train Stage 3 (Hard)"),
    ("maps/map_heldout.png", "map_heldout", "Test 1 (Held-out Generalization)"),
    ("maps/map_urban.png",   "map_urban",   "Test 2 (Urban City Blocks)"),
    ("maps/map_dense.png",   "map_dense",   "Test 3 (Dense Obstacles)"),
]


def draw_overlay(surface: pygame.Surface, map_title: str,
                 ep: int, step: int, total_reward: float,
                 outcome: str, font_title, font_info):
    """Vẽ overlay thông tin lên góc dưới trái màn hình."""
    bar_h = 55
    bar_w = surface.get_width()
    bar_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
    bar_surf.fill((0, 0, 0, 165))
    surface.blit(bar_surf, (0, surface.get_height() - bar_h))

    y_base = surface.get_height() - bar_h + 6
    title_surf = font_title.render(f"  {map_title}", True, (255, 220, 50))
    surface.blit(title_surf, (5, y_base))

    info_text = (f"  Episode {ep}  |  Step {step}  |  "
                 f"Reward: {total_reward:+.1f}  |  Status: {outcome.upper()}")
    info_surf = font_info.render(info_text, True, (220, 220, 220))
    surface.blit(info_surf, (5, y_base + 22))


def record_map(map_path: str, map_name: str, map_title: str,
               model, video_path: str, n_episodes: int = N_EPISODES):
    """
    Quay n_episodes trên một map, lưu thành file MP4.
    Trả về list các frame (numpy array RGB) toàn bộ video.
    """
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Phải init pygame TRƯỜC khi tạo env vì env.render() gọi pygame.font bên trong
    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
    # Cần display surface (dù dummy) để SysFont hoạt động dưới render_mode=rgb_array
    if pygame.display.get_surface() is None:
        pygame.display.set_mode((1, 1), pygame.NOFRAME)

    # Khởi tạo env ở chế độ rgb_array (không mở cửa sổ Pygame thật)
    env = UAVMapEnv(
        map_path=map_path,
        fixed_start_goal=True,
        render_mode="rgb_array",
        max_steps=500,
    )

    # Lấy kích thước frame từ env
    obs, _ = env.reset()
    sample_frame = env.render()
    if sample_frame is None:
        print(f"[Record] WARNING: env.render() trả về None cho {map_name}")
        env.close()
        return []

    h, w, _ = sample_frame.shape

    # Khởi tạo video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (w, h))

    font_title = pygame.font.SysFont("Arial", 13, bold=True)
    font_info  = pygame.font.SysFont("Arial", 11)

    all_frames = []
    outcomes_count = {"success": 0, "collision": 0, "timeout": 0}

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        done = False
        step = 0
        total_reward = 0.0
        outcome = "in_progress"

        while not done:
            # Lấy action từ model (deterministic) hoặc random nếu không có model
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            outcome = info.get("outcome", "in_progress")

            # Render frame
            frame_rgb = env.render()
            if frame_rgb is None:
                continue

            # Chuyển sang Surface pygame để vẽ overlay
            surf = pygame.surfarray.make_surface(frame_rgb.transpose(1, 0, 2))
            draw_overlay(surf, map_title, ep, step, total_reward,
                         outcome if done else "flying",
                         font_title, font_info)

            # Chuyển lại thành numpy array BGR (cho opencv)
            frame_out = pygame.surfarray.array3d(surf).transpose(1, 0, 2)
            frame_bgr = cv2.cvtColor(frame_out, cv2.COLOR_RGB2BGR)

            writer.write(frame_bgr)
            all_frames.append(frame_out)   # giữ RGB cho ghép video

        outcomes_count[outcome if outcome in outcomes_count else "timeout"] += 1
        print(f"  [Episode {ep}/{n_episodes}] Steps: {step:>3} | "
              f"Reward: {total_reward:>8.1f} | Outcome: {outcome}")

    writer.release()
    env.close()

    print(f"  -> Outcomes: {outcomes_count}")
    print(f"  -> Video saved: {video_path}")
    return all_frames


def concat_videos_grid(all_video_frames: dict, output_path: str,
                        map_titles: list, fps: int = FPS):
    """
    Ghép 6 video thành 1 lưới 2x3 (2 hàng x 3 cột).
    all_video_frames: dict {map_name: [frames_rgb]}
    """
    map_names = list(all_video_frames.keys())
    if len(map_names) < 6:
        print("[Concat] Không đủ 6 map để ghép video lưới.")
        return

    # Lấy kích thước frame từ map đầu tiên
    first_frames = all_video_frames[map_names[0]]
    if not first_frames:
        return
    fh, fw, _ = first_frames[0].shape

    # Resize tất cả frame về cùng kích thước (nếu khác nhau)
    target_h, target_w = fh, fw

    # Kích thước lưới 2x3
    grid_w = target_w * 3
    grid_h = target_h * 2

    # Cân bằng độ dài: lấy max số frame
    max_frames = max(len(v) for v in all_video_frames.values())

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (grid_w, grid_h))

    for frame_idx in range(max_frames):
        grid_frame = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for map_idx, map_name in enumerate(map_names[:6]):
            frames = all_video_frames[map_name]
            # Lặp lại frame cuối nếu video ngắn hơn
            f = frames[min(frame_idx, len(frames) - 1)] if frames else np.zeros((target_h, target_w, 3), np.uint8)
            f_resized = cv2.resize(f, (target_w, target_h))

            row = map_idx // 3
            col = map_idx % 3
            y0, y1 = row * target_h, (row + 1) * target_h
            x0, x1 = col * target_w, (col + 1) * target_w
            grid_frame[y0:y1, x0:x1] = f_resized

        # Chuyển sang BGR cho writer
        grid_bgr = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
        writer.write(grid_bgr)

    writer.release()
    print(f"\n[Concat] Video lưới 2x3 đã lưu: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Record UAV PPO Demo Videos")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Đường dẫn tới model SB3 (.zip)")
    parser.add_argument("--episodes", type=int, default=N_EPISODES,
                        help="Số episode quay cho mỗi map")
    parser.add_argument("--no-concat", action="store_true",
                        help="Bỏ qua bước ghép video lưới 2x3")
    args = parser.parse_args()

    os.makedirs(VIDEO_DIR, exist_ok=True)

    # ── Nạp model ───────────────────────────────────────────────
    model = None
    model_path = args.model
    if not model_path.endswith(".zip"):
        model_path += ".zip"

    if SB3_AVAILABLE and os.path.exists(model_path):
        print(f"[Setup] Nạp model: {model_path}")
        model = PPO.load(model_path)
        print("[Setup] Model nạp thành công!")
    else:
        if not os.path.exists(model_path):
            print(f"[Setup] Không tìm thấy model tại '{model_path}'. Sử dụng agent random.")
        else:
            print("[Setup] SB3 không khả dụng. Sử dụng agent random.")

    # ── Quay video từng map ─────────────────────────────────────
    print(f"\n[Record] Bắt đầu quay video {len(ALL_MAPS)} maps | {args.episodes} ep/map\n")

    all_video_frames = {}

    for map_path, map_name, map_title in ALL_MAPS:
        if not os.path.exists(map_path):
            print(f"[Record] Bỏ qua {map_name}: không tìm thấy {map_path}")
            continue

        print(f"\n[Record] === {map_title} ({map_name}) ===")
        video_path = os.path.join(VIDEO_DIR, f"demo_{map_name}.mp4")

        frames = record_map(
            map_path=map_path,
            map_name=map_name,
            map_title=map_title,
            model=model,
            video_path=video_path,
            n_episodes=args.episodes,
        )
        all_video_frames[map_name] = frames

    # ── Ghép video lưới 2x3 ─────────────────────────────────────
    if not args.no_concat and len(all_video_frames) >= 6:
        concat_path = os.path.join(VIDEO_DIR, "demo_all_maps.mp4")
        print(f"\n[Concat] Ghép video lưới 2x3 → {concat_path}")
        map_titles_list = [t for _, _, t in ALL_MAPS]
        concat_videos_grid(all_video_frames, concat_path, map_titles_list)

    print(f"\n[Done] Tất cả video đã lưu vào: {VIDEO_DIR}/")
    pygame.quit()


if __name__ == "__main__":
    main()
