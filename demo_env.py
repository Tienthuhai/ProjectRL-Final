"""
demo_env.py - Script Demo & Thử nghiệm Môi trường UAV Path Planning 2D (Gymnasium)

Cho phép:
1. Chạy thử nghiệm tự động (Random Action) hoặc điều khiển thủ công bằng bàn phím.
2. Đổi bản đồ linh hoạt (Phím 1-5, H).
3. Kiểm tra Observation Space 22D, Action Space, Reward và HUD Dashboard.

Hướng dẫn sử dụng:
- Chạy script: python demo_env.py
- Bàn phím điều khiển trong giao diện Pygame:
    + W / Mũi tên Lên   : Bay thẳng (FORWARD)
    + A / Mũi tên Trái  : Ngoặt trái (LEFT LIGHT / HEAVY)
    + D / Mũi tên Phải  : Ngoặt phải (RIGHT LIGHT / HEAVY)
    + SPACE             : Bật/Tắt chế độ tự động bay ngẫu nhiên (Auto Random Actions)
    + Phím 1 -> 5       : Đổi sang Bản đồ 1 -> 5
    + Phím H            : Đổi sang Bản đồ Heldout (Đánh giá tổng quát)
    + R                 : Reset lại tập (Episode)
    + Q / ESC           : Thoát chương trình
"""

import os
import sys
import time
import pygame
import numpy as np

# Thêm thư mục hiện tại vào sys.path để import uav_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from uav_env.uav_map_env import UAVMapEnv


def main():
    print("=" * 65)
    print("  🚀 DEMO MÔI TRƯỜNG UAV PATH PLANNING 2D (GYMNASIUM ENV)")
    print("=" * 65)

    map_path = "maps/map_easy.png"
    if not os.path.exists(map_path):
        print(f"❌ Không tìm thấy {map_path}. Đang kiểm tra thư mục maps/...")
        if os.path.exists("maps"):
            maps = [f for f in os.listdir("maps") if f.endswith(".png")]
            if maps:
                map_path = os.path.join("maps", maps[0])

    print(f"📌 Đang khởi tạo môi trường với bản đồ: {map_path}")
    env = UAVMapEnv(map_path=map_path, fixed_start_goal=True, render_mode="human")

    obs, info = env.reset()

    print(f"✅ Đã khởi tạo thành công!")
    print(f"   - Observation Space Shape : {env.observation_space.shape} (22D Box Vector)")
    print(f"   - Action Space            : {env.action_space} (5 Discrete Actions)")
    print(f"   - Map Size                : {env.map_w}x{env.map_h} px")
    print(f"   - Start Position          : {info['uav_pos']}")
    print(f"   - Goal Position           : {env.goal_pos}\n")

    print("🎮 Đang mở cửa sổ Pygame... (Phím 1-3: Map Train, Phím 4-6: Map Test, SPACE: Auto Mode, R: Reset, Q/ESC: Quit)")

    running = True
    auto_mode = False
    episode_reward = 0.0
    step_count = 0

    clock = pygame.time.Clock()

    map_key_dict = {
        pygame.K_1: "maps/map_easy.png",
        pygame.K_2: "maps/map_medium.png",
        pygame.K_3: "maps/map_hard.png",
        pygame.K_4: "maps/map_heldout.png",
        pygame.K_5: "maps/map_urban.png",
        pygame.K_6: "maps/map_dense.png",
        pygame.K_h: "maps/map_heldout.png",
    }

    while running:
        action = 0  # Mặc định bay thẳng
        action_chosen = False

        # Lắng nghe sự kiện bàn phím từ Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    obs, info = env.reset()
                    episode_reward = 0.0
                    step_count = 0
                    print("🔄 Đã reset Episode!")
                elif event.key == pygame.K_SPACE:
                    auto_mode = not auto_mode
                    print(f"🤖 Chế độ tự động (Auto Mode): {'BẬT' if auto_mode else 'TẮT'}")
                # Đổi Map bằng bàn phím (1->3: Train, 4->6: Test)
                elif event.key in map_key_dict:
                    new_map = map_key_dict[event.key]
                    if os.path.exists(new_map):
                        print(f"🗺️  Chuyển sang bản đồ: {new_map}")
                        env.close()
                        env = UAVMapEnv(map_path=new_map, fixed_start_goal=True, render_mode="human")
                        obs, info = env.reset()
                        episode_reward = 0.0
                        step_count = 0

        # Xác định Action thực thi
        if auto_mode:
            action = env.action_space.sample()  # Tự động chọn hành động ngẫu nhiên
            action_chosen = True
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                action = 0  # FORWARD
                action_chosen = True
            elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
                # Giữ Shift để ngoặt gắt (15°), bình thường ngoặt nhẹ (5°)
                action = 2 if keys[pygame.K_LSHIFT] else 1
                action_chosen = True
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                action = 4 if keys[pygame.K_LSHIFT] else 3
                action_chosen = True

        # Thực thi Step trong môi trường nếu có action hoặc đang ở chế độ auto
        if action_chosen:
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            if terminated or truncated:
                status_str = "SUCCESS (GOAL REACHED! 🎉)" if reward > 0 else "COLLISION / TIMEOUT 💥"
                print(f"🏁 Episode Kết thúc | Steps: {step_count} | Total Reward: {episode_reward:.2f} | Trạng thái: {status_str}")
                time.sleep(1.0)
                obs, info = env.reset()
                episode_reward = 0.0
                step_count = 0
        else:
            # Nếu ở chế độ thủ công và không nhấn phím di chuyển, vẫn render giao diện
            env.render()

        clock.tick(30)  # 30 FPS

    env.close()
    print("👋 Đã đóng môi trường UAV Demo. Cảm ơn bạn!")


if __name__ == "__main__":
    main()
