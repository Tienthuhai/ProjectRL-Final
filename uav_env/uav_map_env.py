import os
import math
import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import pygame

class UAVMapEnv(gym.Env):
    """
    Môi trường Gymnasium mô phỏng UAV Path Planning 2D trên Ảnh Bản Đồ.
    Cải tiến State Space 22D:
      - 16 tia LiDAR (chuẩn hóa [0, 1])
      - 1 target_dist (chuẩn hóa [0, 1])
      - 2D sin(target_angle), cos(target_angle) (triệt tiêu lỗi gián đoạn góc [-1, 1])
      - 2D sin(heading), cos(heading) (triệt tiêu lỗi gián đoạn góc [-1, 1])
      - 1D min_lidar_distance (tín hiệu nguy hiểm cận kề [0, 1])
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    ACTION_FORWARD = 0
    ACTION_TURN_LEFT_LIGHT = 1
    ACTION_TURN_LEFT_HEAVY = 2
    ACTION_TURN_RIGHT_LIGHT = 3
    ACTION_TURN_RIGHT_HEAVY = 4

    DEFAULT_MAP_CONFIGS = {
        # map_easy (cũ: map_medium) — 1 hình chữ nhật to chính giữa
        # Start góc trên trái, Goal góc dưới phải, đường chéo tránh tường
        "maps/map_easy.png":    {"start": (40.0, 40.0),  "goal": (460.0, 460.0)},
        # map_medium (cũ: map_hard) — các vòng tròn phân tán
        "maps/map_medium.png":  {"start": (40.0, 40.0),  "goal": (460.0, 460.0)},
        # map_hard (cũ: map_easy) — nhiều hình chữ nhật phức tạp
        "maps/map_hard.png":    {"start": (40.0, 40.0),  "goal": (460.0, 460.0)},
        "maps/map_heldout.png": {"start": (50.0, 50.0),  "goal": (450.0, 450.0)},
        "maps/map_urban.png":   {"start": (50.0, 50.0),  "goal": (450.0, 450.0)},
        "maps/map_dense.png":   {"start": (50.0, 50.0),  "goal": (450.0, 450.0)},
    }

    def __init__(self, map_path="maps/map_easy.png", fixed_start_goal=True, render_mode=None, max_steps=700, speed=4.0, max_range=150.0):
        super().__init__()
        self.map_path = map_path
        self.fixed_start_goal = fixed_start_goal
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.speed = speed
        self.max_range = max_range
        self.num_lidar_rays = 24

        # Nạp ảnh bản đồ
        self._load_map(map_path)

        # Action Space & Observation Space (30D Box Vector: 24D LiDAR + 6D Spatial Cues)
        self.action_space = spaces.Discrete(5)
        obs_low  = np.array([0.0]*24 + [0.0] + [-1.0,-1.0] + [-1.0,-1.0] + [0.0], dtype=np.float32)
        obs_high = np.array([1.0]*24 + [1.0] + [ 1.0, 1.0] + [ 1.0, 1.0] + [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # Trạng thái nội tại
        self.uav_pos = np.zeros(2, dtype=np.float32)
        self.heading = 0.0  # Radians [-pi, pi]
        self.start_pos = np.zeros(2, dtype=np.float32)
        self.goal_pos = np.zeros(2, dtype=np.float32)
        self.battery = 100.0
        self.current_step = 0
        self.total_reward = 0.0
        self.trajectory = []
        self.lidar_distances = np.zeros(self.num_lidar_rays, dtype=np.float32)
        self.last_action_str = "None"

        # Pygame Rendering attributes
        self.window = None
        self.clock = None
        self.hud_width = 240
        self.window_width = self.map_w + self.hud_width
        self.window_height = self.map_h

    def _load_map(self, map_path):
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh bản đồ tại: {map_path}")
        
        raw_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise ValueError(f"Không thể đọc file ảnh bản đồ: {map_path}")
        
        self.map_h, self.map_w = raw_img.shape
        self.grid = (raw_img > 127).astype(np.uint8)
        self.map_diagonal = math.hypot(self.map_w, self.map_h)
        self.map_path = map_path
        self.map_surface_pygame = None

    def _is_free(self, x, y, margin=5):
        ix, iy = int(round(x)), int(round(y))
        if ix - margin < 0 or ix + margin >= self.map_w or iy - margin < 0 or iy + margin >= self.map_h:
            return False
        sub_grid = self.grid[iy - margin : iy + margin + 1, ix - margin : ix + margin + 1]
        return np.all(sub_grid == 1)

    def _sample_free_position(self, margin=20):
        """Random vị trí trong vùng free space có khoảng cách an toàn tối thiểu 20px tới tường"""
        for _ in range(3000):
            x = float(self.np_random.uniform(margin, self.map_w - margin))
            y = float(self.np_random.uniform(margin, self.map_h - margin))
            if self._is_free(x, y, margin=margin):
                return np.array([x, y], dtype=np.float32)
        return np.array([50.0, 50.0], dtype=np.float32)

    def _cast_lidar(self):
        angles = self.heading + np.linspace(0, 2 * math.pi, self.num_lidar_rays, endpoint=False)
        distances = np.zeros(self.num_lidar_rays, dtype=np.float32)

        for i, angle in enumerate(angles):
            dx = math.cos(angle)
            dy = math.sin(angle)
            dist = 0.0
            step_size = 1.5

            while dist < self.max_range:
                cx = self.uav_pos[0] + dx * dist
                cy = self.uav_pos[1] + dy * dist

                ix, iy = int(round(cx)), int(round(cy))
                if ix < 0 or ix >= self.map_w or iy < 0 or iy >= self.map_h or self.grid[iy, ix] == 0:
                    break
                dist += step_size

            distances[i] = min(dist, self.max_range)

        return distances

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and "map_path" in options:
            self._load_map(options["map_path"])
            self.map_surface_pygame = None

        # Đặt vị trí Start & Goal
        if self.fixed_start_goal and self.map_path in self.DEFAULT_MAP_CONFIGS:
            cfg = self.DEFAULT_MAP_CONFIGS[self.map_path]
            self.start_pos = np.array(cfg["start"], dtype=np.float32)
            self.goal_pos = np.array(cfg["goal"], dtype=np.float32)
        else:
            self.start_pos = self._sample_free_position(margin=20)
            while True:
                self.goal_pos = self._sample_free_position(margin=20)
                if np.linalg.norm(self.start_pos - self.goal_pos) >= 150.0:
                    break

        self.uav_pos = self.start_pos.copy()
        dx = self.goal_pos[0] - self.uav_pos[0]
        dy = self.goal_pos[1] - self.uav_pos[1]
        self.heading = math.atan2(dy, dx)

        self.battery = 100.0
        self.current_step = 0
        self.total_reward = 0.0
        self.trajectory = [self.uav_pos.copy()]
        self.last_action_str = "RESET"

        self.lidar_distances = self._cast_lidar()
        obs = self._get_obs()
        info = self._get_info(outcome="in_progress")

        if self.render_mode == "human":
            self.render()

        return obs, info

    def _get_obs(self):
        """Tạo vector Observation 22 chiều đã chuẩn hóa (Loại bỏ gián đoạn góc với sin/cos)"""
        # 1. 16 tia LiDAR chuẩn hóa [0, 1] (16D)
        norm_lidar = self.lidar_distances / self.max_range

        # 2. Distance to Goal chuẩn hóa [0, 1] (1D)
        target_dist = np.linalg.norm(self.goal_pos - self.uav_pos)
        norm_target_dist = min(1.0, target_dist / self.map_diagonal)

        # 3. Relative Angle to Goal dạng sin/cos [-1, 1] (2D)
        dx = self.goal_pos[0] - self.uav_pos[0]
        dy = self.goal_pos[1] - self.uav_pos[1]
        angle_to_goal = math.atan2(dy, dx)
        rel_angle = math.atan2(math.sin(angle_to_goal - self.heading), math.cos(angle_to_goal - self.heading))
        sin_rel_angle = math.sin(rel_angle)
        cos_rel_angle = math.cos(rel_angle)

        # 4. Heading dạng sin/cos [-1, 1] (2D)
        sin_heading = math.sin(self.heading)
        cos_heading = math.cos(self.heading)

        # 5. Feature Engineering: Cảm biến nguy hiểm cận kề min_lidar [0, 1] (1D)
        min_lidar_norm = np.min(self.lidar_distances) / self.max_range

        obs = np.hstack([
            norm_lidar,
            np.float32(norm_target_dist),
            np.float32(sin_rel_angle),
            np.float32(cos_rel_angle),
            np.float32(sin_heading),
            np.float32(cos_heading),
            np.float32(min_lidar_norm)
        ]).astype(np.float32)

        return obs

    def _get_info(self, outcome: str = "in_progress"):
        target_dist = float(np.linalg.norm(self.goal_pos - self.uav_pos))
        dx = self.goal_pos[0] - self.uav_pos[0]
        dy = self.goal_pos[1] - self.uav_pos[1]
        angle_to_goal = math.atan2(dy, dx)
        rel_angle_deg = math.degrees(math.atan2(math.sin(angle_to_goal - self.heading), math.cos(angle_to_goal - self.heading)))

        return {
            "step": self.current_step,
            "target_dist": target_dist,
            "relative_angle_deg": rel_angle_deg,
            "battery": self.battery,
            "total_reward": self.total_reward,
            "uav_pos": self.uav_pos.copy(),
            # --- Dùng cho SB3 Callbacks và EvalCallback ---
            "outcome": outcome,           # "success" | "collision" | "timeout" | "in_progress"
            "is_success": outcome == "success",  # bool, dùng cho EvalCallback
        }

    def step(self, action):
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False

        action_names = ["FORWARD", "LEFT_LIGHT (+5°)", "LEFT_HEAVY (+15°)", "RIGHT_LIGHT (-5°)", "RIGHT_HEAVY (-15°)"]
        self.last_action_str = action_names[action] if action < 5 else "UNKNOWN"

        d_theta = 0.0
        if action == self.ACTION_TURN_LEFT_LIGHT:
            d_theta = math.radians(5.0)
        elif action == self.ACTION_TURN_LEFT_HEAVY:
            d_theta = math.radians(15.0)
        elif action == self.ACTION_TURN_RIGHT_LIGHT:
            d_theta = math.radians(-5.0)
        elif action == self.ACTION_TURN_RIGHT_HEAVY:
            d_theta = math.radians(-15.0)

        self.heading = math.atan2(math.sin(self.heading + d_theta), math.cos(self.heading + d_theta))

        prev_dist = np.linalg.norm(self.goal_pos - self.uav_pos)

        new_x = self.uav_pos[0] + self.speed * math.cos(self.heading)
        new_y = self.uav_pos[1] + self.speed * math.sin(self.heading)

        # Kẹp tọa độ UAV trong phạm vi bản đồ (tránh bay sang vùng HUD)
        new_x = float(np.clip(new_x, 2.0, self.map_w - 2.0))
        new_y = float(np.clip(new_y, 2.0, self.map_h - 2.0))

        self.battery = max(0.0, self.battery - 0.15)

        self.uav_pos = np.array([new_x, new_y], dtype=np.float32)
        self.trajectory.append(self.uav_pos.copy())
        self.lidar_distances = self._cast_lidar()

        current_dist = np.linalg.norm(self.goal_pos - self.uav_pos)

        # Tính Toán Thưởng / Phạt
        min_lidar = np.min(self.lidar_distances)
        outcome = "in_progress"
        if min_lidar <= 2.0 or not self._is_free(new_x, new_y, margin=3) or new_x <= 5.0 or new_x >= self.map_w - 5.0 or new_y <= 5.0 or new_y >= self.map_h - 5.0:
            reward = -250.0
            terminated = True
            outcome = "collision"
            self.last_action_str += " (COLLISION! THẤT BẠI)"
        elif current_dist <= 20.0:
            reward = +500.0
            terminated = True
            outcome = "success"
            self.last_action_str += " (GOAL REACHED! THẮNG)"
        else:
            # 1. Distance shaping reward
            shaping_reward = 1.2 * (prev_dist - current_dist)
            reward += shaping_reward

            # 2. Heading alignment reward: thưởng khi hướng về phía đích
            dx_g = self.goal_pos[0] - new_x
            dy_g = self.goal_pos[1] - new_y
            angle_to_g = math.atan2(dy_g, dx_g)
            rel_a = math.atan2(math.sin(angle_to_g - self.heading), math.cos(angle_to_g - self.heading))
            reward += 0.3 * math.cos(rel_a)

            # 3. Phạt nguy hiểm lũy thừa khi áp sát vật cản (LiDAR < 30px)
            # Ép Q-value của hành động tiến thẳng đâm vật cản thấp hơn hành động ngoặt lách
            if min_lidar < 30.0:
                proximity_penalty = 3.5 * ((30.0 - min_lidar) / 30.0) ** 2
                reward -= proximity_penalty

            # 4. Phạt bước thời gian nhẹ
            reward -= 0.1

        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            outcome = "timeout"

        self.total_reward += reward
        obs = self._get_obs()
        info = self._get_info(outcome=outcome)

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption("UAV 2D Path Planning - Gymnasium Environment")
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((240, 243, 246))

        # 1. Vẽ Ảnh Bản Đồ Màu Phố Xá
        if self.map_surface_pygame is None:
            self.map_surface_pygame = pygame.image.load(self.map_path)
        canvas.blit(self.map_surface_pygame, (0, 0))

        # 2. Vẽ Vệt Đường Bay
        if len(self.trajectory) > 1:
            pts = [(int(round(p[0])), int(round(p[1]))) for p in self.trajectory]
            pygame.draw.lines(canvas, (142, 68, 173), False, pts, 2)

        # 3. Vẽ 16 Tia LiDAR
        angles = self.heading + np.linspace(0, 2 * math.pi, self.num_lidar_rays, endpoint=False)
        for i, angle in enumerate(angles):
            dist = self.lidar_distances[i]
            end_x = int(round(self.uav_pos[0] + dist * math.cos(angle)))
            end_y = int(round(self.uav_pos[1] + dist * math.sin(angle)))

            if dist < 30.0:
                color = (231, 76, 60)
            elif dist < 70.0:
                color = (241, 196, 15)
            else:
                color = (46, 204, 113)

            pygame.draw.line(canvas, color, (int(round(self.uav_pos[0])), int(round(self.uav_pos[1]))), (end_x, end_y), 1)

        # 4. Vẽ Vị Trí Start & Goal
        sx, sy = int(round(self.start_pos[0])), int(round(self.start_pos[1]))
        pygame.draw.circle(canvas, (46, 204, 113), (sx, sy), 10)
        pygame.draw.circle(canvas, (255, 255, 255), (sx, sy), 10, 2)

        gx, gy = int(round(self.goal_pos[0])), int(round(self.goal_pos[1]))
        pygame.draw.circle(canvas, (231, 76, 60), (gx, gy), 15)
        pygame.draw.circle(canvas, (255, 255, 255), (gx, gy), 15, 2)
        font = pygame.font.SysFont("Arial", 11, bold=True)
        txt_g = font.render("GOAL", True, (255, 255, 255))
        canvas.blit(txt_g, (gx - 14, gy - 6))

        # 5. Vẽ UAV (Tam Giác Xoay theo Heading)
        ux, uy = self.uav_pos[0], self.uav_pos[1]
        size = 12.0
        p1 = (int(round(ux + size * math.cos(self.heading))), int(round(uy + size * math.sin(self.heading))))
        p2 = (int(round(ux + size * 0.6 * math.cos(self.heading + 2.5))), int(round(uy + size * 0.6 * math.sin(self.heading + 2.5))))
        p3 = (int(round(ux + size * 0.6 * math.cos(self.heading - 2.5))), int(round(uy + size * 0.6 * math.sin(self.heading - 2.5))))
        
        pygame.draw.polygon(canvas, (52, 152, 219), [p1, p2, p3])
        pygame.draw.polygon(canvas, (255, 255, 255), [p1, p2, p3], 1)

        # 6. Bảng HUD Dashboard
        panel_x = self.map_w + 15
        font_title = pygame.font.SysFont("Arial", 15, bold=True)
        font_info = pygame.font.SysFont("Arial", 12)

        title = font_title.render("UAV FLIGHT DASHBOARD", True, (44, 62, 80))
        canvas.blit(title, (panel_x, 15))

        target_dist = float(np.linalg.norm(self.goal_pos - self.uav_pos))
        map_name = os.path.basename(self.map_path)
        info_lines = [
            f"Bản đồ: {map_name}",
            f"State Dim: 22D (sin/cos)",
            f"Mode: {'CỐ ĐỊNH START/GOAL' if self.fixed_start_goal else 'RANDOM'}",
            f"Step: {self.current_step} / {self.max_steps}",
            f"Total Reward: {self.total_reward:.1f}",
            f"Action: {self.last_action_str}",
            f"Dist to Goal: {target_dist:.1f} px",
            f"Heading: {math.degrees(self.heading):.1f}°"
        ]

        y_offset = 45
        for line in info_lines:
            txt = font_info.render(line, True, (52, 73, 94))
            canvas.blit(txt, (panel_x, y_offset))
            y_offset += 22

        # Controls Guide
        y_offset += 20
        pygame.draw.line(canvas, (189, 195, 199), (panel_x, y_offset), (panel_x + 190, y_offset), 1)
        y_offset += 10
        canvas.blit(font_title.render("CHỌN MAP (1-5, H-Heldout):", True, (44, 62, 80)), (panel_x, y_offset))
        y_offset += 24

        controls = [
            "Phím 1-5: Đổi 5 Map Train",
            "Phím H: Test Map Held-out",
            "W / Up: Bay thẳng",
            "A / Left: Ngoặt trái 5°/15°",
            "D / Right: Ngoặt phải 5°/15°",
            "R: Reset Episode",
            "Q / ESC: Thoát"
        ]
        for ctrl in controls:
            canvas.blit(font_info.render(ctrl, True, (127, 140, 141)), (panel_x, y_offset))
            y_offset += 20

        if self.render_mode == "human":
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
