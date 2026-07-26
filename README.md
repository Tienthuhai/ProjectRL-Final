# UAV Path Planning — DQN with Curriculum Learning

Dự án nghiên cứu huấn luyện UAV tự động lập đường bay 2D né tránh vật cản sử dụng **Thuật toán Deep Q-Network (DQN)** với **Curriculum Learning 3 giai đoạn** trên môi trường Gymnasium tùy chỉnh.

---

## Cấu Trúc Dự Án

```
project RL/
├── uav_env/
│   ├── __init__.py
│   └── uav_map_env.py          # Gymnasium Environment (UAVMapEnv)
├── maps/
│   ├── map_easy.png            # Train Stage 1
│   ├── map_medium.png          # Train Stage 2
│   ├── map_hard.png            # Train Stage 3
│   ├── map_heldout.png         # Test 1: Zero-shot Generalization
│   ├── map_urban.png           # Test 2: Urban City Blocks
│   └── map_dense.png           # Test 3: Dense Obstacles
├── train_dqn.py                # Script huấn luyện DQN (Stable-Baselines3)
├── plot_results.py             # Vẽ 4 biểu đồ kết quả từ CSV log
├── record_demo.py              # Quay video demo agent trên 6 map
├── demo_env.py                 # Demo môi trường tương tác Pygame
├── environment_description.md  # Tài liệu mô tả chi tiết môi trường
├── map_design_v2.md            # Thiết kế bản đồ phiên bản tối ưu
├── priority_charts_theory.md   # Lý thuyết các biểu đồ ưu tiên
├── requirements.txt
└── .gitignore
```

---

## Môi Trường (UAVMapEnv)

- **Observation Space**: Box(22,) — 16D LiDAR + 6D thông tin hướng/vị trí (sin/cos encoding)
- **Action Space**: Discrete(5) — Tiến thẳng, Ngoặt trái/phải nhẹ (5°) và gắt (15°)
- **Bản đồ**: 500×500 px, ảnh nhị phân (trắng = tự do, đen = vật cản)
- **Tốc độ UAV**: 4 px/step | **LiDAR**: 16 tia, tầm 150 px

---

## Chiến Lược Huấn Luyện

### Curriculum Learning — 3 Giai Đoạn
| Stage | Bản đồ | Độ khó | Điều kiện chuyển stage |
|:---:|:---|:---:|:---|
| 1 | `map_easy.png` | Dễ | Rolling Success Rate ≥ 70% (200 ep) |
| 2 | `map_medium.png` | Trung bình | Rolling Success Rate ≥ 70% (200 ep) |
| 3 | `map_hard.png` | Khó | — |

### Early Stopping
Dừng huấn luyện sớm khi: SR ≥ 85% **VÀ** cải thiện < 2% trong 500 episode liên tiếp (hoặc đạt 300,000 timesteps).

### Hyperparameters
| Tham số | Giá trị |
|:---|:---:|
| Learning rate | 1e-4 |
| Buffer size | 100,000 |
| Batch size | 64 |
| Gamma (discount) | 0.99 |
| Exploration ε | 1.0 → 0.05 (30% timesteps) |
| Network arch | [256, 256] |
| Max grad norm | 10 |

---

## Cài Đặt & Chạy

```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Huấn luyện DQN
python train_dqn.py

# 3. Vẽ biểu đồ kết quả
python plot_results.py

# 4. Quay video demo 6 map
python record_demo.py

# 5. Demo môi trường tương tác (Pygame)
python demo_env.py
```

---

## Kết Quả Đầu Ra

Sau khi chạy `train_dqn.py`:
- `logs/train_log.csv` — Log 7 cột mỗi episode (episode, timestep, reward, length, outcome, stage, rolling_SR)
- `models/dqn_best.zip` — Checkpoint model tốt nhất (theo success rate)
- `models/dqn_final.zip` — Model sau khi kết thúc huấn luyện

Sau khi chạy `plot_results.py`:
- `results/charts/a_reward_per_episode.png`
- `results/charts/e_rolling_success_rate.png`
- `results/charts/g_outcome_distribution.png`
- `results/charts/x_episode_length.png`

Sau khi chạy `record_demo.py`:
- `results/videos/demo_<map_name>.mp4` — Video riêng cho từng map (×6)
- `results/videos/demo_all_maps.mp4` — Video ghép lưới 2×3 cả 6 map

---

## Yêu Cầu Hệ Thống

- Python 3.10+
- stable-baselines3 ≥ 2.0
- gymnasium ≥ 0.29
- pygame, opencv-python, numpy, matplotlib, pandas
