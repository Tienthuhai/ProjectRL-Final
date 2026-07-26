# 🛸 Tài Liệu Mô Tả Chi Tiết Môi Trường UAV Path Planning 2D (`UAVMapEnv`)

Tài liệu này tổng hợp toàn bộ thông số kỹ thuật, thiết kế không gian trạng thái (State Space), không gian hành động (Action Space), hàm thưởng (Reward Function) và danh sách các bản đồ huấn luyện / kiểm thử cho môi trường `UAVMapEnv`.

---

## 1. 📌 Tổng Quan Môi Trường

Môi trường `UAVMapEnv` là môi trường giả lập chuẩn **Gymnasium 2D** mô phỏng quá trình điều khiển thiết bị bay không người lái (UAV) lập đường bay tự động (Path Planning) và né tránh vật cản trên ảnh bản đồ.

* **Thư viện tích hợp**: `gymnasium`, `pygame` (visualize), `opencv-python` (đọc grid map), `numpy`.
* **Kích thước bản đồ tiêu chuẩn**: $500 \times 500$ pixels.
* **Tốc độ bay (Speed)**: $4.0$ pixels/step.
* **Tầm quét cảm biến LiDAR**: $150.0$ pixels (16 hướng quét $360^\circ$).
* **Số bước tối đa (Max Steps)**: $500$ steps/episode.

---

## 2. 🧠 Không Gian Trạng Thái (Observation Space - 22D Vector)

Observation là một véc-tơ liên tục **22 chiều (22D Box Vector)** trong khoảng $[-1.0, 1.0]$, giải quyết triệt để lỗi gián đoạn góc nghiêng bằng biểu diễn lượng giác ($\sin/\cos$):

| STT | Chiều dữ liệu | Kích thước | Khoảng giá trị | Mô tả |
| :---: | :--- | :---: | :---: | :--- |
| **1** | `lidar_distances` | 16D | $[0.0, 1.0]$ | Khoảng cách từ UAV tới vật cản theo 16 tia LiDAR quét quanh góc $360^\circ$ (chuẩn hóa theo `max_range = 150px`). |
| **2** | `target_dist` | 1D | $[0.0, 1.0]$ | Khoảng cách từ vị trí hiện tại của UAV tới điểm đích Goal (chuẩn hóa theo đường chéo bản đồ). |
| **3** | `sin(target_angle)` | 1D | $[-1.0, 1.0]$ | Sin của góc lệch tương đối giữa hướng bay hiện tại và hướng tới Goal. |
| **4** | `cos(target_angle)` | 1D | $[-1.0, 1.0]$ | Cos của góc lệch tương đối giữa hướng bay hiện tại và hướng tới Goal. |
| **5** | `sin(heading)` | 1D | $[-1.0, 1.0]$ | Sin của hướng góc quay hiện tại của UAV (Heading angle). |
| **6** | `cos(heading)` | 1D | $[-1.0, 1.0]$ | Cos của hướng góc quay hiện tại của UAV (Heading angle). |
| **7** | `min_lidar_distance`| 1D | $[0.0, 1.0]$ | Khoảng cách của tia LiDAR ngắn nhất hiện tại (Cảnh báo vật cản cận kề nguy hiểm). |

---

## 3. 🎮 Không Gian Hành Động (Action Space - 5 Discrete Actions)

Hành động của UAV là không gian rời rạc gồm **5 hành động (Discrete Action Space)**:

| Mã Action | Tên hành động | Góc xoay $\Delta\theta$ | Mô tả chi tiết |
| :---: | :--- | :---: | :--- |
| **`0`** | `ACTION_FORWARD` | $0^\circ$ | Giữ nguyên hướng bay và tiến lên $4.0\text{ px}$. |
| **`1`** | `ACTION_TURN_LEFT_LIGHT` | $+5^\circ$ ($+0.087\text{ rad}$) | Ngoặt nhẹ về bên trái và tiến lên. |
| **`2`** | `ACTION_TURN_LEFT_HEAVY` | $+15^\circ$ ($+0.262\text{ rad}$) | Ngoặt gắt về bên trái và tiến lên. |
| **`3`** | `ACTION_TURN_RIGHT_LIGHT`| $-5^\circ$ ($-0.087\text{ rad}$) | Ngoặt nhẹ về bên phải và tiến lên. |
| **`4`** | `ACTION_TURN_RIGHT_HEAVY`| $-15^\circ$ ($-0.262\text{ rad}$) | Ngoặt gắt về bên phải và tiến lên. |

---

## 4. 🎁 Hàm Thưởng & Điều Kiện Kết Thúc (Reward Function)

Hàm thưởng được thiết kế dạng **Reward Shaping** để định hướng UAV di chuyển nhanh và an toàn về đích:

### 🏆 Các mức Thưởng / Phạt:
1. **Chạm đích Goal ($\le 20\text{ px}$)**: **$+300.0$** *(Thắng - Kết thúc episode `terminated = True`)*
2. **Va chạm vật cản / Ra ngoài biên**: **$-100.0$** *(Thua - Kết thúc episode `terminated = True`)*
3. **Thưởng tiến gần Goal (Distance Shaping)**: **$+1.5 \times (d_{\text{prev}} - d_{\text{curr}})$**
   * Nếu tiến gần đích: Thưởng dương.
   * Nếu đi xa đích: Phạt âm.
4. **Phạt áp sát vật cản ($\min(\text{LiDAR}) < 15\text{ px}$)**: **$-0.5$** *(Khuyến khích giữ khoảng cách an toàn)*
5. **Phạt bước đi (Step Penalty)**: **$-0.2$** *(Khuyến khích tìm đường ngắn nhất)*

---

## 5. 🗺️ Chi Tiết Các Bản Đồ Môi Trường (Maps)

> [!TIP]
> Hệ thống áp dụng chiến lược thiết kế **Map Design v2** (xem chi tiết tại [`map_design_v2.md`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/map_design_v2.md)), tối ưu từ 9 map xuống **5 map đại diện (3 train + 2 eval)** và kết hợp cơ chế **Random Start/Goal** mỗi episode khi huấn luyện.

### 🎓 Tập Bản Đồ Huấn Luyện (Training Maps — 3 Giai Đoạn Curriculum)

| Giai đoạn | Tên File Bản Đồ | Độ khó | Đặc điểm chướng ngại vật & Kỹ năng huấn luyện chính |
| :---: | :--- | :---: | :--- |
| **Stage 1** | [`maps/map_1.png`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/maps/map_1.png) *(map_easy)* | **Dễ** | Vật cản thưa, phân bố đối xứng. Giúp UAV học định hướng đích và duy trì bay thẳng cơ bản. |
| **Stage 2** | [`maps/map_3.png`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/maps/map_3.png) *(map_medium)* | **Trung bình** | Vật cản chắn ngang giữa hành lang. Bắt buộc UAV đưa ra ra quyết định chọn luồng lách (Ngoặt trái/phải). |
| **Stage 3** | [`maps/map_4.png`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/maps/map_4.png) / [`map_5.png`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/maps/map_5.png) *(map_hard)* | **Khó** | Hẻm nhỏ + chùm vật cản phức hợp mật độ cao. Huấn luyện phản xạ bay sát tường và né tránh nguy hiểm. |

---

### 🧪 Tập Bản Đồ Đánh Giá & Kiểm Thử (Evaluation Maps — 3 Test Maps)

Các bản đồ này **KHÔNG** dùng trong quá trình huấn luyện, chỉ phục vụ đánh giá tính tổng quát:

| Tên Bản Đồ | Đặc điểm môi trường | Mục đích kiểm thử |
| :--- | :--- | :--- |
| [`maps/map_heldout.png`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/maps/map_heldout.png) | **Bản đồ Test Độc Lập**: Thiết kế hoàn toàn mới không trùng lặp với 3 map train. | Đánh giá khả năng Zero-shot Generalization của agent đã học. |
| [`maps/map_urban.png`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/maps/map_urban.png) | **Môi trường Đô thị**: Các khối nhà hình chữ nhật mật độ cao mô phỏng quy hoạch đô thị. | Kiểm thử phản xạ cảm biến LiDAR trong không gian hẹp phức tạp. |
| [`maps/map_dense.png`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/maps/map_dense.png) | **Môi trường Mật độ dày**: Các khoảng trống hẹp giữa chùm vật cản. | Kiểm thử phản xạ áp sát chướng ngại vật khẩn cấp. |


---

## 6. 💻 Hướng Dẫn Sử Dụng Code Python

### Nạp môi trường trong Python:
```python
from uav_env.uav_map_env import UAVMapEnv

# 1. Khởi tạo môi trường với bản đồ cụ thể
env = UAVMapEnv(
    map_path="maps/map_1.png",
    fixed_start_goal=True,
    render_mode="human"  # Hoặc "rgb_array" để xuất hình ảnh
)

# 2. Reset môi trường
obs, info = env.reset()

# 3. Chạy vòng lặp RL
terminated = False
truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()  # Chọn action (hoặc từ mô hình RL)
    obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

### Chạy Demo Tương Tác:
```bash
python demo_env.py
```
