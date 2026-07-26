# 🗺️ Thiết Kế Bản Đồ (Maps) — Phiên Bản Tối Ưu v2 Cho `UAVMapEnv`

Tài liệu này chuẩn hóa chiến lược thiết kế bản đồ cho hệ thống **UAV Path Planning 2D**, rút gọn quy mô từ **9 map (5 train + 4 eval)** xuống **5 map tinh gọn (3 train + 2 eval)**, áp dụng cơ chế **Random Start/Goal** và mô hình **Curriculum Learning (Huấn luyện phân cấp)** để tối ưu hóa hiệu quả học tăng cường và khả năng tổng quát hóa.

---

## 0. Cải Tiến Cốt Lõi: Cơ Chế Random Start/Goal Khử Overfitting

### ⚠️ Hạn chế ở phiên bản cũ (`fixed_start_goal=True`)
* Mỗi bản đồ chỉ có **1 cặp tọa độ $(Start, Goal)$ cố định duy nhất**.
* **Hậu quả**: Agent học thuộc lòng quỹ đạo đường bay cụ thể trên bản đồ thay vì học kỹ năng phản xạ né vật cản qua cảm biến LiDAR 16D (bị Overfitting nặng).

### 💡 Giải pháp ở phiên bản v2 (`fixed_start_goal=False`)
* Mỗi khi bắt đầu episode mới (`env.reset()`), vị trí $Start$ và $Goal$ sẽ được **sinh ngẫu nhiên (sample)** hoàn toàn tự động trong vùng khoảng trống (Free-space), đảm bảo:
  1. Khoảng cách an toàn tối thiểu tới chướng ngại vật $\ge 20\text{ px}$.
  2. Khoảng cách từ $Start$ tới $Goal$ tối thiểu $\ge 150\text{ px}$.

#### Code thực thi thực tế trong `uav_map_env.py`:
```python
def _sample_free_position(self, margin=20):
    """Sinh ngẫu nhiên 1 tọa độ trong vùng free-space cách tường tối thiểu margin px"""
    for _ in range(3000):
        x = float(self.np_random.uniform(margin, self.map_w - margin))
        y = float(self.np_random.uniform(margin, self.map_h - margin))
        if self._is_free(x, y, margin=margin):
            return np.array([x, y], dtype=np.float32)
    return np.array([50.0, 50.0], dtype=np.float32)

def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    ...
    if self.fixed_start_goal and self.map_path in self.DEFAULT_MAP_CONFIGS:
        # Cố định Start/Goal (dùng cho Demo / Debug)
        cfg = self.DEFAULT_MAP_CONFIGS[self.map_path]
        self.start_pos = np.array(cfg["start"], dtype=np.float32)
        self.goal_pos = np.array(cfg["goal"], dtype=np.float32)
    else:
        # Random Start/Goal ngẫu nhiên mỗi episode khi Training
        self.start_pos = self._sample_free_position(margin=20)
        while True:
            self.goal_pos = self._sample_free_position(margin=20)
            if np.linalg.norm(self.start_pos - self.goal_pos) >= 150.0:
                break
```

> [!NOTE]
> Tham số `fixed_start_goal` **vẫn được giữ lại trong code** (phục vụ cho `demo_env.py` / debug một kịch bản cố định), nhưng **luôn thiết lập `fixed_start_goal=False` khi chạy huấn luyện RL (`train_dqn.py`)**.

---

## 1. Tập Bản Đồ Huấn Luyện (Training Maps — 3 Giai Đoạn Curriculum)

Chiến lược **Curriculum Learning** chia quá trình huấn luyện thành 3 giai đoạn tăng dần độ khó:

| Giai đoạn | Tên bản đồ | Tên file cũ | Độ khó | Đặc điểm chướng ngại vật | Kỹ năng huấn luyện chính |
| :---: | :--- | :--- | :---: | :--- | :--- |
| **Stage 1** | `maps/map_easy.png` | `map_1.png` | **Dễ** | Vật cản thưa, phân bố đối xứng nhẹ | Bay thẳng, định hướng đích cơ bản, giữ độ ổn định |
| **Stage 2** | `maps/map_medium.png` | `map_3.png` | **Trung bình** | Vật cản chắn giữa hành lang bay | Ra quyết định chọn luồng lách (Ngoặt trái / Ngoặt phải) |
| **Stage 3** | `maps/map_hard.png` | `map_4.png` / `map_5.png` | **Khó** | Hẻm hẹp + chùm chướng ngại vật phức hợp mật độ cao | Bay sát tường, phản xạ lách hẻm, tổng hợp kỹ năng |

### 📈 Tiêu chí chuyển giai đoạn (Curriculum Transition Criteria):
* Chuyển từ **Stage 1 $\rightarrow$ Stage 2 $\rightarrow$ Stage 3** khi **Tỷ lệ thành công (Success Rate)** đạt $\ge 70\%$ trên 100–200 episodes gần nhất.

---

## 2. Tập Bản Đồ Đánh Giá (Evaluation Maps — 3 Maps, KHÔNG dùng để train)

Dùng để đánh giá khả năng **Zero-shot Generalization** và độ bền vững của mô hình sau huấn luyện trên 3 kịch bản môi trường khác nhau:

| Tên bản đồ | Đặc điểm cấu trúc | Mục đích kiểm thử |
| :--- | :--- | :--- |
| [`maps/map_heldout.png`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/maps/map_heldout.png) | Bố cục hoàn toàn mới, độc lập không trùng lắp với 3 map train | Đánh giá khả năng thích ứng môi trường lạ (Zero-shot Generalization) |
| [`maps/map_urban.png`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/maps/map_urban.png) | Khối nhà hình chữ nhật mật độ cao mô phỏng quy hoạch đô thị | Kiểm thử tình huống thực tế phức tạp trong đô thị |
| [`maps/map_dense.png`](file:///d:/University%20Course/RL%20N%C3%A2ng%20Cao/project%20RL/maps/map_dense.png) | Mật độ chướng ngại vật dày đặc, khoảng trống hẹp | Kiểm thử phản xạ khẩn cấp của cảm biến LiDAR khi áp sát vật cản |

---

## 3. Bảng So Sánh Trước vs Sau Cải Tiến

| Tiêu chí | Phiên bản cũ (v1) | Phiên bản mới (v2) | Lợi ích đạt được |
| :--- | :---: | :---: | :--- |
| **Số map Training** | 5 maps | **3 maps** (`map_easy`, `map_medium`, `map_hard`) | Tập trung ngân sách tính toán, giúp hội tụ nhanh hơn |
| **Số map Evaluation** | 4 maps | **3 maps** (`map_heldout`, `map_urban`, `map_dense`) | Đánh giá đa dạng trên 3 loại hình môi trường test |
| **Tổng số maps** | **9 maps** | **6 maps** | Đơn giản hóa dự án, rút ngắn thời gian huấn luyện |
| **Vị trí Start/Goal** | Cố định per map | **Random per episode** | **Triệt tiêu 100% hiện tượng học vẹt (Overfitting)** |
| **Chiến lược Training** | Độc lập / Phẳng | **Curriculum 3 cấp độ** | Tăng tốc độ hội tụ và độ ổn định của RL Agent |

---

## 4. 🎯 5 Lý Do Thiết Kế Mới Tối Ưu Cho Đồ Án / Báo Cáo

1. **Triệt tiêu hiện tượng Overfitting**: Việc random vị trí $Start/Goal$ mỗi episode buộc UAV phải dựa hoàn toàn vào nhận diện cảm biến 16D LiDAR để ra quyết định thay vì học thuộc lòng tọa độ đường bay.
2. **Loại bỏ trùng lặp kỹ năng**: Bỏ các bản đồ không mang lại tri thức mới (như `map_2` trùng kỹ năng với `map_1` do tính bất biến góc $\sin/\cos$).
3. **Tối ưu ngân sách tính toán (Training Budget)**: Số bước bước bay (timesteps) được tập trung luyện kỹ trên 3 cấu trúc vật cản đại diện thay vì bị dàn mỏng trên 9 map.
4. **Vô số kịch bản sinh tự động**: Một bản đồ duy nhất khi kết hợp với random $Start/Goal$ sẽ tự sinh ra hàng ngàn kịch bản đường bay khác nhau (từ chéo, ngang, dài, ngắn, áp sát tường,...).
5. **Cấu trúc luận văn/báo cáo mạch lạc**: Phân chia rõ ràng Curriculum 3 cấp độ (Dễ $\rightarrow$ Trung bình $\rightarrow$ Khó) và 2 map Test độc lập giúp phần thực nghiệm có tính thuyết phục và chuyên nghiệp cao.
