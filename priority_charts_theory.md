# Lý Thuyết Biểu Đồ Ưu Tiên Khi Train DQN (a → e → i → g)

Tài liệu này tổng hợp lý thuyết cho 4 biểu đồ ưu tiên cao nhất khi ngân sách thời gian có hạn, theo thứ tự triển khai: **(a) Reward → (e) Rolling Success Rate → (i) Trajectory Plot + Video Demo → (g) Outcome Distribution**.

---

## (a) Reward theo Episode

**Ý nghĩa:** đo tổng phần thưởng tích lũy agent nhận được mỗi episode — thước đo cơ bản nhất cho biết agent có đang "học tốt hơn theo thời gian" hay không.

**Cách vẽ:**
- Trục X: số thứ tự episode. Trục Y: tổng reward của episode đó.
- Vẽ **2 đường chồng lên nhau**:
  - Đường **raw reward** (mờ/nhạt màu) — dữ liệu gốc từng episode.
  - Đường **moving average** (đậm, trung bình trượt 50–100 episode) — đường quan trọng để đánh giá xu hướng thật.

**Cách đọc:**
- Raw reward sẽ rất nhiễu (dao động mạnh) vì còn phụ thuộc epsilon-greedy (random action) và vị trí start/goal khác nhau mỗi lần → **bình thường, không đáng lo**.
- **Dấu hiệu tốt:** đường trung bình trượt tăng dần rồi ổn định (plateau) ở mức dương cao.
- **Dấu hiệu xấu:**
  - Đường phẳng gần 0 hoặc âm suốt quá trình → agent không học được gì.
  - Tăng rồi sụp đột ngột → training không ổn định (learning rate quá cao, hoặc replay buffer chưa đủ đa dạng).

---

## (e) Rolling Success Rate theo cửa sổ thời gian

**Ý nghĩa:** % episode agent tới đích thành công, tính theo cửa sổ trượt (vd mỗi 100–200 episode gần nhất).

**Cách vẽ:**
- Trục X: episode (hoặc block episode). Trục Y: % thành công.
- Với mỗi episode, tính tỉ lệ thành công trong N episode gần nhất (cửa sổ trượt), không tính cộng dồn toàn bộ lịch sử.

**Tại sao quan trọng hơn cả reward thô:**
- Reward là số liên tục, khó cảm nhận trực quan "agent giỏi cỡ nào".
- Success rate là con số % dễ hiểu (0–100%), thể hiện trực tiếp năng lực hoàn thành nhiệm vụ — đây là chỉ số **người đọc báo cáo/giám khảo sẽ hỏi đầu tiên**.
- **Bắt buộc dùng cửa sổ trượt (rolling)** thay vì cộng dồn: nếu tính cộng dồn toàn bộ lịch sử, 100 episode đầu toàn thất bại (agent còn random) sẽ kéo % trung bình xuống mãi mãi, che mất sự cải thiện thật sự gần đây.

**Cách đọc:**
- Đường tăng dần, có thể có vùng dao động ở giữa quá trình (do vẫn còn explore) nhưng xu hướng chung phải đi lên và ổn định dần về cuối.
- Nên đối chiếu với đường epsilon decay (nếu có vẽ (d)): success rate thường tăng rõ rệt khi epsilon xuống mức thấp.

---

## (i) Quỹ đạo bay trên bản đồ (Trajectory Plot) + Video Demo

### Phần 1: Trajectory Plot (ảnh tĩnh)

**Ý nghĩa:** biểu đồ "kể chuyện" trực quan nhất — người đọc không cần hiểu RL vẫn nhìn ra ngay agent bay tốt hay tệ.

**Cách vẽ:**
- Vẽ đường đi UAV (nối các vị trí (x, y) qua từng step) chồng lên ảnh bản đồ gốc.
- Đánh dấu: Start (chấm xanh), Goal (sao đỏ), điểm kết thúc (khác màu tùy outcome: thành công/va chạm/timeout).
- Nên vẽ **nhiều quỹ đạo trên nhiều map/nhiều cặp start-goal khác nhau** trong cùng 1 lưới ảnh (subplot), để không kết luận vội vàng chỉ từ 1 lần bay may mắn.

**Cách đọc:**
- Đường đi mượt, gần đường thẳng nối start–goal, né vật cản gọn gàng → policy tốt.
- Đường đi ngoằn ngoèo, lượn qua lượn lại nhiều lần quanh 1 khu vực (hiện tượng "wobbly" — giống DQN thất bại trong báo cáo tham khảo khi action space không đủ mịn) → dấu hiệu policy chưa ổn định.

**Mở rộng (j) — so sánh checkpoint:** cùng kỹ thuật này, có thể vẽ quỹ đạo tại các mốc training khác nhau (đầu/giữa/cuối) trên cùng 1 map để minh họa quá trình "agent học dần" theo thời gian — rất mạnh cho phần thuyết trình.

### Phần 2: Video Demo (MP4) — bổ sung theo yêu cầu

**Ý nghĩa:** trajectory plot cho thấy **kết quả cuối cùng** (đường đi), nhưng không cho thấy **quá trình ra quyết định theo thời gian thực** — vd tốc độ phản ứng khi LiDAR phát hiện vật cản, cách UAV điều chỉnh góc lái liên tục. Video giải quyết đúng khoảng trống này, đặc biệt hữu ích khi thuyết trình/bảo vệ đồ án.

**Những gì nên có trong video:**
1. **Video demo agent đã train xong** — chạy 1–3 episode trên các map khác nhau (dễ → khó, và cả map held-out) để minh họa khả năng tổng quát hóa trực quan, không chỉ bằng con số.
2. **HUD hiển thị trong lúc quay** (đã có sẵn trong thiết kế Pygame ở kế hoạch trước): Step hiện tại, Reward tích lũy, khoảng cách tới Goal, trạng thái (đang bay/va chạm/thành công) — giúp người xem hiểu ngữ cảnh mà không cần giải thích thêm bằng lời.
3. **16 tia LiDAR đổi màu theo độ nguy hiểm** (xanh → vàng → đỏ) — chi tiết này chỉ video mới thể hiện được rõ, vì nó thay đổi liên tục theo từng frame, khác với ảnh tĩnh chỉ chụp được 1 khoảnh khắc.
4. *(Tùy chọn, nếu muốn thể hiện sự tiến bộ)* — video ngắn ghép 3 đoạn: agent ở checkpoint sớm (bay lộn xộn) → giữa (đã khá) → cuối (mượt mà) để minh họa trực quan quá trình học, tương tự tinh thần của (j) nhưng ở dạng động thay vì ảnh tĩnh.

**Lưu ý khi quay:**
- Chỉ bật chế độ ghi hình đầy đủ (Pygame render, thu frame) khi **chạy demo agent đã train xong**, tuyệt đối không bật trong lúc training (vì sẽ làm chậm training nghiêm trọng — đã lưu ý ở phần thiết kế môi trường trước đó).
- Nên quay ở **nhiều map khác nhau** (ít nhất 1 map dễ trong tập train + 1 map held-out) để video vừa minh họa hành vi cơ bản, vừa chứng minh khả năng tổng quát hóa — nhất quán với thông điệp của biểu đồ (h).
- Video là phần **bổ trợ trực quan cho báo cáo/thuyết trình**, không thay thế cho các biểu đồ số liệu (a), (e), (g) — vẫn cần đủ số liệu định lượng để chứng minh kết quả một cách khoa học.

---

## (g) Phân bố kết quả cuối (Success / Collision / Timeout)

**Ý nghĩa:** biểu đồ chẩn đoán nguyên nhân thất bại — hữu ích để **debug reward function**, không chỉ để báo cáo kết quả.

**Cách vẽ:**
- Bar chart 3 cột: % Success / % Collision / % Timeout, tính trên tập đánh giá (vd 20–50 episode test sau khi train xong, có thể tách riêng cho từng map để chi tiết hơn).

**Cách đọc và hành động tương ứng:**
- **Nhiều Collision, ít Timeout** → agent hành động quá "liều", chưa đủ sợ vật cản → cân nhắc tăng phạt va chạm hoặc tăng phạt tiệm cận nguy hiểm.
- **Nhiều Timeout, ít Collision** → agent quá "rụt rè" (giống hiện tượng SAC trong báo cáo tham khảo: an toàn nhưng không dám tiến về đích) → cân nhắc tăng trọng số shaping reward theo khoảng cách, hoặc tăng phạt mỗi bước để thúc đẩy tiến nhanh hơn.
- **Success thấp nhưng Collision/Timeout đều thấp** → về mặt logic không nên xảy ra (mỗi episode luôn kết thúc bằng 1 trong 3 trạng thái) — nếu thấy vậy, kiểm tra lại điều kiện `done`/`truncated` trong code, có thể có bug.

---

## Tóm tắt thứ tự triển khai

| Thứ tự | Biểu đồ | Mục đích chính | Dữ liệu cần log |
|---|---|---|---|
| 1 | (a) Reward/episode | Xác nhận agent có học không | reward mỗi episode |
| 2 | (e) Rolling success rate | Đo năng lực hoàn thành nhiệm vụ theo thời gian | outcome (success/fail) mỗi episode |
| 3 | (i) Trajectory + Video | Trực quan hóa hành vi, dùng cho báo cáo/thuyết trình | vị trí (x,y), heading, lidar mỗi step; frame render (cho video) |
| 4 | (g) Outcome distribution | Chẩn đoán nguyên nhân thất bại để tinh chỉnh reward | outcome (success/collision/timeout) trên tập eval |
