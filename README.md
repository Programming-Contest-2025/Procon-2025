<!-- ![Koddaku](./psyduck-256x256.png) -->
<p align="center">
  <img src="./psyduck-256x256.png" width="200">
</p>

# 🧩 Procon Solving by Local Search Algorithm & Value Function Approximation

> **Project type:** Vietnam Student Olympiad in Informatics- Programming Con
test adapted from The Naprock Contest (Japan)  
> **Event:** [Progamming Contest 2025](https://www.olp.vn/procon-pmmn/procon)    
> **Goal:** Tối ưu hóa lời giải bài toán Procon bằng Local Search kết hợp học máy  

---

## 🏁 Giới thiệu về cuộc thi

**Procon (Programming Contest)** là cuộc thi lập trình thường niên nơi các đội thi giải quyết những bài toán tối ưu phức tạp trong thời gian giới hạn.  
Các bài toán thường có:
- Không gian trạng thái lớn  
- Đầu vào đa dạng, thay đổi liên tục  
- Nhiều nghiệm cục bộ (local optima)

Mục tiêu của nhóm là xây dựng **bộ giải (solver)** có khả năng:
1. Tìm ra nghiệm tốt cho hầu hết trường hợp đầu vào.
2. Cân bằng giữa **độ chính xác**, **thời gian chạy**, và **tính ổn định**.

---

## 🧠 Phương pháp đề xuất

### 1️⃣ Brute-force Evaluation
Trước tiên, nhóm tiến hành **Brute-force search** trên các bài test nhỏ để:
- Kiểm tra xem có tồn tại **nghiệm tối ưu tuyệt đối** hay không.
- Phân tích **mối quan hệ giữa không gian trạng thái** và **điểm đánh giá (score)**.
  
Việc này giúp xác định **giới hạn hiệu quả của Local Search** trong các bài có quy mô lớn.

---

### 2️⃣ Simulated Annealing (SA)
Trọng tâm của bộ giải là **thuật toán Simulated Annealing**, một biến thể của Local Search với khả năng:
- **Thoát khỏi local minima** nhờ cơ chế "làm nguội" (annealing schedule).
- **Khám phá đa dạng nghiệm** thông qua xác suất chấp nhận nghiệm kém.

**Ý tưởng chính:**
- Bắt đầu từ nghiệm ngẫu nhiên.
- Lặp lại quá trình sinh nghiệm mới → đánh giá → quyết định chấp nhận.
- Giảm dần "nhiệt độ" để hội tụ về nghiệm tốt.

---

### 3️⃣ Value Function Approximation (CNN Heuristic)
Để tăng tốc độ hội tụ, nhóm kết hợp **học sâu** nhằm ước lượng **hàm heuristic** hỗ trợ SA.

- Sử dụng **Convolutional Neural Network (CNN)** để học **Value Function** (ước lượng chất lượng trạng thái).  
- Dữ liệu huấn luyện được sinh ra từ các lần chạy SA và brute-force trước đó.  
- Kết quả giúp SA chọn hướng tìm kiếm tốt hơn, giảm thời gian tìm nghiệm.

---

### 4️⃣ Hậu xử lý kết quả (Post-processing)
Sau khi tìm được lời giải tạm thời, nhóm thực hiện bước **rút gọn và tối ưu hóa đường đi (solution simplification)**:

#### 4.1 Loại bỏ chuỗi quay vòng 360°
Nếu xuất hiện 4 thao tác liên tiếp tương đương với **một vòng quay 360°**, toàn bộ chuỗi này được loại bỏ.

#### 4.2 Rút gọn giữa các trạng thái trùng lặp
Khi hai trạng thái trùng lặp (đã từng xuất hiện), các bước trung gian giữa chúng bị xóa để rút ngắn lời giải mà vẫn đảm bảo tính hợp lệ.
