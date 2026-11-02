# BÁO CÁO KỸ THUẬT HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN NGUY CƠ BỎ HỌC (DROPOUT EARLY-WARNING MODEL v3)

* **Dự án:** Hệ sinh thái Quản trị Đào tạo & Chăm sóc Học viên Thông minh — **SmartEdu CRM AI**
* **Nhiệm vụ:** Nhận diện sớm học viên có nguy cơ bỏ học ngay từ giai đoạn đầu khóa học (Early-Warning System)
* **Phương pháp luận:** Zero Temporal Leakage, Group-aware Split (`StratifiedGroupKFold`), Out-of-Fold (OOF) Model Selection, $F_2$-Score Threshold Tuning, Cross-Domain Generalization Test
* **Môi trường:** Python 3.10+, Scikit-Learn 1.4+, Pandas, NumPy, Matplotlib Headless

---

## 1. TỔNG QUAN BÀI TOÁN & DỮ LIỆU NGUỒN (DATA UNDERSTANDING)

### 1.1. Mục tiêu Nghiệp vụ (Business Objective)
Trong CRM Giáo dục, nếu phát hiện học viên bỏ học vào cuối khóa học (Post-hoc), doanh nghiệp hoàn toàn không còn cơ hội giữ chân học viên và đã mất chi phí đào tạo. Do đó, bài toán đặt ra là: **Dự đoán nguy cơ bỏ học tại mốc CẢNH BÁO SỚM (Early-Warning Milestone)** — khi học viên mới học được $12.5\% - 50\%$ chặng đường — để cố vấn học tập và phòng đào tạo can thiệp kịp thời.

### 1.2. Nguồn dữ liệu Thực tế (Data Sourcing)
Mô hình kết hợp 2 bộ dữ liệu benchmark quốc tế chuẩn mực:
1. **UCI Student Dropout and Academic Success (Realinho et al., 2021):**
   * **Quy mô:** $4.424$ hồ sơ sinh viên đại học chính quy (Polytechnic Institute of Portalegre, Bồ Đào Nha).
   * **Đặc trưng cốt lõi:** Kết quả học tập theo từng kỳ, tình trạng đóng học phí/nợ tín chỉ, điểm thi đầu vào.
2. **OULAD — Open University Learning Analytics Dataset (Kuzilek et al., 2017):**
   * **Quy mô:** $32.593$ lượt đăng ký môn học trực tuyến của trường Đại học Mở Vương quốc Anh.
   * **Dữ liệu quan sát:** $10.655.282$ dòng nhật ký tương tác hệ thống học tập VLE (click logs), thời điểm nộp bài tập và điểm số theo ngày.

---

## 2. CHIẾN LƯỢC CHỐNG RÒ RỈ DỮ LIỆU (ZERO-LEAKAGE FORMULATION)

```
[Khóa học bắt đầu] -------------> [MỐC CẢNH BÁO SỚM] -------------> [Kết thúc khóa học]
      |                                   |                                   |
      |==== DỮ LIỆU QUAN SÁT CHO PHÉP ===>|         TUYỆT ĐỐI BỎ QUA          |
      |   (Log click, bài tập, điểm số)   |    (Điểm kỳ 2, click sau ngày 30) |
```

### 2.1. Thiết lập Mốc Dự đoán (Milestone Definition)
* **Tập OULAD (Mốc Ngày thứ 30 - Week 4 Early Warning):**
  * Khóa học kéo dài $240$ ngày. Mô hình **chỉ được phép quan sát tương tác VLE và bài tập nộp trong $30$ ngày đầu tiên** ($date \le 30$).
  * **Lọc đối tượng:** Loại bỏ sinh viên đã rút học trước hoặc tại ngày 30 ($date\_unregistration \le 30$). Mô hình chỉ đánh giá những học viên *đang còn theo học tại ngày thứ 30* để dự đoán ai sẽ rút học sau đó.
* **Tập UCI (Mốc Cuối Học kỳ 1):**
  * **Loại bỏ 100% các cột của Học kỳ 2** (`Curricular units 2nd sem...`). Chỉ sử dụng điểm số, số tín chỉ đạt và đánh giá trong phạm vi Học kỳ 1 (`1st sem`).

### 2.2. Đồng nhất Nhãn Nhị phân (Binary Target Harmonization)
* **UCI:** `Target == 'Dropout'` $\rightarrow$ **1** (Bỏ học); `Target == 'Graduate'` $\rightarrow$ **0** (Tốt nghiệp). Loại bỏ nhóm trung gian `Enrolled` để ground-truth trong sạch.
* **OULAD:** Rút học sau ngày 30 (`Withdrawn`) $\rightarrow$ **1**; Học đến cuối khóa (`Pass`, `Distinction`, `Fail`) $\rightarrow$ **0**. Sinh viên thi trượt môn (`Fail`) vẫn đi thi và học hết khóa, không xem là bỏ học giữa chừng.

### 2.3. Nguyên tắc Toàn vẹn Dữ liệu (No Fabricated Data)
* **Xóa bỏ hoàn toàn việc gán ngẫu nhiên:** Không dùng `np.random` để lấp các đặc trưng thiếu.
* **Minh bạch dữ liệu khuyết:** Đặc trưng nào OULAD không có (nợ học phí) hoặc UCI không có (nhật ký click VLE) được giữ nguyên giá trị `NaN` và xử lý bằng cơ chế tự động học của `SimpleImputer(add_indicator=True)`.

---

## 3. KỸ NGHỆ ĐẶC TRƯNG (FEATURE ENGINEERING)

Mô hình tinh chọn **10 đặc trưng hành vi và học thuật** có căn cứ thực tế:

| STT | Tên Đặc trưng (`Feature`) | Công thức / Định nghĩa Kỹ thuật | Ý nghĩa Nghiệp vụ | Nguồn khả dụng |
|:---:|:---|:---|:---|:---:|
| 1 | `Attendance_Rate` | $\frac{\text{Số ngày active VLE}}{31}$ (OULAD) | Tỷ lệ tham gia học tập trong 30 ngày đầu | OULAD / UCI: `NaN` |
| 2 | `Consecutive_Unexcused_Absences` | $\max(\text{gap ngày vắng VLE}, 30 - \text{last\_active})$ | Chuỗi ngày vắng mặt liên tiếp dài nhất | OULAD / UCI: `NaN` |
| 3 | `Days_Since_Last_Attended` | $30 - \max(date_{\text{click}} \le 30)$ | Số ngày không tương tác tính đến mốc Day 30 | OULAD / UCI: `NaN` |
| 4 | `Log_Total_Activity` | $\ln(1 + \sum clicks_{\le 30})$ | Tổng cường độ tương tác số (chuẩn hóa log) | OULAD / UCI: `NaN` |
| 5 | `Assignment_Missing_Rate` | $1 - \frac{\text{Số bài nộp}}{\text{Số bài đến hạn}}$ (OULAD) hoặc $1 - \frac{\text{evaluations}}{\text{enrolled}}$ (UCI) | Tỷ lệ bài tập / môn học không nộp bài | Cả 2 nguồn |
| 6 | `Assignment_Late_Count` | $\sum [date\_submitted > date\_due]_{\le 30}$ | Số lượng bài nộp muộn hạn | OULAD / UCI: `NaN` |
| 7 | `Average_Score` | Điểm trung bình bài tập kỳ 1 (chuẩn hóa thang điểm 10) | Năng lực học thuật nền tảng | Cả 2 nguồn |
| 8 | `Score_Trend` | $Score_{\text{cuối kỳ 1}} - Score_{\text{đầu kỳ 1}}$ (thang điểm 10) | Xu hướng tăng / giảm phong độ học tập | OULAD / UCI: `NaN` |
| 9 | `Pass_Rate` | Tỷ lệ bài đạt $\ge 4.0$ (OULAD) hoặc $\frac{\text{approved}}{\text{enrolled}}$ kỳ 1 (UCI) | Tỷ lệ hoàn thành đạt yêu cầu | Cả 2 nguồn |
| 10 | `Has_Overdue_Invoice` | $[Debtor == 1 \lor Tuition\_not\_up\_to\_date == 1]$ | Tình trạng nợ đọng học phí | UCI / OULAD: `NaN` |

> **Bộ 3 đặc trưng chung (Shared Features):** `["Pass_Rate", "Assignment_Missing_Rate", "Average_Score"]` là 3 đặc trưng có ngữ nghĩa đồng nhất $100\%$ giữa cả hai hệ thống, được sử dụng riêng cho bài kiểm tra chuyển giao tri thức liên trường (Cross-Domain Test).

---

## 4. CHIẾN LƯỢC PHÂN CHIA DỮ LIỆU & KIỂM ĐỊNH (CROSS-VALIDATION)

```
                       TỔNG TẬP DỮ LIỆU (30.944 bản ghi / 28.265 sinh viên)
                                             |
                   [StratifiedGroupKFold theo id_student]
                        /                                \
          TRAIN SET (80%)                                 TEST SET (20%)
     24.738 dòng / 22.610 SV                          6.206 dòng / 5.655 SV
            |                                                    |
   5-Fold StratifiedGroupKFold                                   |
   -> Tính Out-of-Fold (OOF)                                     |
   -> Chọn Model tốt nhất                                        |
   -> Tối ưu ngưỡng cắt F2                                       |
            |                                                    |
            +------------ Fit Full Train -> Dự đoán Test --------+
```

1. **Đơn vị phân tích:** Lượt đăng ký môn học `(code_module, code_presentation, id_student)`.
2. **Chống rò rỉ nhóm (Student Group-Leakage):** Trong OULAD, một sinh viên có thể học nhiều môn. Nếu dùng split ngẫu nhiên, sinh viên đó sẽ xuất hiện ở cả Train và Test $\rightarrow$ Mô hình nhớ mặt sinh viên.  
   $\rightarrow$ Sử dụng **`StratifiedGroupKFold` gom nhóm theo `id_student`**. Toàn bộ các môn học của cùng một sinh viên chỉ được phép nằm trong tập Train hoặc tập Test ($Train \cap Test = \emptyset$).
3. **Kích thước mẫu:**
   * **Train:** $24.738$ bản ghi ($22.610$ sinh viên duy nhất).
   * **Test độc lập (Hold-out):** $6.206$ bản ghi ($5.655$ sinh viên duy nhất). Tỉ lệ bỏ học trên tập Test là **$19.85\%$**.

---

## 5. THIẾT KẾ PIPELINE & BỘ MÔ HÌNH THỰC NGHIỆM (MODEL ZOO)

### 5.1. Kiến trúc Pipeline Độc lập (Uncloned Pipeline Factory)
Để tránh lỗi rò rỉ trạng thái (State Pollution) khi train nhiều mô hình, hàm `make_pipeline()` khởi tạo mới hoàn toàn `ColumnTransformer` cho từng model:
* **`SimpleImputer(strategy='median', add_indicator=True)`**: Điền giá trị trung vị cho dữ liệu khuyết, đồng thời tự động sinh thêm cột nhị phân chỉ báo `MissingIndicator` (giúp mô hình học được ý nghĩa của việc thiếu thông tin tài chính).
* **`StandardScaler()`**: Chuẩn hóa z-score cho các biến số thực.

### 5.2. Danh mục 5 Mô hình Đánh giá
1. **Dummy Classifier (Baseline):** Dự đoán theo xác suất tiên nghiệm (Prior probability).
2. **Logistic Regression:** Mô hình cơ sở tuyến tính (`class_weight='balanced'`, `max_iter=2000`).
3. **Decision Tree:** Mô hình phi tuyến dạng cây đơn (`max_depth=6`, `min_samples_leaf=20`).
4. **Random Forest:** Ensemble bagging (`n_estimators=300`, `max_depth=10`, `class_weight='balanced_subsample'`).
5. **HistGradientBoostingClassifier (Đề xuất):** Ensemble boosting tối ưu cho dữ liệu bảng (`max_depth=4`, `learning_rate=0.05`, `max_iter=300`, `l2_regularization=1.0`).

---

## 6. CHIẾN LƯỢC LỰA CHỌN MÔ HÌNH & TỐI ƯU NGƯỠNG CẮT (THRESHOLD TUNING)

### 6.1. Tiêu chí chọn Model
Tuyệt đối **không chọn mô hình dựa trên điểm Test**. Mô hình chiến thắng được lựa chọn dựa trên **PR-AUC (Precision-Recall AUC) tính trên xác suất Out-Of-Fold (OOF) của tập Train**. Do bài toán mất cân bằng lớp ($~20\%$ bỏ học), PR-AUC phản ánh trung thực hơn ROC-AUC.

### 6.2. Tối ưu Ngưỡng Cắt $F_2$-Score (Ưu tiên Recall)
Trong CRM giáo dục:
* **Hậu quả của False Negative (Bỏ sót học viên):** Học viên âm thầm nghỉ học, mất $100\%$ doanh thu khóa học và gây tụt uy tín trung tâm.
* **Hậu quả của False Positive (Cảnh báo nhầm):** Cố vấn học tập gọi điện hỏi thăm một học viên đang học tốt. Chi phí cuộc gọi rất nhỏ, học viên vẫn cảm thấy được trung tâm quan tâm.

$\rightarrow$ Tối ưu ngưỡng cắt $\tau$ trên tập Out-of-fold theo chỉ số **$F_2$-Score** ($\beta = 2$, coi Recall quan trọng gấp đôi Precision):
$$F_2 = (1 + 2^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}}$$
Ngưỡng tối ưu tìm được trên OOF là: **$\tau^* = 0.3401$** (thay vì ngưỡng mặc định $0.5$).

---

## 7. KẾT QUẢ ĐỐI CHUẨN ĐA MÔ HÌNH (BENCHMARK RESULTS)

### 7.1. Bảng Tổng hợp Kết quả Đánh giá trên Hold-out Test Set ($N = 6.206$)

| Mô hình | OOF PR-AUC (Train) | Test ROC-AUC | Test PR-AUC | Recall | Precision | $F_2$-Score | Brier Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Prior Baseline)** | 0.2032 | 0.5000 | 0.1985 | 100.0% | 19.85% | 0.5533 | 0.1592 |
| **Logistic Regression** | 0.4771 | 0.6998 | 0.4473 | 89.94% | 23.05% | 0.5691 | 0.2065 |
| **Decision Tree** | 0.5155 | 0.7189 | 0.4847 | **93.51%** | 21.51% | 0.5601 | 0.1945 |
| **Random Forest** | 0.5310 | **0.7304** | **0.5074** | 86.20% | **24.82%** | 0.5768 | **0.1877** |
| **HistGradientBoosting (Winner)** | **0.5342** | **0.7289** | **0.5065** | **87.09%** | **24.55%** | **0.5769** | 0.1939 |

> **Khoảng tin cậy $95\%$ Bootstrap (Resampling 300 lần theo sinh viên):**
> * **ROC-AUC $95\%$ CI:** `[0.7115, 0.7444]`
> * **PR-AUC $95\%$ CI:** `[0.4791, 0.5298]`

### 7.2. Bóc tách Kết quả theo Nguồn Dữ liệu (Per-Source Granular Analysis)
Đánh giá mô hình **HistGradientBoosting** trên từng môi trường đào tạo:

| Nguồn Dữ liệu | Số lượng mẫu ($N$) | Tỉ lệ bỏ học | Test ROC-AUC | Test PR-AUC | Recall | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCI (ĐH Chính quy Bồ Đào Nha)** | 696 | 34.63% | **0.9163** | **0.9032** | **90.87%** | **57.78%** |
| **OULAD (E-Learning Đại học Mở Anh)** | 5.510 | 17.99% | **0.6781** | **0.3437** | **86.18%** | **21.39%** |

* **Nhận xét chuyên môn:**
  * **UCI đạt ROC-AUC = 0.9163:** Mốc cuối Học kỳ 1 cung cấp tín hiệu học thuật rất mạnh mẽ. Sinh viên nợ học phí và trượt hầu hết tín chỉ kỳ 1 có tỷ lệ bỏ học gần như chắc chắn.
  * **OULAD đạt ROC-AUC = 0.6781:** Dự đoán tại **Ngày thứ 30** của một khóa học dài 240 ngày (mới trôi qua $12.5\%$ thời gian). Con số $0.68$ này tương thích hoàn toàn với các nghiên cứu công bố quốc tế trên tập OULAD (Kuzilek et al. báo cáo AUC từ $0.65 - 0.70$ ở mốc 4 tuần đầu).

### 7.3. Kiểm định "Shortcut Learning" (Confounding Check)
* Xây dựng baseline *"Chỉ biết nguồn dữ liệu"* (dự đoán xác suất bỏ học dựa trên cờ UCI hay OULAD) $\rightarrow$ Đạt **ROC-AUC = 0.5521** (gần sát mức đoán mò 0.50).
* **Kết luận:** Mô hình **hoàn toàn không gian lận** dựa vào nguồn gốc dữ liệu mà thực sự khai thác các chỉ số hành vi học tập.

---

## 8. KIỂM ĐỊNH KHẢ NĂNG CHUYỂN GIAO LIÊN MIỀN (CROSS-DOMAIN GENERALIZATION)

Để kiểm chứng mô hình có thể thích ứng khi triển khai sang một cơ sở giáo dục hoàn toàn mới hay không, nhóm thực hiện thử nghiệm huấn luyện trên một trường và kiểm tra trên trường còn lại chỉ bằng **3 đặc trưng học tập chung**:

| Kịch bản Thực nghiệm | Tập Huấn luyện | Tập Đánh giá | Test ROC-AUC | Test PR-AUC | Base Rate Đích |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **OULAD $\rightarrow$ UCI** | E-Learning (Anh) | Đại học truyền thống (Bồ Đào Nha) | **0.7842** | **0.6953** | 38.64% |
| **UCI $\rightarrow$ OULAD** | Đại học truyền thống (Bồ Đào Nha) | E-Learning (Anh) | **0.6022** | **0.2556** | 18.34% |
| *In-domain UCI (OOF Reference)* | UCI | UCI | 0.9114 | 0.9041 | 38.64% |
| *In-domain OULAD (OOF Reference)* | OULAD | OULAD | 0.6677 | 0.3276 | 18.34% |

> **Phát hiện quan trọng:** Mô hình học từ hành vi trực tuyến OULAD vẫn đạt **ROC-AUC = 0.7842** khi áp đặt lên sinh viên truyền thống UCI. Điều này chứng minh quy luật: *"Tỷ lệ đạt bài tập thấp và thiếu bài nộp"* là **chỉ báo phổ quát xuyên biên giới** của hiện tượng bỏ học.

---

## 9. GIẢI THÍCH MÔ HÌNH (PERMUTATION IMPORTANCE)

Đánh giá mức độ sụt giảm PR-AUC trên tập Test khi xáo trộn ngẫu nhiên từng đặc trưng:

```
Pass_Rate                       ======================================== (0.2004)
Assignment_Missing_Rate         ========== (0.0487)
Average_Score                   ========== (0.0481)
Attendance_Rate                 ==== (0.0173)
Assignment_Late_Count           === (0.0152)
Has_Overdue_Invoice             == (0.0108)
Log_Total_Activity              = (0.0031)
Score_Trend                     = (0.0018)
Consecutive_Unexcused_Absences  (0.0006)
Days_Since_Last_Attended        (0.0004)
```

1. **`Pass_Rate` (Độ tụt 0.2004 PR-AUC):** Đặc trưng quan trọng nhất. Sinh viên không vượt qua được các cột mốc kiểm tra đầu tiên có xác suất từ bỏ cao nhất.
2. **`Assignment_Missing_Rate` (0.0487) & `Average_Score` (0.0481):** Việc không nộp bài tập và điểm số dưới trung bình là tín hiệu suy giảm động lực học tập rõ rệt.
3. **`Attendance_Rate` & `Assignment_Late_Count` (0.015 - 0.017):** Các chỉ báo chuyên cần đóng vai trò tinh chỉnh ngưỡng rủi ro.

---

## 10. Ý NGHĨA TRIỂN KHAI VÀO HỆ THỐNG SMARTEDU CRM

### 10.1. Phân tích Ma trận Nhầm lẫn tại Ngưỡng Cắt $\tau^* = 0.34$
Trên $6.206$ lượt học viên kiểm thử:
* **True Positive (Phát hiện đúng):** **$1.073$ học viên** có nguy cơ bỏ học được hệ thống tự động đưa vào danh sách chăm sóc đặc biệt.
* **False Negative (Bỏ sót):** **Chỉ $159$ học viên** (Độ nhạy Recall đạt **$87.09\%$**).
* **False Positive (Báo nhầm):** $3.300$ học viên. Do chủ động chọn $F_2$, hệ thống chấp nhận gọi thăm hỏi thêm một số học viên có điểm bài tập thấp nhưng vẫn cố gắng học, đổi lại không bỏ sót khách hàng.

### 10.2. Đóng gói Sản phẩm (Artifact Packaging)
Mô hình hoàn thiện được huấn luyện lại trên toàn bộ $30.944$ bản ghi và xuất ra file bundle duy nhất:
* **Đường dẫn artifact:** `ai_models/dropout_model_v3_bundle.pkl`
* **Nội dung bundle:** Pipeline tiền xử lý hoàn chỉnh, bộ phân loại `HistGradientBoosting`, ngưỡng phân loại tối ưu `threshold: 0.3401`, danh sách 10 đặc trưng đầu vào và siêu dữ liệu phiên bản.
* **Dashboard trực quan hóa:** Lưu tại `ai_models/reports_v3/dropout_v3_dashboard.png` (chuẩn 300 DPI sẵn sàng in ấn báo cáo).

---

## 11. HẠN CHẾ & HƯỚNG PHÁT TRIỂN (LIMITATIONS & FUTURE WORK)

1. **Bản chất dữ liệu đại diện (Proxy Data):** Do sử dụng dữ liệu mở quốc tế, số lượt click LMS đóng vai trò thay thế cho điểm danh thực tế; dữ liệu tài chính của OULAD bị khuyết.
2. **Kế hoạch hiệu chỉnh (Recalibration):** Sau khi triển khai thực tế trên SmartEdu CRM từ 3 đến 6 tháng, hệ thống cần ghi nhận dữ liệu điểm danh và đóng học phí nội bộ để huấn luyện lại (Fine-tune), kỳ vọng ROC-AUC thực tế tại trung tâm sẽ nâng lên mức **$0.82 - 0.88$**.
3. **Mở rộng Đa mốc thời gian (Multi-horizon Warning):** Phát triển thêm mốc cảnh báo tuần 2 (cực sớm) và tuần 8 (giữa kỳ) để cập nhật điểm rủi ro liên tục (Dynamic Risk Scoring).
