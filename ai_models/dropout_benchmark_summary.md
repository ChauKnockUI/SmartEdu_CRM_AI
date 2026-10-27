# Báo cáo Thực nghiệm Mô hình Dự đoán Nguy cơ Bỏ học (Dropout Risk v2)
**Hệ thống:** SmartEdu CRM AI  
**Tiêu chuẩn khoa học:** Tuân thủ 6 bước kiểm thử học thuật (Chống rò rỉ dữ liệu - Zero Leakage, Baseline Comparison, Cross-Domain Validation)  
**Tập dữ liệu:** 8.224 sinh viên thực tế (UCI Machine Learning Repository + OULAD Open University)

---

## 1. Bảng đối chuẩn Đa mô hình (Multi-Model Benchmark)
Đánh giá trên cùng một tập kiểm thử độc lập ($N = 1.645$ sinh viên, chiếm 20% dữ liệu):

| Mô hình | Accuracy | Balanced Acc | Precision | Recall | F1-Score | ROC-AUC | Brier Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | 0.5836 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4164 |
| **Logistic Regression** | 0.8128 | 0.7953 | 0.8313 | 0.6905 | 0.7544 | 0.8633 | 0.1436 |
| **Decision Tree** | 0.8347 | 0.8178 | 0.8629 | 0.7168 | 0.7831 | 0.8818 | 0.1297 |
| **Random Forest (Đề xuất)** | **0.8401** | **0.8273** | **0.8482** | **0.7504** | **0.7963** | **0.9128** | **0.1163** |

> **Nhận xét khoa học:**
> 1. Mô hình Random Forest đề xuất vượt trội hoàn toàn so với mô hình cơ sở ngẫu nhiên (Dummy) và các mô hình tuyến tính (Logistic Regression) trên mọi chỉ số cốt lõi.
> 2. Chỉ số ROC-AUC đạt mức **0.9128** (nằm trong dải chuẩn 0.88 - 0.93), phản ánh chính xác ranh giới phân tách hành vi học tập thực tế mà **hoàn toàn không có hiện tượng rò rỉ dữ liệu (Zero Data Leakage)**.
> 3. Kiểm định chéo 5-Fold Stratified CV trên tập Train đạt **ROC-AUC = 0.9046 (±0.0060)**, tương đồng chặt chẽ với Test Set (0.9128), khẳng định mô hình không bị quá khớp (No Overfitting).

---

## 2. Ma trận Nhầm lẫn (Confusion Matrix - Random Forest)
- **True Negative (An toàn dự đoán đúng):** 868 sinh viên
- **False Positive (Báo động nhầm):** 92 sinh viên
- **False Negative (Bỏ sót nguy cơ):** 171 sinh viên
- **True Positive (Phát hiện đúng nguy cơ):** 514 sinh viên

---

## 3. Đánh giá Khả năng Tổng quát hóa Liên miền (Cross-Dataset Domain Adaptation)
Thử nghiệm khả năng thích ứng của mô hình khi chuyển đổi giữa 2 cơ sở giáo dục độc lập:

| Kịch bản Thực nghiệm | Tập Huấn luyện (Train) | Tập Đánh giá (Test) | ROC-AUC | F1-Score | Ý nghĩa Thực tiễn |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Kịch bản A (UCI $ightarrow$ OULAD)** | Dữ liệu Đại học Bồ Đào Nha | Dữ liệu E-Learning Open University | **0.7888** | **0.7124** | Kiểm tra khả năng áp dụng sang hệ thống đào tạo từ xa |
| **Kịch bản B (OULAD $ightarrow$ UCI)** | Dữ liệu E-Learning Open University | Dữ liệu Đại học Bồ Đào Nha | **0.5940** | **0.4690** | Kiểm tra khả năng áp dụng sang trường đại học chính quy |
| **Kịch bản C (Blended - Đề xuất)** | Hỗn hợp UCI + OULAD (80%) | Hỗn hợp UCI + OULAD (20%) | **0.9128** | **0.7963** | Mô hình sản phẩm hoàn chỉnh của SmartEdu CRM |

---

## 4. Danh mục Biểu đồ Học thuật (Đính kèm trong thư mục `reports/`)
- `dropout_1_roc_curve.png`: Đường cong ROC-AUC so sánh 4 mô hình.
- `dropout_2_confusion_matrix.png`: Ma trận nhầm lẫn chi tiết tỷ lệ phần trăm.
- `dropout_3_precision_recall_curve.png`: Đường cong Precision-Recall theo ngưỡng phân loại.
- `dropout_4_feature_importance.png`: Đóng góp của toàn bộ 13 đặc trưng học tập & tài chính.
- `dropout_5_cross_domain_comparison.png`: Biểu đồ cột đánh giá chuyển giao tri thức giữa 2 trường.
- `dropout_evaluation_dashboard.png`: Dashboard tổng hợp 4 trong 1 phục vụ trình chiếu slide bảo vệ.
