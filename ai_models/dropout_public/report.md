# Báo cáo dropout: baseline UCI cho nhóm học tập–tài chính

UCI là nguồn huấn luyện chính. OULAD là thực nghiệm bổ sung riêng về bài nộp và điểm; không trộn hai nguồn, không dùng VLE hoặc giả lập điểm danh.

## Dữ liệu và đặc trưng

- `dataset/processed/dropout_uci_center_baseline.csv`: Average_Score, Pass_Rate, Not_Evaluated_Rate, Has_Debt, Tuition_Up_To_Date.
- UCI: điểm kỳ 1 chia 2 về 0–10 khi có học phần đạt; Pass_Rate = học phần đạt / đăng ký; Not_Evaluated_Rate = học phần không đánh giá / đăng ký. Không đồng nhất thiếu đánh giá với không nộp bài.
- Has_Debt và Tuition_Up_To_Date giữ riêng hai cờ gốc; không suy ra số ngày quá hạn hoặc hóa đơn.
- `dataset/processed/dropout_oulad_assessment.csv`: Average_Score, Pass_Rate, Assignment_Missing_Rate, Assignment_Late_Count, Score_Trend; giữ khóa môn/kỳ/sinh viên.
- OULAD: học viên đã đăng ký và chưa hủy tại ngày 30; bài nộp đến ngày 30; thiếu bài chỉ tính trên bài đã đến hạn; nhãn Withdrawn vs Pass/Distinction/Fail.
- CSV giữ NaN và thang đo nghiệp vụ. Giá trị sai miền được ghi audit và chuyển thành NaN, không clip. Median imputation + missing indicators + StandardScaler được fit trong pipeline của từng fold.

## Protocol

Holdout 1/5 bằng StratifiedGroupKFold theo sinh viên; OOF 5 fold chỉ trên train. Hai cấu hình UCI dùng cùng train/test. Chọn model bằng average precision OOF và ngưỡng bằng F2 OOF. Test chỉ dùng để báo cáo.
UCI không có ID sinh viên gốc; student_id là ID dòng ổn định trong file nguồn. Không thể xác minh trùng danh tính thực. OULAD giữ ID sinh viên gốc để ngăn chồng lấn giữa các kỳ.

## Kết quả holdout

PR-AUC trong báo cáo được tính bằng average_precision_score; ROC-AUC không phải tỷ lệ dự đoán đúng.

| Cấu hình | Model | N test | Tỷ lệ dropout | ROC-AUC | PR-AUC | Recall | Precision | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline chính: học tập | Random Forest | 695 | 38.27% | 0.9033 | 0.8979 | 86.84% | 68.34% | 0.1085 |
| Thực nghiệm: học tập + tài chính | Logistic Regression | 695 | 38.27% | 0.9208 | 0.9209 | 87.22% | 76.07% | 0.0915 |
| Bổ sung: bài nộp và điểm | HistGradientBoosting | 5500 | 18.64% | 0.6703 | 0.3300 | 90.73% | 20.55% | 0.2219 |

### Baseline chính: học tập

Train/test: 2783/695. Feature: Average_Score, Pass_Rate, Not_Evaluated_Rate.
Model: Random Forest; ngưỡng F2 OOF: 0.257437. ROC-AUC CI 95% [0.8761, 0.9323]; AP CI 95% [0.8667, 0.9254] (200 bootstrap theo sinh viên).
TN=322, FP=107, FN=35, TP=231; gắn cờ 48.63% test.
Baseline prior: AP=0.3827, Brier=0.2363. Brier thấp hơn là tốt hơn.

| Model | AP OOF train | ROC-AUC test | AP test | F2 test |
|---|---:|---:|---:|---:|
| Dummy (prior) | 0.3795 | 0.5000 | 0.3827 | 0.7561 |
| Logistic Regression | 0.9022 | 0.8955 | 0.8933 | 0.8157 |
| Decision Tree | 0.8952 | 0.8986 | 0.8804 | 0.8232 |
| Random Forest | 0.9052 | 0.9033 | 0.8979 | 0.8238 |
| HistGradientBoosting | 0.9004 | 0.9008 | 0.8971 | 0.8181 |

### Thực nghiệm: học tập + tài chính

Train/test: 2783/695. Feature: Average_Score, Pass_Rate, Not_Evaluated_Rate, Has_Debt, Tuition_Up_To_Date.
Model: Logistic Regression; ngưỡng F2 OOF: 0.305887. ROC-AUC CI 95% [0.8958, 0.9475]; AP CI 95% [0.8949, 0.9452] (200 bootstrap theo sinh viên).
TN=356, FP=73, FN=34, TP=232; gắn cờ 43.88% test.
Baseline prior: AP=0.3827, Brier=0.2363. Brier thấp hơn là tốt hơn.

| Model | AP OOF train | ROC-AUC test | AP test | F2 test |
|---|---:|---:|---:|---:|
| Dummy (prior) | 0.3795 | 0.5000 | 0.3827 | 0.7561 |
| Logistic Regression | 0.9237 | 0.9208 | 0.9209 | 0.8473 |
| Decision Tree | 0.9125 | 0.9205 | 0.9060 | 0.8441 |
| Random Forest | 0.9227 | 0.9236 | 0.9223 | 0.8474 |
| HistGradientBoosting | 0.9192 | 0.9230 | 0.9215 | 0.8573 |

### Bổ sung: bài nộp và điểm

Train/test: 21943/5500. Feature: Average_Score, Pass_Rate, Assignment_Missing_Rate, Assignment_Late_Count, Score_Trend.
Model: HistGradientBoosting; ngưỡng F2 OOF: 0.361716. ROC-AUC CI 95% [0.6527, 0.6868]; AP CI 95% [0.3065, 0.3584] (200 bootstrap theo sinh viên).
TN=880, FP=3595, FN=95, TP=930; gắn cờ 82.27% test.
Baseline prior: AP=0.1864, Brier=0.1516. Brier thấp hơn là tốt hơn.

| Model | AP OOF train | ROC-AUC test | AP test | F2 test |
|---|---:|---:|---:|---:|
| Dummy (prior) | 0.1791 | 0.5000 | 0.1864 | 0.5339 |
| Logistic Regression | 0.3223 | 0.6600 | 0.3085 | 0.5379 |
| Decision Tree | 0.3320 | 0.6681 | 0.3249 | 0.5375 |
| Random Forest | 0.3336 | 0.6657 | 0.3291 | 0.5408 |
| HistGradientBoosting | 0.3370 | 0.6703 | 0.3300 | 0.5391 |

## Audit và phạm vi sử dụng

UCI: 4424 dòng gốc; loại 794 dòng chưa có nhãn Dropout/Graduate, 152 dòng không có mẫu số đăng ký hợp lệ; còn 3478 dòng.
OULAD: 32593 lượt gốc; còn 27443 lượt tại ngày 30. Chi tiết lỗi/loại mẫu: data_audit.json; missingness: missing_rates.csv.

- Baseline chính là `uci_academic_bundle.pkl`; cấu hình tài chính là thực nghiệm riêng, không tự động được chọn thay baseline vì điểm test cao hơn.
- UCI là dữ liệu đại học cuối kỳ 1, không chứng minh hiệu quả cảnh báo tuần 3–4 hoặc hiệu quả tại trung tâm. Không có chuyên cần thực trong các CSV này.
- Chưa xác minh thời điểm snapshot tài chính UCI và công bố điểm OULAD; kết quả không chứng minh đã loại hết leakage thời gian. Nhãn UCI là kết quả cuối cùng, không có ngày bỏ học để xác minh mọi ca đều xảy ra sau mốc kỳ 1.
- Điểm test thuộc artifact *_evaluation.pkl (chỉ fit train); *_bundle.pkl được refit toàn bộ nguồn sau đánh giá, không có phép kiểm thử độc lập riêng.
- API /predict-dropout dùng duy nhất uci_academic_bundle.pkl với ba feature học tập. Backend ánh xạ các bài đánh giá trung tâm thành proxy UCI; chạy từ điểm hợp lệ đầu tiên, không chờ 1/3 khóa. Chuyên cần và học phí dùng quy tắc riêng ngay cả khi chưa có điểm. Đây là tích hợp thử nghiệm; benchmark công khai không xác nhận hiệu quả tại trung tâm. Xem ../DROPOUT_INTEGRATION.md.
- UI hiện dùng điểm rủi ro tổng hợp hybrid_risk_1: AI học tập 50%, chuyên cần 30%, học phí 20%, phân bổ lại trọng số khi thiếu dữ liệu và có mức sàn cảnh báo. Đây là quy tắc nghiệp vụ ban đầu, không phải xác suất đã hiệu chỉnh. Chỉ số benchmark ở trên thuộc model công khai, không phải chất lượng của điểm tổng hợp tại trung tâm.
- Cần snapshot và nhãn thực tế của trung tâm để huấn luyện/kiểm chứng chuyên cần, học phí và cảnh báo trong khóa.
