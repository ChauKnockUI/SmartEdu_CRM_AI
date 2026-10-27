# Báo cáo mô hình Dropout Risk v3 (SmartEdu CRM)

## 1. Thiết lập
- UCI: mốc cuối học kỳ 1 (không dùng dữ liệu học kỳ 2). OULAD: mốc Day 30.
- Chia Train/Test và CV bằng `StratifiedGroupKFold` theo `id_student` (không sinh viên nào xuất hiện ở cả hai tập).
- Đặc trưng không quan sát được ở một nguồn để NaN (không bịa), được impute trung vị + indicator trong pipeline.
- Mô hình được chọn bằng PR-AUC out-of-fold trên **tập train**: **HistGradientBoosting**. Ngưỡng chọn theo F2 trên OOF.

## 2. Benchmark (test set)
| Mô hình | OOF PR-AUC (train) | ROC-AUC | PR-AUC | Recall | Precision | F2 | Brier |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Dummy (prior) | 0.2032 | 0.5000 | 0.1985 | 1.0000 | 0.1985 | 0.5533 | 0.1592 |
| Logistic Regression | 0.4771 | 0.6998 | 0.4473 | 0.8994 | 0.2305 | 0.5691 | 0.2065 |
| Decision Tree | 0.5155 | 0.7189 | 0.4847 | 0.9351 | 0.2151 | 0.5601 | 0.1945 |
| Random Forest | 0.5310 | 0.7304 | 0.5074 | 0.8620 | 0.2482 | 0.5768 | 0.1877 |
| HistGradientBoosting | 0.5342 | 0.7289 | 0.5065 | 0.8709 | 0.2455 | 0.5769 | 0.1939 |

Mô hình chọn: ROC-AUC 95% CI [0.7115, 0.7444], PR-AUC 95% CI [0.4791, 0.5298] (bootstrap theo sinh viên).

## 3. Kết quả theo từng nguồn (HistGradientBoosting)
| Nguồn | N | Tỉ lệ dropout | ROC-AUC | PR-AUC | Recall | Precision |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| OULAD | 5,510 | 17.99% | 0.6781 | 0.3437 | 0.8618 | 0.2139 |
| UCI | 696 | 34.63% | 0.9163 | 0.9032 | 0.9087 | 0.5778 |

Kiểm tra shortcut: ROC-AUC của baseline "chỉ biết nguồn dữ liệu" = **0.5521**.
Nếu AUC gộp cao hơn rõ rệt so với AUC từng nguồn, mô hình đang tận dụng khác biệt giữa hai dataset chứ không chỉ hành vi sinh viên.

## 4. Confusion matrix (HistGradientBoosting)
TN=1,674 | FP=3,300 | FN=159 | TP=1,073

## 5. Chuyển miền (chỉ dùng đặc trưng chung: Average_Score, Assignment_Missing_Rate, Pass_Rate)
| Kịch bản | ROC-AUC | PR-AUC | Base rate đích |
|:--|:-:|:-:|:-:|
| UCI → OULAD | 0.6022 | 0.2556 | 0.1834 |
| OULAD → UCI | 0.7842 | 0.6953 | 0.3864 |
| In-domain UCI (OOF, cùng đặc trưng) | 0.9114 | 0.9041 | - |
| In-domain OULAD (OOF, cùng đặc trưng) | 0.6677 | 0.3276 | - |

Khoảng chênh giữa chuyển miền và in-domain cho thấy mức độ khái quát hóa thực sự.

## 6. Hạn chế
- Hai tập có định nghĩa "bỏ học" khác nhau (UCI: Dropout vs Graduate; OULAD: Withdrawn sau Day 30 vs còn lại).
- Đặc trưng là proxy (VLE ≠ điểm danh thực; UCI không có dữ liệu chuyên cần). Cần hiệu chỉnh/huấn luyện lại bằng dữ liệu thật của SmartEdu.
- Chưa đánh giá công bằng (fairness) theo nhóm nhân khẩu học.
