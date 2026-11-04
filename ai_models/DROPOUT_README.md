# Dropout: quy trình hiện hành

Điểm bắt đầu duy nhất cho thực nghiệm dữ liệu công khai: **`train_dropout_public.py`**.
Kết quả hiện hành nằm trong **[dropout_public/report.md](dropout_public/report.md)**. Không dùng các kết quả gộp UCI/OULAD cũ để mô tả model này.

## Chạy

Từ thư mục `SmartEdu_CRM_AI`:

```powershell
.\venv\Scripts\python.exe -B -m unittest discover -s ai_models -p test_dropout_public.py -v
.\venv\Scripts\python.exe -B -u ai_models/train_dropout_public.py
```

Dữ liệu đầu vào là file UCI và bốn file OULAD trong `dataset/archive`. Không đọc `studentVle.csv` và không dùng dữ liệu mô phỏng trong pipeline này.

## Hai CSV độc lập

| File trong `dataset/processed` | Phạm vi | Feature |
|---|---|---|
| `dropout_uci_center_baseline.csv` | Nguồn chính, cuối học kỳ 1 | `Average_Score`, `Pass_Rate`, `Not_Evaluated_Rate`, `Has_Debt`, `Tuition_Up_To_Date` |
| `dropout_oulad_assessment.csv` | Nguồn bổ sung, ngày 30 | `Average_Score`, `Pass_Rate`, `Assignment_Missing_Rate`, `Assignment_Late_Count`, `Score_Trend` |

CSV giữ NaN để phân biệt thông tin không quan sát được; không điền sẵn trên toàn bộ dữ liệu. Điểm về 0–10, tỷ lệ về 0–1. Giá trị sai miền được chuyển thành NaN và ghi trong audit, không clip che lỗi. Imputer và StandardScaler được fit trên từng fold train trong pipeline đã lưu.

`record_id`, `student_id`, `Data_Source`, mốc quan sát và nhãn không phải feature. OULAD giữ khóa môn/kỳ/sinh viên để không làm mất danh tính lượt ghi danh. Không tạo cột chuyên cần giả, không gán học phí UCI cho người OULAD.

Điểm UCI tổng hợp khi không có học phần đạt được giữ là NaN. Điểm 0 được quan sát của một bài OULAD được giữ là 0. Không có bài đến hạn khác với có bài đến hạn nhưng không nộp: trường hợp đầu có tỷ lệ thiếu bài NaN.

## Model và đánh giá

Ba cấu hình được benchmark độc lập:

1. `uci_academic`: ba feature học tập — đối chứng và dự phòng khi thiếu học phí.
2. `uci_academic_finance`: năm feature học tập và học phí — model tích hợp khi có thông tin học phí; thời điểm snapshot tài chính UCI vẫn chưa xác minh.
3. `oulad_assessment`: bài nộp và điểm — kết quả bổ sung, không phải dữ liệu trung tâm.

Hai cấu hình UCI dùng chính xác cùng train/test và cùng fold OOF. Chọn thuật toán bằng AP OOF của train, chọn ngưỡng F2 bằng OOF; không chọn theo test. CSV chưa impute có thể đưa thẳng vào pipeline bằng cách chọn danh sách `features` trong bundle.

```python
import joblib
import pandas as pd

bundle = joblib.load("ai_models/dropout_public/uci_academic_finance_bundle.pkl")
df = pd.read_csv("dataset/processed/dropout_uci_center_baseline.csv")
probability = bundle["pipeline"].predict_proba(df[bundle["features"]])[:, 1]
```

Ví dụ trên minh họa cách gọi pipeline; dự đoán trên chính CSV đã train không phải phép kiểm thử độc lập. Xem `*_holdout_predictions.csv` và `report.md` để đánh giá.

Trong `ai_models/dropout_public`:

- `report.md`, `benchmark.json`, `dashboard.png`: kết quả hiện hành.
- `data_audit.json`, `missing_rates.csv`: kiểm tra chất lượng và dòng bị loại.
- `uci_split.csv`, `oulad_split.csv`: danh sách train/test.
- `*_evaluation.pkl`: estimator chỉ fit train, tái hiện điểm test.
- `*_bundle.pkl`: refit toàn bộ nguồn sau đánh giá; có feature, schema, hash và metadata.
- `*_holdout_predictions.csv`: dự đoán trên test, kèm nhãn và khóa.
- `cleanup_manifest.json`: danh sách file kết quả/pipeline cũ đã bỏ trong lần chuyển đổi này.

## Giới hạn nghiệp vụ và API

Đây là baseline nghiên cứu cho học tập–tài chính. UCI không phải dữ liệu trung tâm và không chứng minh cảnh báo sớm giữa khóa. Thời điểm tài chính UCI, ngày bỏ học UCI và thời điểm công bố điểm OULAD chưa đầy đủ để xác minh tính sẵn có của mọi tín hiệu tại mốc dự báo. Chưa có chuyên cần thực hoặc xác nhận hiệu quả tại SmartCRM.

`api_server.py` nạp `dropout_public/uci_academic_finance_bundle.pkl` cho `/predict-dropout` với năm feature: `Average_Score`, `Pass_Rate`, `Not_Evaluated_Rate`, `Has_Debt`, `Tuition_Up_To_Date`. Hai cờ học phí nhận 0/1 hoặc null. Khi một cờ học phí chưa biết, endpoint dùng `uci_academic_bundle.pkl` với ba feature học tập, không điền giả học phí. Phản hồi có `includes_finance` và `model_version` để backend nhận biết đúng model. Không dùng model legacy v1/v2. OULAD vẫn là thực nghiệm riêng.

Backend tính proxy từ bài đánh giá hiện có và hóa đơn của đúng học viên–lớp. `Has_Debt` là còn dư nợ kể cả chưa đến hạn; `Tuition_Up_To_Date` là không có khoản còn dư nợ đã quá hạn. Thiếu hóa đơn thì hai cờ là null; còn dư nợ nhưng không có hạn thanh toán thì trạng thái đúng hạn là unknown. Không dùng hóa đơn hủy/hoàn tiền hoặc giao dịch tương lai. Đây là ánh xạ nghiệp vụ cần kiểm chứng, không khẳng định tương đương hoàn toàn với UCI. Model chạy từ điểm hợp lệ đầu tiên; chưa có điểm vẫn kiểm tra chuyên cần/học phí bằng quy tắc. Xem **[DROPOUT_INTEGRATION.md](DROPOUT_INTEGRATION.md)**.

UI hiển thị **điểm rủi ro tổng hợp** theo `hybrid_risk_2`: AI học tập và học phí 70% + chuyên cần 30% khi có model năm feature. Không cộng học phí thành trọng số riêng lần nữa. Khi dùng model học tập dự phòng hoặc AI không sẵn sàng, chính sách là 50/30/20 cho học tập/chuyên cần/học phí, phân bổ lại khi thiếu thành phần. Mức sàn cảnh báo vẫn được giữ. Đây là chính sách 0–100 chưa hiệu chỉnh thành xác suất. Điểm AI gốc lưu tại `model_score`; `academic_score` chỉ có khi AI dùng riêng học tập. Các chỉ số report.md chỉ đánh giá model trên dữ liệu công khai.

Khi có dữ liệu trung tâm, cần snapshot điểm danh, học tập, học phí và kết quả bỏ học xảy ra sau snapshot; huấn luyện và xác nhận trên dữ liệu tương lai trước khi thay model phục vụ thực tế.
