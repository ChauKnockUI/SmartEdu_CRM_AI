# Cảnh báo bỏ học: điểm tổng hợp AI và quy tắc

## Luồng hoạt động

`Dữ liệu trung tâm → AI học tập và học phí + chuyên cần → một điểm tổng hợp → mức hỗ trợ và thông báo`.

Điểm hiển thị dưới dạng % tương ứng thang 0–100 của **mức rủi ro theo chính sách**. Ví dụ 70% nghĩa là 70/100 điểm ưu tiên hỗ trợ, chưa có nghĩa 70 trong 100 người tương tự sẽ bỏ học. Chưa có dữ liệu trung tâm để hiệu chỉnh thành xác suất thực tế.

AI service nạp `dropout_public/uci_academic_finance_bundle.pkl` làm model chính. `/predict-dropout` nhận `Average_Score`, `Pass_Rate`, `Not_Evaluated_Rate`, `Has_Debt`, `Tuition_Up_To_Date`. Khi hai cờ học phí đều là 0/1, phản hồi dùng version `uci_academic_finance_public_1`, `includes_finance=true`. Khi thiếu ít nhất một cờ, dùng `uci_academic_bundle.pkl` với ba feature học tập, version `uci_academic_public_1`, `includes_finance=false`. Không điền giả học phí. Backend giữ điểm AI gốc tại `model_score`; `academic_score` chỉ dành cho model học tập dự phòng.

## Chính sách tính điểm `hybrid_risk_2`

Tập trung ở `SmartEdu_CRM_BE/src/services/ai/dropoutRisk.policy.ts`, gồm một hàm thuần `combinedRisk`:

| Thành phần | Điểm 0–100 | Trọng số ban đầu |
|---|---|---:|
| AI học tập và học phí | Điểm AI năm feature | 70% |
| Chuyên cần | Theo mức vắng không phép bên dưới | 30% |

Khi AI đã có học phí, không có thành phần học phí có trọng số riêng. Khi AI dùng model học tập dự phòng hoặc không sẵn sàng, giữ ba thành phần học tập/chuyên cần/học phí với trọng số ban đầu 50/30/20. Quy tắc sàn học phí bên dưới vẫn áp dụng để giữ ưu tiên hỗ trợ; đây không phải một lần cộng trọng số học phí thứ hai.

**Điểm nền** = tổng `(điểm thành phần × trọng số)` / tổng trọng số của các thành phần có dữ liệu.

Không có điểm AI thì không điền AI =0. Không có ghi nhận điểm danh thì chuyên cần là unknown. Không có hóa đơn hoặc còn dư nợ nhưng thiếu hạn thanh toán thì tài chính có thể unknown. Hóa đơn chưa đến hạn/đã trả đủ là dữ liệu quan sát được. `total_components` là 2 khi AI chứa học phí, 3 khi dùng chính sách dự phòng; UI không cố định số thành phần. Mọi thành phần đều unknown thì tổng hợp null, không tự tạo %.

Ví dụ chỉ có chuyên cần và học phí: trọng số thực tế là 60% và 40%. UI ghi rõ thiếu nhóm học tập, hiển thị số nhóm đã có và trọng số áp dụng. Nếu AI tạm lỗi, bỏ AI khỏi tính toán và thông báo rõ trạng thái đó.

### Chuyên cần

- Vắng liên tiếp >=3 buổi, hoặc vắng >=4 buổi và chiếm >=40% buổi đã có điểm danh: 100 điểm.
- Vắng liên tiếp >=2 buổi, hoặc vắng >=3 buổi và chiếm >=20% buổi đã có điểm danh: 75 điểm.
- Chưa đạt hai mức trên: min(35, tỷ lệ vắng không phép ×100).
- Không vắng: 0. Thiếu điểm danh không tự biến thành absent; excused không phải vắng không phép. Khoảng trống điểm danh làm gián đoạn chuỗi vắng để tránh tự suy diễn.

### Học phí

Chỉ xét hóa đơn của đúng học viên–lớp đã tồn tại tại lúc dự đoán, loại cancelled/refunded. Dư nợ = tổng tiền − giảm giá − các giao dịch đã thu đến thời điểm hiện tại; không dùng thanh toán tương lai. Dư nợ thực được ưu tiên hơn status hóa đơn.

- `Has_Debt=1` khi còn ít nhất một khoản dư nợ, kể cả chưa đến hạn; bằng 0 khi không còn dư nợ.
- `Tuition_Up_To_Date=0` khi còn ít nhất một khoản dư nợ đã quá hạn; bằng 1 khi các khoản đến hạn đều đã thanh toán, kể cả còn khoản chưa đến hạn.
- Không có hóa đơn: cả hai cờ null. Còn dư nợ thiếu hạn thanh toán: cờ đúng hạn null nếu chưa có khoản quá hạn xác định. Giá trị tiền sai miền khiến cờ tài chính unknown.
- Hai cờ này là proxy CRM, chưa được xác nhận tương đương hoàn toàn với hai cờ UCI. Thời điểm ghi nhận tài chính UCI vẫn chưa xác minh.

Điểm quy tắc học phí dùng trong chính sách dự phòng:

- Không quá hạn: 0 điểm.
- Quá hạn 1–7 ngày: 50 điểm.
- Quá hạn 8–14 ngày: 70 điểm.
- Quá hạn >14 ngày: 90 điểm.

### Giữ cảnh báo quan trọng và suy ra mức hỗ trợ

Điểm cuối = max(điểm nền, mức sàn), làm tròn một chữ số thập phân:

- AI học tập/học phí vượt ngưỡng OOF hoặc chuyên cần đạt mức cần liên hệ: sàn 70.
- Còn dư nợ quá hạn: sàn 40; quá hạn >14 ngày: sàn 70.
- Không có căn cứ sàn: giữ điểm nền.

Mức hỗ trợ lấy duy nhất từ điểm cuối: <40 chưa có cảnh báo; 40–<70 cần theo dõi; >=70 cần liên hệ. UI, lưu DB và thông báo dùng cùng điểm/mức. Mức sàn giữ cảnh báo không bị pha loãng bởi các thành phần tốt. Các trọng số, ngưỡng, mức sàn là chính sách ban đầu, **không phải hệ số đã học hoặc đã kiểm chứng bằng dữ liệu trung tâm**.

Điểm nền, thành phần, trọng số và lý do nâng sàn được lưu trong snapshot để kiểm tra. UI chỉ hiển thị kết quả, căn cứ chính và việc cần làm, không đưa bảng tính cho người dùng vận hành. Ví dụ AI gốc 30/100 nhưng tổng hợp 70/100 nếu AI vượt ngưỡng OOF là quy đổi ưu tiên xử lý, không phải xác suất AI tăng lên 70%.

## Dữ liệu học tập và thời điểm chạy

Theo dõi học viên ghi danh active trong lớp ongoing. Không chờ 1/3 khóa. AI được gọi từ điểm hợp lệ đầu tiên; chưa có điểm vẫn tính tổng hợp từ chuyên cần/học phí nếu các nhóm đó đã có dữ liệu.

- Dùng test đã đến ngày thi; bài tập đã đến hạn hoặc đã nộp và có điểm sớm. Loại bài được miễn, bài tương lai chưa được chấm và sự kiện chưa xảy ra.
- Điểm quy về 0–10 bằng max_score; thiếu max_score thì coi thang 10. Điểm sai miền được coi là chưa có điểm, không clip.
- Pass_Rate = số bài đạt / tổng bài đánh giá được xét. Not_Evaluated_Rate = số bài chưa có điểm hợp lệ / tổng bài được xét, gồm cả bài đã nộp nhưng chưa chấm.
- Average_Score là trung bình các bài đạt; không có bài đạt thì null, tương ứng proxy aggregate grade UCI. `DROPOUT_PASS_SCORE` mặc định 5/10.
- UCI dùng học phần cuối kỳ, trung tâm dùng bài đánh giá trong khóa. Đây là mapping bằng proxy, cần kiểm chứng; không đồng nhất hai bối cảnh chỉ bằng chuẩn hóa.

## Lưu trữ, thông báo và UI

- AIPrediction lưu snapshot năm feature, `finance_known`, điểm AI gốc, model version, `includes_finance`, kết quả hybrid, chính sách/version, mức hỗ trợ, dữ liệu thiếu và thời điểm. Cảnh báo từ AI kết hợp không được gán nguyên nhân riêng cho học tập.
- Mỗi kết quả thuộc riêng một cặp học viên–lớp. Hồ sơ hiển thị cột “Nguy cơ bỏ học” trong tab “Lớp đang học”; không có phần trăm chung cho mọi lớp. Student vẫn lưu bản tóm tắt lớp có điểm cao nhất để tương thích dữ liệu cũ, nhưng UI không dùng bản tóm tắt đó làm kết quả của lớp.
- Tận dụng cột Json `dropout_risk_reasons` để lưu envelope có version, reasons và details. Repository vẫn trả `dropout_risk_reasons` dạng mảng để tương thích UI; thêm `dropout_risk_type` và `dropout_risk_details` cho chi tiết. Không cần migration.
- Kết quả cũ, kể cả `hybrid_risk_1`, được đánh dấu academic_legacy và yêu cầu đánh giá lại; không coi điểm cũ là kết quả của model mới. Job định kỳ hoặc nút đánh giá sẽ tạo snapshot hybrid mới.
- Khi tổng hợp >=40, gửi **một cảnh báo tổng hợp cho mỗi người nhận**, gồm quản trị viên active và giáo viên phụ trách active trong đúng tenant. Link mở tab lớp của hồ sơ và chi tiết đúng lớp qua `?tab=classes&risk_class=:classId`. Không tự gửi tin học viên.
- Chống lặp 24 giờ theo người nhận + học viên + lớp + mức hỗ trợ. Từ medium lên high được gửi ngay, không bị cảnh báo medium cũ chặn. Advisory lock theo tenant/sinh viên bảo vệ cập nhật đồng thời.
- Tự cập nhật sau điểm danh, cập nhật bài nộp, hoặc nút đánh giá. Job khởi động sau 30 giây, quét mỗi 60 phút; `DROPOUT_SCAN_INTERVAL_MINUTES` tối thiểu 5. Job giữ ngữ cảnh tenant cho mọi truy vấn nghiệp vụ.
- Trang lớp đọc snapshot qua GET `/classes/:id/dropout-risk`; hồ sơ đọc kết quả theo từng lớp qua GET `/students/:id/dropout-risk`. Giáo viên chỉ đọc đánh giá thuộc lớp được phân công. Chỉ mở trang không chấm lại hoặc tạo thông báo.
- Các màn hình dùng chung phần trăm, mức hỗ trợ, tối đa ba lý do ngắn, một câu hướng xử lý và thời điểm cập nhật. Không hiển thị bảng trọng số, điểm nền, log `[Tổng hợp]`/`[Dữ liệu]` hoặc nhiều hộp thông báo xanh. Thông tin thiếu được gói trong một dòng; lưu ý ý nghĩa phần trăm đặt trong tooltip. Chuông thông báo dùng polling 30 giây sẵn có.
- Bảng chỉ hiển thị phần trăm theo lớp. Bấm phần trăm hoặc “Đánh giá” mở drawer bên phải, ghi rõ học viên và tên lớp, gồm ba mục ngắn: điểm danh, bài tập/kiểm tra, học phí; kèm nút đánh giá lại đúng lớp đang xem. Lớp chưa có kết quả không mượn điểm lớp khác. Trang Cảnh báo bỏ học chọn một lớp trước khi liệt kê học viên; danh sách học viên chung dẫn tới “Xem theo lớp”.
- Drawer hiển thị “Tham gia 2/3 buổi đã điểm danh”, số buổi vắng không phép/liên tiếp, số bài đạt/có điểm và học phí quá hạn. Tỷ lệ tham gia chỉ dùng buổi đã ghi nhận; thiếu điểm danh không tự tính là vắng. Lịch sử đóng học phí trễ được ghi riêng, không đồng nhất với dư nợ đang quá hạn và không tự tăng điểm.
- Nút Quay lại trên các trang chi tiết lùi một bước lịch sử trong ứng dụng; khi truy cập trực tiếp không có lịch sử thì về trang nguồn hoặc danh sách tương ứng. Tab đang xem của lớp được lưu trong query `tab` bằng replace, tránh tạo thêm các bước back chỉ vì đổi tab. Menu chọn theo mục cha của trang chi tiết; hồ sơ học viên mở từ lớp giữ mục Lớp học được đánh dấu.
- Khi mở hồ sơ từ lớp hoặc buổi học, giữ classId trong navigation state và mặc định mở tab lớp đang học. Đổi tab hay mở drawer dùng replace và giữ navigation state; nút quay lại vẫn lùi đúng một trang, giữ nguyên tab lớp đã mở.

## Kiểm tra và áp dụng

```powershell
# SmartEdu_CRM_BE: mock DB, không gửi thông báo thật
npm test -- tests/dropout-risk.test.ts tests/dropout-ui.test.ts tests/dropout-hybrid.test.ts
npm run typecheck
# AdminDashboard
npm run build
```

Huấn luyện bằng `venv/Scripts/python.exe -B -u ai_models/train_dropout_public.py` trong SmartEdu_CRM_AI. Chạy `venv/Scripts/python.exe -B -m unittest test_dropout_integration -v` để kiểm tra model thật và API. Khởi động lại AI service và backend, tải frontend mới, rồi bấm đánh giá lại. Nếu chạy Docker AI thì rebuild image để lấy mã và bundle mới. Không cần migration cơ sở dữ liệu.

Benchmark trong `dropout_public/report.md` chỉ đánh giá các model công khai, không phải chất lượng của điểm hybrid. Cần đối chiếu cảnh báo với kết quả thực tế để điều chỉnh chính sách và huấn luyện model tổng hợp từ dữ liệu trung tâm khi đủ lịch sử.
