# Cảnh báo bỏ học: điểm tổng hợp AI và quy tắc

## Luồng hoạt động

`Dữ liệu trung tâm → AI học tập + chuyên cần + học phí → một điểm tổng hợp → mức hỗ trợ và thông báo`.

Điểm hiển thị dưới dạng % tương ứng thang 0–100 của **mức rủi ro theo chính sách**. Ví dụ 70% nghĩa là 70/100 điểm ưu tiên hỗ trợ, chưa có nghĩa 70 trong 100 người tương tự sẽ bỏ học. Chưa có dữ liệu trung tâm để hiệu chỉnh thành xác suất thực tế.

AI service vẫn chỉ nạp `dropout_public/uci_academic_bundle.pkl`. `/predict-dropout` nhận `Average_Score`, `Pass_Rate`, `Not_Evaluated_Rate` và trả điểm AI gốc cùng cờ vượt ngưỡng OOF. Backend giữ riêng điểm AI và thực hiện phép tổng hợp; không cần train lại model hoặc đổi hợp đồng API AI.

## Chính sách tính điểm `hybrid_risk_1`

Tập trung ở `SmartEdu_CRM_BE/src/services/ai/dropoutRisk.policy.ts`, gồm một hàm thuần `combinedRisk`:

| Thành phần | Điểm 0–100 | Trọng số ban đầu |
|---|---|---:|
| Học tập | Điểm AI gốc | 50% |
| Chuyên cần | Theo mức vắng không phép bên dưới | 30% |
| Học phí | Theo số ngày quá hạn và dư nợ thực | 20% |

**Điểm nền** = tổng `(điểm thành phần × trọng số)` / tổng trọng số của các thành phần có dữ liệu.

Không có điểm AI thì không điền AI =0. Không có ghi nhận điểm danh thì chuyên cần là unknown. Không có hóa đơn có hạn hoặc đã thanh toán thì tài chính có thể unknown; hóa đơn chưa đến hạn/đã trả đủ là dữ liệu tài chính quan sát được, điểm 0. Mọi thành phần đều unknown thì tổng hợp null, không tự tạo %.

Ví dụ chỉ có chuyên cần và học phí: trọng số thực tế là 60% và 40%. UI ghi rõ thiếu nhóm học tập, hiển thị số nhóm đã có và trọng số áp dụng. Nếu AI tạm lỗi, bỏ AI khỏi tính toán và thông báo rõ trạng thái đó.

### Chuyên cần

- Vắng liên tiếp >=3 buổi, hoặc vắng >=4 buổi và chiếm >=40% buổi đã có điểm danh: 100 điểm.
- Vắng liên tiếp >=2 buổi, hoặc vắng >=3 buổi và chiếm >=20% buổi đã có điểm danh: 75 điểm.
- Chưa đạt hai mức trên: min(35, tỷ lệ vắng không phép ×100).
- Không vắng: 0. Thiếu điểm danh không tự biến thành absent; excused không phải vắng không phép. Khoảng trống điểm danh làm gián đoạn chuỗi vắng để tránh tự suy diễn.

### Học phí

Chỉ xét hóa đơn của lớp còn dư nợ thực, trạng thái pending/partial/overdue và hạn trước ngày hiện tại ở Việt Nam. Dư nợ = tổng tiền − giảm giá − các giao dịch đã thu. Không cảnh báo người đã trả đủ vì status cũ. Hóa đơn cancelled/refunded không dùng làm căn cứ quá hạn.

- Không quá hạn: 0 điểm.
- Quá hạn 1–7 ngày: 50 điểm.
- Quá hạn 8–14 ngày: 70 điểm.
- Quá hạn >14 ngày: 90 điểm.

### Giữ cảnh báo quan trọng và suy ra mức hỗ trợ

Điểm cuối = max(điểm nền, mức sàn), làm tròn một chữ số thập phân:

- AI vượt ngưỡng OOF hoặc chuyên cần đạt mức cần liên hệ: sàn 70.
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

- AIPrediction lưu snapshot, điểm AI gốc, kết quả hybrid, chính sách/version, mức hỗ trợ, dữ liệu thiếu và thời điểm.
- Student lấy snapshot hybrid mới nhất của mỗi lớp đang học, chọn lớp có điểm tổng hợp cao nhất. Điểm, mức và giải thích đến từ cùng snapshot, không trộn điểm lớp này với lý do lớp khác.
- Tận dụng cột Json `dropout_risk_reasons` để lưu envelope có version, reasons và details. Repository vẫn trả `dropout_risk_reasons` dạng mảng để tương thích UI; thêm `dropout_risk_type` và `dropout_risk_details` cho chi tiết. Không cần migration.
- Kết quả cũ được đánh dấu academic_legacy, UI yêu cầu đánh giá lại; không tự đổi nhãn phần trăm AI cũ thành điểm tổng hợp. Job định kỳ hoặc nút đánh giá sẽ tạo snapshot hybrid mới.
- Khi tổng hợp >=40, gửi **một cảnh báo tổng hợp cho mỗi người nhận**, gồm quản trị viên active và giáo viên phụ trách active trong đúng tenant, kèm link hồ sơ và căn cứ. Không tự gửi tin học viên.
- Chống lặp 24 giờ theo người nhận + học viên + lớp + mức hỗ trợ. Từ medium lên high được gửi ngay, không bị cảnh báo medium cũ chặn. Advisory lock theo tenant/sinh viên bảo vệ cập nhật đồng thời.
- Tự cập nhật sau điểm danh, cập nhật bài nộp, hoặc nút đánh giá. Job khởi động sau 30 giây, quét mỗi 60 phút; `DROPOUT_SCAN_INTERVAL_MINUTES` tối thiểu 5. Job giữ ngữ cảnh tenant cho mọi truy vấn nghiệp vụ.
- Trang lớp đọc snapshot đã lưu qua GET `/classes/:id/dropout-risk`. Không chấm lại hoặc tạo thông báo khi chỉ mở trang.
- Các màn hình dùng chung phần trăm, mức hỗ trợ, tối đa ba lý do ngắn, một câu hướng xử lý và thời điểm cập nhật. Không hiển thị bảng trọng số, điểm nền, log `[Tổng hợp]`/`[Dữ liệu]` hoặc nhiều hộp thông báo xanh. Thông tin thiếu được gói trong một dòng; lưu ý ý nghĩa phần trăm đặt trong tooltip. Chuông thông báo dùng polling 30 giây sẵn có.

## Kiểm tra và áp dụng

```powershell
# SmartEdu_CRM_BE: mock DB, không gửi thông báo thật
npm test -- tests/dropout-risk.test.ts tests/dropout-ui.test.ts tests/dropout-hybrid.test.ts
npm run typecheck
# AdminDashboard
npm run build
```

Khởi động lại backend, tải frontend mới, rồi bấm đánh giá lại. Thay đổi tổng hợp nằm ở backend/frontend nên không cần rebuild Docker AI hoặc train lại model.

Benchmark trong `dropout_public/report.md` chỉ đánh giá các model công khai, không phải chất lượng của điểm hybrid. Cần đối chiếu cảnh báo với kết quả thực tế để điều chỉnh chính sách và huấn luyện model tổng hợp từ dữ liệu trung tâm khi đủ lịch sử.
