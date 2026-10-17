# Slide Notes - Dropout Early Warning AI

## 1. Van de
- Trung tam can phat hien som hoc vien co nguy co nghi hoc.
- Neu chi doi den khi hoc vien nghi han moi xu ly thi da muon.
- Muc tieu cua AI la uu tien danh sach hoc vien can CSKH/giao vien can thiep som.

## 2. Giai phap da xay
- Xay module Dropout Early Warning trong SmartEdu CRM.
- Backend tong hop tin hieu tu attendance, assignment, invoice, test result.
- AI service tinh dropout probability, risk level, top reasons va recommended action.
- Frontend hien thi risk tren danh sach hoc vien, chi tiet hoc vien va trang AI Predictions.

## 3. Feature Engineering
- Attendance: ty le di hoc, vang khong phep, vang lien tiep, so ngay chua di hoc.
- Assignment: ty le khong nop bai, so bai nop tre.
- Learning: diem trung binh, xu huong diem.
- Finance/context: hoc phi qua han, so ngay qua han, tien do lop.

## 4. Model
- Model v2 dung RandomForestClassifier trong sklearn pipeline.
- Pipeline gom imputer, scaler va classifier.
- Dataset v2 la synthetic business dataset theo persona, dung de bootstrap demo.
- Model file deploy: `dropout_model_v2_pipeline.pkl`.

## 5. Ket qua danh gia
- ROC-AUC: 0.8981.
- Recall tai nguong hanh dong 40%: 0.8860.
- False negative rate: 11.4%.
- Behavior tests: 100%.
- Diem quan trong: uu tien giam bo sot hoc vien rui ro thay vi chi toi uu accuracy.

## 6. Luong nghiep vu
- Giao vien diem danh hoac cap nhat bai tap.
- Backend aggregate feature theo student_id + class_id.
- Backend goi AI `/predict-dropout`.
- Backend luu `dropout_risk`, `dropout_risk_level`, `dropout_risk_reasons`.
- Backend ghi log `AIPrediction` de audit va retrain ve sau.

## 7. API minh hoa
Endpoint:

```http
POST /predict-dropout
```

Input la 13 feature da aggregate.

Output:

```json
{
  "risk_level": "HIGH_RISK",
  "dropout_probability": 82.4,
  "top_reasons": ["Vang khong phep 2 buoi lien tiep"],
  "action": "Contact student within 24 hours"
}
```

## 8. Gioi han hien tai
- Dataset training hien tai la synthetic, chua phai du lieu lich su that.
- Model nen duoc xem la he thong canh bao ho tro, khong phai ket luan tu dong.
- Can thu thap du lieu that sau khi van hanh de retrain va calibration.

## 9. Cau ket thuc
- Diem manh cua module la khong chi co model AI rieng le, ma co ca business workflow day du:
  du lieu -> feature -> AI -> luu risk -> hien thi ly do -> hanh dong CSKH.
