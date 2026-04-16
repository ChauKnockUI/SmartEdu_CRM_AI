import os
import joblib
import pandas as pd
import numpy as np
import shap
import sys
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


# -------------------------------------------------------------------------
# 1. Mapping Features to Vietnamese for the Sales Team
# -------------------------------------------------------------------------
FEATURE_TRANSLATION = {
    "Lead_Source": "Nguồn khách hàng",
    "Occupation": "Nghề nghiệp / Phân khúc",
    "Study_Purpose": "Mục tiêu học tập",
    "Course_Interested": "Khóa học quan tâm",
    "Call_Attempt_Count": "Số lần gọi chưa nghe máy",
    "Days_Since_Created": "Số ngày từ lúc tạo lead",
    "Last_Engagement_Status": "Phản hồi gần nhất của khách"
}

VALUE_TRANSLATION = {
    "positive_interaction": "Tương tác rất ấn tượng/Tích cực",
    "busy_call_back": "Báo bận, gọi lại sau",
    "not_answering": "Chưa nhấc máy",
    "interested_need_time": "Quan tâm nhưng cần thời gian suy nghĩ",
    "rejected": "Từ chối khéo / Không có nhu cầu",
    "wrong_number": "Sai số / Lead rác",
    "student_y3_y4": "Sinh viên năm 3/4",
    "working_professional": "Người đi làm",
    "study_abroad": "Đi du học",
    "referral": "Khách được giới thiệu (Mối quan hệ)",
    "facebook": "Chạy Ads Facebook",
}

def translate_feature_name(raw_feat):
    """Convert raw OneHotEncoded feature names into readable text."""
    if raw_feat.startswith("cat__"):
        parts = raw_feat.split("_")
        # Extract value (handles multiple underscores in value like 'study_abroad')
        # Structure is cat__FeatureName_FeatureValue
        if "Engagement_Status" in raw_feat:
            val = raw_feat.split("Engagement_Status_")[1]
            readable_val = VALUE_TRANSLATION.get(val, val)
            return f"Phản hồi gần nhất: {readable_val}"
        elif "Occupation" in raw_feat:
            val = raw_feat.split("Occupation_")[1]
            readable_val = VALUE_TRANSLATION.get(val, val)
            return f"Phân khúc: {readable_val}"
        elif "Study_Purpose" in raw_feat:
            val = raw_feat.split("Study_Purpose_")[1]
            readable_val = VALUE_TRANSLATION.get(val, val)
            return f"Mục tiêu: {readable_val}"
        elif "Lead_Source" in raw_feat:
            val = raw_feat.split("Lead_Source_")[1]
            readable_val = VALUE_TRANSLATION.get(val, val)
            return f"Nguồn kênh: {readable_val}"
        elif "Course" in raw_feat:
            val = raw_feat.split("Course_Interested_")[1]
            readable_val = VALUE_TRANSLATION.get(val, val)
            return f"Khóa học: {readable_val}"
        return raw_feat
    elif raw_feat.startswith("num__"):
        base_name = raw_feat[5:]
        return FEATURE_TRANSLATION.get(base_name, base_name)
    return raw_feat

import re
def _safe_float(v):
    """Safely convert SHAP values (which might be arrays, strings, or weird scalars) to float."""
    try:
        if isinstance(v, (list, np.ndarray, pd.Series)):
            # Flatten recursively in case of nested arrays to get the first real value
            v = np.array(v).ravel()
            if len(v) > 0:
                v = v[0]
            else:
                return 0.0
        
        v_str = str(v)
        v_str = re.sub(r'[^\d\.\-eE]', '', v_str)
        if not v_str or v_str == '-' or v_str == '.':
            return 0.0
        return float(v_str)
    except Exception:
        return 0.0

# -------------------------------------------------------------------------
# 2. Extract SHAP Logic for a Given Lead
# -------------------------------------------------------------------------
def explain_lead_for_sales(pipeline, lead_df: pd.DataFrame, index: int = 0):
    """
    Generate a human-readable explanation using SHAP values.
    """
    # 1. Lấy dữ liệu và dự đoán pipeline
    lead_row = lead_df.iloc[[index]]
    probability = pipeline.predict_proba(lead_row)[0][1]
    
    # 2. Rã đông pipeline (Tách Preprocessor và Classifier)
    preprocessor = pipeline.named_steps["preprocessor"]
    calibrated_clf = pipeline.named_steps["classifier"]
    
    # Kéo mô hình Random Forest gốc ra khỏi lớp bọc Calibration
    base_model = calibrated_clf.calibrated_classifiers_[0].estimator
    
    X_transformed = preprocessor.transform(lead_row)
    
    # Extract feature names manually since FunctionTransformer lacks get_feature_names_out in older sklearn
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = trans.named_steps["onehot"]
            feats = ohe.get_feature_names_out(cols)
            feature_names.extend([f"cat__{f}" for f in feats])
        elif name == "num":
            feature_names.extend([f"num__{f}" for f in cols])
    
    # 3. Chạy SHAP TreeExplainer với Fallback An Toàn (Graceful Degradation)
    impacts = []
    try:
        explainer = shap.TreeExplainer(base_model)
        shap_vals_raw = explainer.shap_values(X_transformed)
        if isinstance(shap_vals_raw, list):
            shap_values = shap_vals_raw[1][0]
        else:
            shap_values = shap_vals_raw[0]
            
        shap_values = np.array(shap_values).flatten()
        X_array = X_transformed.toarray()[0] if hasattr(X_transformed, "toarray") else np.array(X_transformed).flatten()

        for i, f_name in enumerate(feature_names):
            if i < len(shap_values):
                val = _safe_float(shap_values[i])
            else:
                val = 0.0
                
            # Filter out inactive One-Hot Encoded features (value is 0) to avoid confusing Sales team
            if f_name.startswith("cat__") and i < len(X_array):
                if X_array[i] == 0:
                    continue
                    
            if abs(val) > 0.01: 
                impacts.append({
                    "feature": f_name,
                    "shap_val": val,
                    "magnitude": abs(val)
                })
                
        impacts = sorted(impacts, key=lambda x: x["magnitude"], reverse=True)
    except Exception as e:
        print(f"[Cảnh Báo] SHAP Explanation thất bại do xung đột Version. Lỗi: {e}")
        # Explain fail nhưng vẫn tiếp tục luồng code cho Predict
    
    # 5. Dịch ra văn bản cho Sale
    if probability >= 0.3:
        status_color = "🔴 HOT LEAD"
    elif probability >= 0.15:
        status_color = "🟡 WARM LEAD"
    else:
        status_color = "🔵 COLD LEAD"
        
    print(f"\n=======================================================")
    print(f"📊 GIẢI THÍCH TỪ AI (Dành cho Sale)")
    print(f"=======================================================")
    print(f"Khách hàng: Lead #{index+1}")
    print(f"Dự đoán   : {probability:.1%} chốt")
    print(f"Xếp loại  : {status_color}")
    print(f"---")
    
    if probability >= 0.3:
        print("✅ Lý do AI đánh giá khách tiềm năng (Kéo xác suất LÊN):")
        for imp in impacts:
            if imp["shap_val"] > 0:
                readable = translate_feature_name(imp["feature"])
                print(f"  + Điểm cộng lớn: Ở đặc điểm '{readable}'")
                
        print("\n⚠️ Cảnh báo (Điểm trừ):")
        for imp in impacts:
            if imp["shap_val"] < 0:
                readable = translate_feature_name(imp["feature"])
                print(f"  - Điểm trừ duy nhất: '{readable}' (hơi rủi ro lúc chốt)")
                
    else:
        print("❌ Lý do AI đánh giá khách lạnh (Kéo xác suất XUỐNG):")
        for imp in impacts:
            if imp["shap_val"] < 0:
                readable = translate_feature_name(imp["feature"])
                print(f"  - Cờ đỏ (Red Flag): Khách vướng ở '{readable}'")
                
        print("\n✨ Điểm vớt vát (Nếu cố gắng đeo bám):")
        for imp in impacts:
            if imp["shap_val"] > 0:
                readable = translate_feature_name(imp["feature"])
                print(f"  + Bù lại: Đặc điểm '{readable}' của khách vẫn gỡ gạc được.")
                
    print(f"=======================================================\n")


def get_lead_explanation_json(pipeline, lead_dict: dict) -> dict:
    """
    Generate JSON-serializable explanation using SHAP values for a single lead dict.
    Returns: dict suitable for API response.
    """
    
    lead_df = pd.DataFrame([lead_dict])
    probability = _safe_float(pipeline.predict_proba(lead_df)[0][1])
    
    preprocessor = pipeline.named_steps["preprocessor"]
    calibrated_clf = pipeline.named_steps["classifier"]
    base_model = calibrated_clf.calibrated_classifiers_[0].estimator
    
    X_transformed = preprocessor.transform(lead_df)
    
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = trans.named_steps["onehot"]
            feats = ohe.get_feature_names_out(cols)
            feature_names.extend([f"cat__{f}" for f in feats])
        elif name == "num":
            feature_names.extend([f"num__{f}" for f in cols])
            
    impacts = []
    # Khối giải thích SHAP cực kỳ cần tính an toàn tuyệt đối
    # Nếu Explain lỗi thì Predict vẫn phải trả về được cho Hệ thống
    try:
        explainer = shap.TreeExplainer(base_model)
        shap_vals_raw = explainer.shap_values(X_transformed)
        if isinstance(shap_vals_raw, list):
            shap_values = shap_vals_raw[1][0]
        else:
            shap_values = shap_vals_raw[0]
        
        shap_values = np.array(shap_values).flatten()
        X_array = X_transformed.toarray()[0] if hasattr(X_transformed, "toarray") else np.array(X_transformed).flatten()
            
        for i, f_name in enumerate(feature_names):
            if i < len(shap_values):
                val = _safe_float(shap_values[i])
            else:
                val = 0.0
                
            # Filter out inactive One-Hot Encoded features
            if f_name.startswith("cat__") and i < len(X_array):
                if X_array[i] == 0:
                    continue
                    
            if abs(val) > 0.01: 
                impacts.append({
                    "feature": f_name,
                    "shap_val": val,
                    "magnitude": abs(val)
                })
                
        impacts = sorted(impacts, key=lambda x: x["magnitude"], reverse=True)
    except Exception as e:
        import traceback
        import logging
        logging.error(f"[SHAP Fallback] Không thể tính toán SHAP Explanation: {e}")
        traceback.print_exc()
        # impacts rỗng -> frontend k hiển thị chart nhưng điểm xác suất vẫn OK
    
    if probability >= 0.3:
        status = "HOT_LEAD"
    elif probability >= 0.15:
        status = "WARM_LEAD"
    else:
        status = "COLD_LEAD"
        
    positive_factors = []
    negative_factors = []
        
    # Nếu tính toán thành công impacts
    if len(impacts) > 0:
        for imp in impacts:
            readable = translate_feature_name(imp["feature"])
            if imp["shap_val"] > 0:
                positive_factors.append(readable)
            else:
                negative_factors.append(readable)
    else:
        # SHAP Fail fallback logic -> Hiển thị cảnh báo hoặc để mảng rỗng
        positive_factors.append("⚠️ Chưa thể truy xuất lý do điểm cộng (Lỗi Module SHAP Server)")
        negative_factors.append("⚠️ Vui lòng Fix version XGBoost / Sklearn để giải thích được hoạt động.")
            
    return {
        "prediction": "Converted" if status == "HOT_LEAD" else "Not Converted",
        "probability_score": round(probability * 100, 2),
        "recommendation": status,
        "explanation": {
            "positive_factors": positive_factors,
            "negative_factors": negative_factors
        }
    }

# -------------------------------------------------------------------------
# 3. Main Test
# -------------------------------------------------------------------------
if __name__ == "__main__":
    rf_path = os.path.join(MODEL_DIR, "lead_scoring_v2_rf_pipeline.pkl")
    if not os.path.exists(rf_path):
        print("Không tìm thấy model. Hãy chạy train_lead_scoring_v2.py trước.")
        exit(1)
        
    pipeline = joblib.load(rf_path)
    
    # Tạo 2 Lead ảo cho CRM
    demo_leads = pd.DataFrame([
        {
            # HOT LEAD: Sinh viên năm 3, IELTS, Tương tác tốt, Được giới thiệu
            "Lead_Source": "referral",
            "Occupation": "student_y3_y4",
            "Study_Purpose": "study_abroad",
            "Course_Interested": "ielts",
            "Last_Engagement_Status": "positive_interaction",
            "Call_Attempt_Count": 1,
            "Days_Since_Created": 1.0,
        },
        {
            # COLD LEAD: Rác từ TikTok, không nghe máy 5 lần, để 30 ngày
            "Lead_Source": "tiktok",
            "Occupation": "unemployed",
            "Study_Purpose": "hobby",
            "Course_Interested": "communication_english",
            "Last_Engagement_Status": "not_answering",
            "Call_Attempt_Count": 5,
            "Days_Since_Created": 30.0,
        }
    ])
    
    # Explain Lead 1 (HOT)
    explain_lead_for_sales(pipeline, demo_leads, index=0)
    
    # Explain Lead 2 (COLD)
    explain_lead_for_sales(pipeline, demo_leads, index=1)
