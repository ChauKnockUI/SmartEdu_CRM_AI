"""
api_server.py
--------------
SmartEdu CRM — AI Prediction Microservice

Exposes three REST API endpoints powered by trained sklearn pipelines:

    GET  /                  — Health check, lists available endpoints
    POST /predict-lead      — Lead conversion probability v1 (Kaggle features)
    POST /predict-lead-v2   — Lead conversion probability v2 (CRM features)
    POST /predict-dropout   — Student dropout risk (early warning system)

Models are loaded once at server startup and reused across all requests.

Run:
    uvicorn api_server:app --reload
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Interactive docs:
    http://127.0.0.1:8000/docs      (Swagger UI)
    http://127.0.0.1:8000/redoc     (ReDoc)
"""

import os
import sys
import logging
import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Thêm logic Explainable AI
try:
    from ai_models.explain_lead_scoring import get_lead_explanation_json
    HAS_EXPLAINER = True
except ImportError:
    HAS_EXPLAINER = False


# Make ai_models importable so joblib can unpickle v2 model pipelines
# (they reference preprocessing_utils.log_transform_days)
_ai_models_dir = os.path.join(os.path.dirname(__file__), "ai_models")
if _ai_models_dir not in sys.path:
    sys.path.insert(0, _ai_models_dir)
try:
    from preprocessing_utils import log_transform_days  # noqa: F401
except ImportError:
    pass  # Model v2 won't load, but v1 and dropout will still work


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title       = "SmartEdu CRM — AI Service",
    description = (
        "Microservice providing AI-powered predictions for the SmartEdu CRM system.\n\n"
        "**Available models:**\n"
        "- **Lead Scoring** — predicts likelihood of a lead converting to a paying student\n"
        "- **Dropout Risk** — early warning system detecting students at risk of dropping out"
    ),
    version = "3.0.0",
)

# ---------------------------------------------------------------------------
# Global model registry
# Populated once at startup; all prediction handlers read from here.
# ---------------------------------------------------------------------------
_models: dict = {
    "lead_scoring"    : None,
    "lead_scoring_v2" : None,   # v2 model (CRM business features)
    "dropout_risk"    : None,
    "dropout_risk_version": "none",
}


# ---------------------------------------------------------------------------
# Startup: load all trained model pipelines
# ---------------------------------------------------------------------------
@app.on_event("startup")
def _load_models() -> None:
    """Load all .pkl model files into memory when the server starts."""
    base = os.path.dirname(__file__)

    # Lead Scoring model
    lead_path = os.path.join(base, "ai_models", "lead_scoring_pipeline.pkl")
    if os.path.exists(lead_path):
        _models["lead_scoring"] = joblib.load(lead_path)
        logger.info("Lead Scoring model loaded.")
    else:
        logger.warning(f"Lead Scoring model not found: {lead_path}")

    # Lead Scoring v2 model (CRM business-aligned features)
    # Tries Random Forest first (supported by SHAP without JSON parsing bug), falls back to XGBoost
    lead_v2_xgb_path = os.path.join(base, "ai_models", "lead_scoring_v2_xgb_pipeline.pkl")
    lead_v2_rf_path  = os.path.join(base, "ai_models", "lead_scoring_v2_rf_pipeline.pkl")
    if os.path.exists(lead_v2_rf_path):
        _models["lead_scoring_v2"] = joblib.load(lead_v2_rf_path)
        logger.info("Lead Scoring v2 (Random Forest) model loaded.")
    elif os.path.exists(lead_v2_xgb_path):
        _models["lead_scoring_v2"] = joblib.load(lead_v2_xgb_path)
        logger.info("Lead Scoring v2 (XGBoost) model loaded.")
    else:
        logger.warning("Lead Scoring v2 model not found.")

    # Dropout Risk model
    # Mô hình cảnh báo sớm học viên dựa trên chuyên cần và bài tập
    dropout_v2_path = os.path.join(base, "ai_models", "dropout_model_v2_pipeline.pkl")
    dropout_path    = os.path.join(base, "ai_models", "dropout_model_pipeline.pkl")
    if os.path.exists(dropout_v2_path):
        _models["dropout_risk"] = joblib.load(dropout_v2_path)
        _models["dropout_risk_version"] = "v2"
        logger.info("Dropout Risk v2 model loaded.")
    elif os.path.exists(dropout_path):
        _models["dropout_risk"] = joblib.load(dropout_path)
        _models["dropout_risk_version"] = "v1"
        logger.info("Dropout Risk legacy model loaded.")
    else:
        logger.warning(f"Dropout Risk model not found: {dropout_path}")


# ---------------------------------------------------------------------------
# Input schemas (Pydantic models for request body validation)
# ---------------------------------------------------------------------------

class LeadFeatures(BaseModel):
    """
    Input features for the Lead Scoring prediction.
    Collected from the CRM intake form when a prospective student enquires.
    """
    Lead_Origin    : str   = Field(..., example="Landing Page Submission")
    Lead_Source    : str   = Field(..., example="Google")
    Total_Time_Spent: float = Field(..., example=500, description="Total seconds spent on website")
    Occupation     : str   = Field(..., example="Student")
    Student_Year   : str   = Field(..., example="Year 3",
                                   description="'Year 1'–'Year 4' for students, 'Not Applicable' otherwise")
    Specialization : str   = Field(..., example="Finance Management")
    TotalVisits    : float = Field(..., example=3)
    Page_Views     : float = Field(..., example=2.5, description="Average page views per visit")

class LeadFeaturesV2(BaseModel):
    """
    Input features for Lead Scoring v2 prediction.

    Designed around CRM business logic — all features are either captured at
    lead intake or automatically computed by the CRM backend.

    Bối cảnh nghiệp vụ:
    - Lead_Source             : Kênh marketing dẫn khách đến
    - Occupation              : Phân khúc đối tượng (gộp cả năm học SV)
    - Study_Purpose           : Mục tiêu / động lực học tập
    - Course_Interested       : Sản phẩm (khóa học) quan tâm
    - Call_Attempt_Count      : Số lần Sales đã liên hệ (auto từ CRM)
    - Last_Engagement_Status  : Kết quả tương tác gần nhất (auto từ CRM)
    - Days_Since_Created      : Số ngày từ lúc tạo lead (auto tính)
    """
    Lead_Source            : str   = Field(..., example="google_ads",
                                           description="Marketing channel: facebook, google_ads, website_organic, referral, tiktok, other")
    Occupation             : str   = Field(..., example="student_y3_y4",
                                           description="Customer segment: student_y1_y2, student_y3_y4, working_professional, unemployed, parent_enrolling")
    Study_Purpose          : str   = Field(..., example="study_abroad",
                                           description="Learning goal: pass_exam, study_abroad, career_advancement, hobby, company_training")
    Course_Interested      : str   = Field(..., example="ielts",
                                           description="Target course: communication_english, ielts, toeic, frontend, backend_nodejs, data_analysis, kids_english")
    Call_Attempt_Count     : int   = Field(..., ge=0, example=2,
                                           description="Number of contact attempts by Sales (auto-tracked by CRM)")
    Last_Engagement_Status : str   = Field(..., example="positive_interaction",
                                           description="Latest interaction result: not_answering, busy_call_back, interested_need_time, positive_interaction, wrong_number, rejected")
    Days_Since_Created     : int   = Field(..., ge=0, example=3,
                                           description="Days since lead was created in CRM (auto-computed)")


class DropoutFeatures(BaseModel):
    """
    Input features for the Dropout Risk prediction.

    These values are computed by the CRM backend from the raw attendance
    and homework records stored per student, then passed to this endpoint.

    Bối cảnh nghiệp vụ:
    - Excused_Absences      : Số buổi vắng có báo trước/xin phép
    - Unexcused_Absences    : Số buổi vắng không báo (dấu hiệu rủi ro cao nhất)
    - Consecutive_Absences  : Số buổi vắng LIÊN TIẾP ở thời điểm hiện tại (còi báo đỏ)
    - Missed_Homework_Ratio : Tỷ lệ bài tập không nộp tính đến hiện tại (0.0 = không lỡ, 1.0 = lỡ hết)
    """
    Attendance_Rate: float = Field(..., ge=0.0, le=1.0, example=0.72)
    Unexcused_Absence_Count: int = Field(..., ge=0, example=3)
    Unexcused_Absence_Rate: float = Field(..., ge=0.0, le=1.0, example=0.18)
    Late_Count: int = Field(..., ge=0, example=2)
    Consecutive_Unexcused_Absences: int = Field(..., ge=0, example=2)
    Days_Since_Last_Attended: int = Field(..., ge=0, example=7)
    Assignment_Missing_Rate: float = Field(..., ge=0.0, le=1.0, example=0.45)
    Assignment_Late_Count: int = Field(..., ge=0, example=1)
    Average_Score: float = Field(..., ge=0.0, le=10.0, example=6.4)
    Score_Trend: float = Field(..., example=-1.2)
    Has_Overdue_Invoice: int = Field(..., ge=0, le=1, example=1)
    Days_Overdue: int = Field(..., ge=0, example=10)
    Class_Progress_Ratio: float = Field(..., ge=0.0, le=1.0, example=0.55)


def _dropout_reasons(features: DropoutFeatures) -> list[str]:
    reasons: list[str] = []

    if features.Consecutive_Unexcused_Absences >= 2:
        reasons.append(f"Vang khong phep {features.Consecutive_Unexcused_Absences} buoi lien tiep")
    if features.Unexcused_Absence_Rate >= 0.2 or features.Unexcused_Absence_Count >= 3:
        reasons.append(f"Vang khong phep {features.Unexcused_Absence_Count} buoi")
    if features.Assignment_Missing_Rate >= 0.35:
        reasons.append(f"Khong nop {round(features.Assignment_Missing_Rate * 100)}% bai tap")
    if features.Days_Since_Last_Attended >= 7:
        reasons.append(f"Da {features.Days_Since_Last_Attended} ngay chua tham gia buoi hoc")
    if features.Average_Score < 5.0:
        reasons.append(f"Diem trung binh thap ({features.Average_Score:.1f}/10)")
    if features.Score_Trend <= -1.0:
        reasons.append("Ket qua hoc tap dang giam")
    if features.Has_Overdue_Invoice:
        reasons.append(f"Co hoc phi qua han {features.Days_Overdue} ngay")

    if not reasons:
        reasons.append("Khong co dau hieu rui ro lon trong du lieu hien tai")

    return reasons[:4]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
def health_check():
    """
    Health check endpoint.
    Returns service status and the list of prediction endpoints.
    """
    return {
        "service" : "SmartEdu_CRM_AI_Service",
        "version" : "3.0.0",
        "status"  : "healthy",
        "models_loaded": {k: (v is not None) for k, v in _models.items()},
        "endpoints": [
            {"path": "/predict-lead",    "method": "POST", "description": "Lead scoring v1 (Kaggle features)"},
            {"path": "/predict-lead-v2", "method": "POST", "description": "Lead scoring v2 (CRM business features)"},
            {"path": "/predict-dropout", "method": "POST", "description": "Student dropout early warning"},
        ],
    }


@app.post("/predict-lead", tags=["Lead Scoring v1"])
def predict_lead(features: LeadFeatures):
    """
    Predict the probability that a prospective student (lead) will convert
    into a paying enrolled student.

    Returns a probability score and a sales action label:
    - **HOT_LEAD**  (>= 80%) — high priority, follow up immediately
    - **WARM_LEAD** (>= 50%) — moderate interest, nurture with content
    - **COLD_LEAD** (<  50%) — low intent, low-touch outreach only
    """
    if _models["lead_scoring"] is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Lead Scoring model is not loaded. Run train_lead_scoring.py first.",
        )

    try:
        input_df = pd.DataFrame([{
            "Lead Origin"                   : features.Lead_Origin,
            "Lead Source"                   : features.Lead_Source,
            "Total Time Spent on Website"   : features.Total_Time_Spent,
            "What is your current occupation": features.Occupation,
            "Specialization"                : features.Specialization,
            "TotalVisits"                   : features.TotalVisits,
            "Page Views Per Visit"          : features.Page_Views,
            "Student_Year"                  : features.Student_Year,
        }])

        prediction  = int(_models["lead_scoring"].predict(input_df)[0])
        probability = float(_models["lead_scoring"].predict_proba(input_df)[0][1])

        if probability >= 0.8:
            label = "HOT_LEAD"
        elif probability >= 0.5:
            label = "WARM_LEAD"
        else:
            label = "COLD_LEAD"

        return {
            "prediction"       : "Converted" if prediction == 1 else "Not Converted",
            "probability_score": round(probability * 100, 2),
            "recommendation"   : label,
        }

    except Exception as e:
        logger.error(f"Lead scoring prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict-lead-v2", tags=["Lead Scoring v2"])
def predict_lead_v2(features: LeadFeaturesV2):
    """
    **Lead Scoring v2** — Predict conversion probability using CRM business features.

    This endpoint uses the new 7-feature model designed around actual CRM workflow:
    - 4 features from lead intake form (Lead_Source, Occupation, Study_Purpose, Course)
    - 3 features auto-computed by CRM (Call_Attempt_Count, Last_Engagement, Days_Since_Created)

    Returns a probability score and a sales action label:
    - **HOT_LEAD**  (>= 80%) — high priority, follow up immediately
    - **WARM_LEAD** (>= 50%) — moderate interest, nurture with content
    - **COLD_LEAD** (<  50%) — low intent, low-touch outreach only
    """
    if _models["lead_scoring_v2"] is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Lead Scoring v2 model is not loaded. Run train_lead_scoring_v2.py first.",
        )

    try:
        input_df = pd.DataFrame([{
            "Lead_Source"            : features.Lead_Source,
            "Occupation"             : features.Occupation,
            "Study_Purpose"          : features.Study_Purpose,
            "Course_Interested"      : features.Course_Interested,
            "Call_Attempt_Count"     : features.Call_Attempt_Count,
            "Last_Engagement_Status" : features.Last_Engagement_Status,
            "Days_Since_Created"     : features.Days_Since_Created,
        }])

        prediction  = int(_models["lead_scoring_v2"].predict(input_df)[0])
        probability = float(_models["lead_scoring_v2"].predict_proba(input_df)[0][1])

        if probability >= 0.8:
            label  = "HOT_LEAD"
            action = "High priority — follow up within 1 hour. Prepare pricing & schedule."
        elif probability >= 0.5:
            label  = "WARM_LEAD"
            action = "Moderate interest — send course brochure, schedule a callback."
        else:
            label  = "COLD_LEAD"
            action = "Low intent — add to nurture email sequence, revisit in 7 days."

        return {
            "model_version"    : "v2",
            "prediction"       : "Converted" if prediction == 1 else "Not Converted",
            "probability_score": round(probability * 100, 2),
            "recommendation"   : label,
            "action"           : action,
        }

    except Exception as e:
        logger.error(f"Lead scoring v2 prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain-lead-v2", tags=["Lead Scoring v2"])
def explain_lead_v2(features: LeadFeaturesV2):
    """
    Predict AND EXPLAIN the probability that a lead will convert into a paying student.
    
    Returns the classification, probability score, and an NLP-friendly
    SHAP explanation breaking down EXACTLY why the AI made this decision.
    Specifically designed to be rendered directly onto the CRM interface for Sales Teams.
    """
    if _models["lead_scoring_v2"] is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Lead Scoring v2 model is not loaded. Run train_lead_scoring_v2.py first.",
        )
        
    if not HAS_EXPLAINER:
        raise HTTPException(
            status_code = 503,
            detail      = "Explainable AI module not loaded. Missing dependencies like 'shap'.",
        )

    try:
        # Convert schema to dict matching training data columns
        input_dict = features.dict()
        
        # Gọi hàm Explain từ ai_models
        response_json = get_lead_explanation_json(_models["lead_scoring_v2"], input_dict)
        return response_json

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Lead scoring v2 explanation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))



@app.post("/predict-dropout", tags=["Early Warning"])
def predict_dropout(features: DropoutFeatures):
    """
    Predict the probability that a currently enrolled student will drop out
    of their English language course before completion.

    Input values should be aggregated by the CRM backend from raw per-session
    attendance and homework records before calling this endpoint.

    Returns a risk level and a recommended CRM action:
    - **HIGH_RISK**   (>= 70%) — contact the student within 24 hours
    - **MEDIUM_RISK** (>= 40%) — send a check-in message, monitor closely
    - **LOW_RISK**    (<  40%) — student is on track, no action needed
    """
    if _models["dropout_risk"] is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Dropout Risk model is not loaded. Run train_dropout_model.py first.",
        )

    try:
        input_df = pd.DataFrame([{
            "Attendance_Rate": features.Attendance_Rate,
            "Unexcused_Absence_Count": features.Unexcused_Absence_Count,
            "Unexcused_Absence_Rate": features.Unexcused_Absence_Rate,
            "Late_Count": features.Late_Count,
            "Consecutive_Unexcused_Absences": features.Consecutive_Unexcused_Absences,
            "Days_Since_Last_Attended": features.Days_Since_Last_Attended,
            "Assignment_Missing_Rate": features.Assignment_Missing_Rate,
            "Assignment_Late_Count": features.Assignment_Late_Count,
            "Average_Score": features.Average_Score,
            "Score_Trend": features.Score_Trend,
            "Has_Overdue_Invoice": features.Has_Overdue_Invoice,
            "Days_Overdue": features.Days_Overdue,
            "Class_Progress_Ratio": features.Class_Progress_Ratio,
        }])

        probability = float(_models["dropout_risk"].predict_proba(input_df)[0][1])

        if probability >= 0.70:
            risk_level = "HIGH_RISK"
            action     = "Contact student within 24 hours — high dropout probability."
        elif probability >= 0.40:
            risk_level = "MEDIUM_RISK"
            action     = "Send a check-in message and monitor attendance next session."
        else:
            risk_level = "LOW_RISK"
            action     = "Student is on track. No immediate action required."

        return {
            "model_version"       : _models["dropout_risk_version"],
            "risk_level"         : risk_level,
            "dropout_probability": round(probability * 100, 2),
            "top_reasons"        : _dropout_reasons(features),
            "action"             : action,
        }

    except Exception as e:
        logger.error(f"Dropout prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
