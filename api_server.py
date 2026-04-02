"""
api_server.py
--------------
SmartEdu CRM — AI Prediction Microservice

Exposes three REST API endpoints powered by trained sklearn pipelines:

    GET  /                  — Health check, lists available endpoints
    POST /predict-lead      — Lead conversion probability (sales use case)
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
import logging
import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

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
    version = "2.0.0",
)

# ---------------------------------------------------------------------------
# Global model registry
# Populated once at startup; all prediction handlers read from here.
# ---------------------------------------------------------------------------
_models: dict = {
    "lead_scoring" : None,
    "dropout_risk" : None,
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

    # Dropout Risk model
    # Mô hình cảnh báo sớm học viên dựa trên chuyên cần và bài tập
    dropout_path = os.path.join(base, "ai_models", "dropout_model_pipeline.pkl")
    if os.path.exists(dropout_path):
        _models["dropout_risk"] = joblib.load(dropout_path)
        logger.info("Dropout Risk model loaded.")
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
    Excused_Absences    : int   = Field(..., ge=0, example=1,
                                        description="Sessions missed with prior notice")
    Unexcused_Absences  : int   = Field(..., ge=0, example=2,
                                        description="Sessions missed without any notice")
    Consecutive_Absences: int   = Field(..., ge=0, example=2,
                                        description="Current streak of consecutive missed sessions")
    Missed_Homework_Ratio: float = Field(..., ge=0.0, le=1.0, example=0.30,
                                         description="Ratio of homework assignments not submitted (0.0–1.0)")


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
        "version" : "2.0.0",
        "status"  : "healthy",
        "models_loaded": {k: (v is not None) for k, v in _models.items()},
        "endpoints": [
            {"path": "/predict-lead",    "method": "POST", "description": "Lead conversion scoring"},
            {"path": "/predict-dropout", "method": "POST", "description": "Student dropout early warning"},
        ],
    }


@app.post("/predict-lead", tags=["Lead Scoring"])
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
            "Excused_Absences"    : features.Excused_Absences,
            "Unexcused_Absences"  : features.Unexcused_Absences,
            "Consecutive_Absences": features.Consecutive_Absences,
            "Missed_Homework_Ratio": features.Missed_Homework_Ratio,
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
            "risk_level"         : risk_level,
            "dropout_probability": round(probability * 100, 2),
            "action"             : action,
        }

    except Exception as e:
        logger.error(f"Dropout prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
