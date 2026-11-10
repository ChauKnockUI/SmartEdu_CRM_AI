"""FastAPI inference for lead scoring and dropout risk.

Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
Lead CRM endpoints: /predict-lead-v2 and /explain-lead-v2.
"""

import os
import sys
import logging
import joblib
import pandas as pd
from dropout_inference import ACADEMIC_FEATURES, MODEL_VERSIONS, load_dropout_bundle, score_dropout
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Resolve serialized feature functions when loading model artifacts.
_ai_models_dir = os.path.join(os.path.dirname(__file__), "ai_models")
if _ai_models_dir not in sys.path:
    sys.path.insert(0, _ai_models_dir)
from lead_contract import MODEL_VERSION, CRM_MODEL_VERSION, get_lead_prediction_json
from ai_models.explain_lead_scoring import get_lead_explanation_json

# Logging
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# App instance
app = FastAPI(
    title       = "SmartEdu CRM — AI Service",
    description = (
        "Microservice providing AI-powered predictions for the SmartEdu CRM system.\n\n"
        "**Available models:**\n"
        "- **Lead Scoring** — estimates the likelihood of a lead becoming a student\n"
        "- **Dropout Risk** — early warning system detecting students at risk of dropping out"
    ),
    version = "3.0.0",
)

# Global model registry
# Populated once at startup; all prediction handlers read from here.
_models: dict = {
    "lead_scoring"    : None,
    "lead_scoring_v2" : None,
    "dropout_risk"    : None,
    "dropout_academic_fallback": None,
    "dropout_risk_version": "none",
}

# Startup: load all trained model pipelines
@app.on_event("startup")
def _load_models() -> None:
    """Load serving artifacts and reject incompatible or unapproved lead models."""
    base = os.path.dirname(__file__)

    # Lead Scoring model
    lead_path = os.path.join(base, "ai_models", "lead_scoring_pipeline.pkl")
    if os.path.exists(lead_path):
        _models["lead_scoring"] = joblib.load(lead_path)
        logger.info("Lead Scoring model loaded.")
    else:
        logger.warning(f"Lead Scoring model not found: {lead_path}")

    # Do not silently substitute an old synthetic artifact or a different feature contract.
    _models["lead_scoring_v2"] = None
    for filename, version in [("lead_scoring_v2_rf_pipeline.pkl", MODEL_VERSION),
                              ("lead_scoring_crm_pipeline.pkl", CRM_MODEL_VERSION)]:
        path = os.path.join(base, "ai_models", filename)
        if not os.path.exists(path):
            continue
        try:
            model = joblib.load(path)
            metadata = getattr(model, "lead_metadata_", {})
            if metadata.get("model_version") != version:
                raise ValueError("Lead model contract is outdated; run train_real_lead_scoring_rf.py")
            if version == CRM_MODEL_VERSION and (metadata.get("outcome_window_days") != 30 or not metadata.get("temporal_validated") or not metadata.get("approved_for_serving") or metadata.get("training_stage") != "post_contact"):
                raise ValueError("CRM artifact must be validated for post-contact scoring with a 30-day horizon")
            if version == MODEL_VERSION:
                if not hasattr(model, "select_pipeline"):
                    raise ValueError("Lead model must support initial and post-contact scoring")
                _models["lead_scoring_v2"] = model
            elif _models["lead_scoring_v2"] is not None:
                # The CRM history model is used after actual contact only.
                _models["lead_scoring_v2"].crm_model_ = model
            logger.info("Lead model %s loaded.", version)
        except Exception:
            logger.exception("Cannot load lead artifact %s", filename)
    if _models["lead_scoring_v2"] is None:
        logger.warning("Lead Scoring v2 model not found.")

    # Finance is the primary model; use the academic model only for unknown tuition.
    for key, variant in [("dropout_risk", "uci_academic_finance"),
                         ("dropout_academic_fallback", "uci_academic")]:
        dropout_path = os.path.join(base, "ai_models", "dropout_public", f"{variant}_bundle.pkl")
        try:
            _models[key] = load_dropout_bundle(dropout_path, variant)
            logger.info("Dropout %s loaded.", variant)
        except Exception:
            _models[key] = None
            logger.exception("Cannot load dropout %s bundle", variant)
    _models["dropout_risk_version"] = MODEL_VERSIONS["uci_academic_finance"] if _models["dropout_risk"] else "none"

# Input schemas (Pydantic models for request body validation)

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

    Compatible CRM input. The model groups source, occupation, goal, engagement
    and interaction count; course and lead age are accepted but not used.

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
                                           description="Compatibility field; not used by the aligned model")
    Call_Attempt_Count     : int   = Field(..., ge=0, example=2,
                                           description="Number of contact attempts by Sales (auto-tracked by CRM)")
    Last_Engagement_Status : str   = Field(..., example="positive_interaction",
                                           description="Latest interaction result: not_answering, busy_call_back, interested_need_time, positive_interaction, wrong_number, rejected")
    Days_Since_Created     : int   = Field(..., ge=0, example=3,
                                           description="Compatibility field; not used by the aligned model")
    Answered_Call_Count: Optional[int] = Field(None, ge=0)
    Has_Contact_Activity: Optional[bool] = Field(None, description="Actual call/email/SMS/meeting exists; notes are excluded")
    Activity_Count_7d: Optional[int] = Field(None, ge=0)
    Days_Since_Last_Activity: Optional[int] = Field(None, ge=0)

class DropoutFeatures(BaseModel):
    """CRM snapshot; unknown tuition selects the academic-only fallback."""
    Average_Score: Optional[float] = Field(..., ge=0, le=10)
    Pass_Rate: float = Field(..., ge=0, le=1)
    Not_Evaluated_Rate: float = Field(..., ge=0, le=1)
    Has_Debt: Optional[Literal[0, 1]] = None
    Tuition_Up_To_Date: Optional[Literal[0, 1]] = None

    class Config:
        extra = "forbid"

# Endpoints

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
        "models_loaded": {k: (v is not None) for k, v in _models.items() if k != "dropout_risk_version"},
        "lead_model_version": getattr(_models["lead_scoring_v2"], "lead_metadata_", {}).get("model_version"),
        "endpoints": [
            {"path": "/predict-lead",    "method": "POST", "description": "Lead scoring v1 (Kaggle features)"},
            {"path": "/predict-lead-v2", "method": "POST", "description": "Lead scoring v2 (CRM business features)"},
            {"path": "/explain-lead-v2", "method": "POST", "description": "Same score with optional forest explanation"},
            {"path": "/predict-dropout", "method": "POST", "description": "Student dropout early warning"},
        ],
    }

@app.post("/predict-lead", deprecated=True, tags=["Lead Scoring v1"])
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
            detail      = "Legacy lead model is not loaded. Use /predict-lead-v2 or /explain-lead-v2.",
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

    The compatible CRM request is transformed into broad source, occupation,
    goal, engagement and interaction-volume groups. Active features are returned
    with the score. Course choice and lead age are not used by this model.

    Returns a probability score and a sales action label:
    - **HOT_LEAD**  (>= 80%) — high priority, follow up immediately
    - **WARM_LEAD** (>= 50%) — moderate interest, nurture with content
    - **COLD_LEAD** (<  50%) — low intent, low-touch outreach only
    """
    if _models["lead_scoring_v2"] is None:
        raise HTTPException(
            status_code = 503,
            detail      = "Lead Scoring model is not loaded. Run train_real_lead_scoring_rf.py first.",
        )

    try:
        return get_lead_prediction_json(_models["lead_scoring_v2"], features.model_dump())

    except Exception as e:
        logger.exception("Lead scoring prediction failed")
        raise HTTPException(status_code=503, detail="Lead scoring is temporarily unavailable") from e

@app.post("/explain-lead-v2", tags=["Lead Scoring v2"])
def explain_lead_v2(features: LeadFeaturesV2):
    """
    Predict AND EXPLAIN the probability that a lead will convert into a paying student.
    
    Returns the same prediction as /predict-lead-v2, plus optional SHAP factors.
    Factors describe the forest ensemble before calibration. Missing or failed
    SHAP does not disable predictions.
    """
    prediction = predict_lead_v2(features)
    return get_lead_explanation_json(
        _models["lead_scoring_v2"], features.model_dump(), prediction=prediction,
    )

@app.post("/predict-dropout", tags=["Early Warning"])
def predict_dropout(features: DropoutFeatures):
    """Use five UCI features when tuition is observed, otherwise three academic features."""
    finance_known = features.Has_Debt is not None and features.Tuition_Up_To_Date is not None
    bundle = _models["dropout_risk" if finance_known else "dropout_academic_fallback"]
    if bundle is None:
        raise HTTPException(status_code=503, detail="Run ai_models/train_dropout_public.py and restart the AI service.")
    try:
        values = features.model_dump()
        if not finance_known:
            values = {key: values[key] for key in ACADEMIC_FEATURES}
        return score_dropout(bundle, values)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
