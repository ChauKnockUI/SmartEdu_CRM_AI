"""Engineer public and CRM records into the same broad feature vocabulary."""
import math
import pandas as pd

MODEL_VERSION = "lead_staged_rf_3"
CRM_MODEL_VERSION = "lead_crm_snapshot_rf_1"
CRM_CATEGORICAL = ["Lead_Source", "Occupation", "Study_Purpose", "Course_Interested", "Last_Engagement_Status"]
CRM_NUMERIC = ["Call_Attempt_Count", "Days_Since_Created", "Answered_Call_Count",
               "Activity_Count_7d", "Days_Since_Last_Activity"]
CRM_FEATURES = CRM_CATEGORICAL + CRM_NUMERIC
CORE_FEATURES = ["Source_Group", "Occupation_Group", "Goal_Group"]
FEATURES = CORE_FEATURES + ["Engagement_Group"]
HOT_THRESHOLD = 0.8
CONVERSION_THRESHOLD = 0.5
FEATURE_DESCRIPTIONS = {
    "Lead_Source": "Nguồn khách hàng", "Occupation": "Nghề nghiệp", "Study_Purpose": "Mục tiêu học",
    "Course_Interested": "Nhóm khóa học", "Last_Engagement_Status": "Phản hồi gần nhất",
    "Call_Attempt_Count": "Số lần gọi", "Days_Since_Created": "Tuổi lead",
    "Answered_Call_Count": "Số cuộc gọi đã kết nối", "Activity_Count_7d": "Tương tác trong 7 ngày",
    "Days_Since_Last_Activity": "Số ngày chưa liên hệ",
    "Source_Group": "Nhóm nguồn khách hàng",
    "Occupation_Group": "Nhóm nghề nghiệp",
    "Goal_Group": "Nhóm mục tiêu học tập",
    "Engagement_Group": "Mức độ tương tác",
}


def _text(value):
    if pd.isna(value):
        return "unknown"
    value = str(value).strip().lower()
    return "unknown" if value in {"", "select", "nan", "none", "unknown"} else value


def source_group(value):
    value = _text(value)
    if value in {"google", "google_ads", "google form"}:
        return "search"
    if value in {"facebook", "tiktok", "social media"}:
        return "social"
    if value in {"reference", "referral", "welingak website"}:
        return "referral"
    if value in {"organic search", "direct traffic", "website_organic", "website",
                 "organic", "olark chat", "website_chat"}:
        return "website"
    return "unknown" if value == "unknown" else "other"


def occupation_group(value):
    value = _text(value)
    if value in {"student", "student_y1_y2", "student_y3_y4"}:
        return "student"
    if value in {"working professional", "working_professional", "businessman"}:
        return "working"
    if value == "unemployed":
        return "unemployed"
    return "unknown" if value == "unknown" else "other"


def goal_group(value):
    value = _text(value)
    if value in {"better career prospects", "career_advancement", "career", "company_training"}:
        return "career"
    if value in {"study_abroad", "pass_exam", "exam"}:
        return "academic"
    if value in {"hobby", "personal"}:
        return "personal"
    # Convenience, vague 'improve', specialization and course name do not
    # establish a learning goal. Do not fill these using the target or RNG.
    return "unknown"


def engagement_group(value):
    value = _text(value)
    if value in {"positive_interaction", "interested_need_time", "busy_call_back",
                 "high", "medium", "low", "active"}:
        return "active"
    if value in {"not_answering", "wrong_number", "rejected", "unreachable"}:
        return "unreachable"
    return "unknown"


def source_engagement(row):
    # Tags may encode the sales outcome. Only activity evidence is considered.
    activities = [_text(row.get(name)) for name in ["Last Activity", "Last Notable Activity"]]
    if any(value in {"email bounced", "unreachable", "unsubscribed", "email marked spam"}
           for value in activities):
        return "unreachable"
    if any(value in {"email opened", "email received", "email link clicked",
                     "had a phone conversation", "olark chat conversation"}
           for value in activities):
        return "active"
    # Sending an SMS and 'Converted to Lead' are not customer responses.
    return "unknown"


def engineer_source_features(raw):
    """Public data adapter. Deliberately never reads Converted, Tags or scores."""
    return pd.DataFrame({
        "Source_Group": raw["Lead Source"].map(source_group),
        "Occupation_Group": raw["What is your current occupation"].map(occupation_group),
        "Goal_Group": raw["What matters most to you in choosing a course"].map(goal_group),
        "Engagement_Group": raw.apply(source_engagement, axis=1),
    }, index=raw.index)[FEATURES]


def engineer_lead_features(frame):
    """Use the same grouping vocabulary for engineered training rows and CRM."""
    if set(FEATURES).issubset(frame.columns):
        return frame[FEATURES].copy()
    return pd.DataFrame({
        "Source_Group": frame["Lead_Source"].map(source_group),
        "Occupation_Group": frame["Occupation"].map(occupation_group),
        "Goal_Group": frame["Study_Purpose"].map(goal_group),
        "Engagement_Group": frame["Last_Engagement_Status"].map(engagement_group),
    }, index=frame.index)[FEATURES]


def prediction_fields(probability):
    probability = float(probability)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid lead probability")
    recommendation = ("HOT_LEAD" if probability >= HOT_THRESHOLD else
                      "WARM_LEAD" if probability >= CONVERSION_THRESHOLD else "COLD_LEAD")
    return {
        "probability": probability,
        "conversion_probability": probability,
        "conversion_band": "high" if probability >= HOT_THRESHOLD else "medium" if probability >= .5 else "low",
        "prediction": "Converted" if probability >= CONVERSION_THRESHOLD else "Not Converted",
        "probability_score": round(probability * 100, 2),
        "recommendation": recommendation,
    }


def get_lead_prediction_json(pipeline, lead_dict):
    if hasattr(pipeline, "select_pipeline"):
        selected, stage, basis = pipeline.select_pipeline(lead_dict)
        return {**get_lead_prediction_json(selected, lead_dict),
                "scoring_stage": stage, "assessment_basis": basis}
    frame = pd.DataFrame([lead_dict])
    metadata = getattr(pipeline, "lead_metadata_", {})
    active = metadata.get("active_features", FEATURES)
    engineer = getattr(pipeline, "named_steps", {}).get("engineer")
    engineered = engineer.transform(frame) if engineer is not None else engineer_lead_features(frame)
    supported = metadata.get("observed_categories", {})
    unsupported = [name for name in active if name in supported
                   if supported and engineered.iloc[0][name] not in supported.get(name, [])]
    fields = prediction_fields(pipeline.predict_proba(frame)[0][1])
    actions = {
        "HOT_LEAD": "Follow up within 1 hour; prepare course pricing and schedule.",
        "WARM_LEAD": "Send course information and schedule a callback.",
        "COLD_LEAD": "Continue nurturing and review when new information is available.",
    }
    return {
        **fields, "model_version": metadata.get("model_version", MODEL_VERSION),
        "api_contract_version": "lead_scoring_3",
        "scoring_stage": "post_contact" if metadata.get("model_version") == CRM_MODEL_VERSION else metadata.get("training_stage", "initial"),
        "assessment_basis": "crm_history" if metadata.get("model_version") == CRM_MODEL_VERSION else "profile_only" if active == CORE_FEATURES else "profile_and_response",
        "score_type": "crm_conversion_probability" if metadata.get("model_version") == CRM_MODEL_VERSION else "public_dataset_model_score",
        "prediction_horizon_days": 30 if metadata.get("model_version") == CRM_MODEL_VERSION else None,
        "active_features": active,
        "engineered_features": engineered.astype(object).where(pd.notna(engineered), None).iloc[0].to_dict(),
        "unsupported_features": unsupported, "action": actions[fields["recommendation"]],
        "thresholds": {"hot": HOT_THRESHOLD, "warm": CONVERSION_THRESHOLD,
                       "conversion": CONVERSION_THRESHOLD},
        "limitations": [
            "Care priority is provided separately by CRM workflow rules.",
        ] if metadata.get("model_version") == CRM_MODEL_VERSION else [
            "Grouped public-data features approximate CRM observations.",
            "The score has not been calibrated on SmartEdu CRM outcomes.",
        ],
    }


def engineer_crm_features(frame):
    """Actual scoring-time CRM values; no public-data proxy or outcome fields."""
    result = pd.DataFrame(index=frame.index)
    aliases = {
        "Lead_Source": {"google": "google_ads", "google form": "google_ads", "website": "website_organic"},
        "Study_Purpose": {"career": "career_advancement", "exam": "pass_exam"},
        "Last_Engagement_Status": {"high": "positive_interaction", "medium": "interested_need_time", "low": "busy_call_back"},
    }
    for feature in CRM_CATEGORICAL:
        result[feature] = frame[feature].map(_text).replace(aliases.get(feature, {}))
    def course_family(value):
        for text, family in [("ielts", "ielts"), ("toeic", "toeic"), ("kid", "kids_english"),
                             ("giao tiếp", "communication_english"), ("communication", "communication_english"),
                             ("data", "data_analysis"), ("frontend", "frontend"), ("backend", "backend_nodejs")]:
            if text in value:
                return family
        return "unknown"
    result["Course_Interested"] = result["Course_Interested"].map(course_family)
    for feature in CRM_NUMERIC:
        result[feature] = pd.to_numeric(frame[feature], errors="raise")
    return result[CRM_FEATURES]
