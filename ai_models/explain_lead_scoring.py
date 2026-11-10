"""Optional SHAP explanations; prediction never depends on SHAP availability.

Tree SHAP explains the mean uncalibrated forest output across all calibration
folds. It is not an exact decomposition of the final calibrated score.
"""
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_models_dir = str(Path(__file__).resolve().parent)
if _models_dir not in sys.path:
    sys.path.insert(0, _models_dir)
from lead_contract import FEATURE_DESCRIPTIONS, get_lead_prediction_json

logger = logging.getLogger(__name__)
VALUE_LABELS = {
    "google_ads": "Google", "website_organic": "Website", "facebook": "Facebook", "tiktok": "TikTok",
    "student_y1_y2": "Sinh viên năm 1–2", "student_y3_y4": "Sinh viên năm 3–4",
    "working_professional": "Người đi làm", "parent_enrolling": "Phụ huynh",
    "career_advancement": "Phát triển nghề nghiệp", "pass_exam": "Thi chứng chỉ", "study_abroad": "Du học",
    "positive_interaction": "Tương tác tích cực", "interested_need_time": "Cần thêm thời gian",
    "busy_call_back": "Hẹn gọi lại", "not_answering": "Chưa nghe máy", "wrong_number": "Sai số",
    "rejected": "Từ chối", "ielts": "IELTS", "toeic": "TOEIC",
    "communication_english": "Tiếng Anh giao tiếp", "kids_english": "Tiếng Anh thiếu nhi",
    "search": "Tìm kiếm", "social": "Mạng xã hội", "referral": "Giới thiệu",
    "website": "Website/chat", "other": "Nhóm khác", "unknown": "Chưa rõ",
    "student": "Sinh viên", "working": "Người đi làm", "unemployed": "Chưa có việc làm",
    "career": "Phát triển nghề nghiệp", "academic": "Học tập/chứng chỉ",
    "personal": "Mục tiêu cá nhân", "active": "Đã có tương tác",
    "unreachable": "Khó tiếp cận hoặc từ chối", "none": "Chưa có lượt tương tác",
    "low": "Ít", "medium": "Vừa", "high": "Nhiều",
}


def positive_class_shap(values, n_features):
    """Support old list[class] and modern (sample, feature, class) outputs."""
    if isinstance(values, list):
        result = np.asarray(values[1])[0]
    else:
        array = np.asarray(values)
        if array.shape != (1, n_features, 2):
            raise ValueError(f"Unsupported binary SHAP shape: {array.shape}")
        result = array[0, :, 1]
    if result.shape != (n_features,) or not np.isfinite(result).all():
        raise ValueError("Invalid SHAP contribution vector")
    return result


def forest_ensemble_impacts(pipeline, frame):
    shap = importlib.import_module("shap")
    engineered = pipeline.named_steps["engineer"].transform(frame)
    folds = pipeline.named_steps["classifier"].calibrated_classifiers_
    active = pipeline.lead_metadata_["active_features"]
    grouped = {feature: 0.0 for feature in active}
    baseline, raw_probability = 0.0, 0.0
    for fold in folds:
        estimator = fold.estimator
        preprocessor = estimator.named_steps["preprocessor"]
        forest = estimator.named_steps["forest"]
        transformed = preprocessor.transform(engineered)
        names = preprocessor.get_feature_names_out()
        explainer = shap.TreeExplainer(forest)
        values = positive_class_shap(explainer.shap_values(transformed), len(names))
        for name, value in zip(names, values):
            raw_name = name.split("__", 1)[-1]
            feature = next(f for f in active if raw_name == f or raw_name.startswith(f + "_"))
            # Include both present and absent one-hot categories: omitting
            # absent categories discards part of the actual explanation.
            grouped[feature] += float(value) / len(folds)
        baseline += float(np.asarray(explainer.expected_value)[1]) / len(folds)
        raw_probability += float(forest.predict_proba(transformed)[0, 1]) / len(folds)
    if not np.isclose(baseline + sum(grouped.values()), raw_probability, atol=1e-5):
        raise ValueError("SHAP additivity check failed")
    return grouped, baseline, raw_probability


def get_lead_explanation_json(pipeline, lead_dict, prediction=None):
    result = dict(prediction if prediction is not None else get_lead_prediction_json(pipeline, lead_dict))
    if hasattr(pipeline, "select_pipeline"):
        pipeline, _, _ = pipeline.select_pipeline(lead_dict)
    result["explanation"] = {"positive_factors": [], "negative_factors": []}
    result["explanation_status"] = "unavailable"
    result["explanation_scope"] = "uncalibrated_forest_ensemble"
    try:
        grouped, baseline, raw_probability = forest_ensemble_impacts(pipeline, pd.DataFrame([lead_dict]))
        result["explanation_status"] = "available"
        result["explanation_details"] = {
            "base_probability": baseline, "uncalibrated_probability": raw_probability,
            "contributions": grouped,
            "note": "Contributions explain the forests before probability calibration, not the final score.",
        }
        for feature, contribution in sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True):
            if abs(contribution) <= 0.01:
                continue
            value = result["engineered_features"][feature]
            if value is None:
                value = "unknown"
            text = f"{FEATURE_DESCRIPTIONS[feature]}: {VALUE_LABELS.get(value, value)}"
            key = "positive_factors" if contribution > 0 else "negative_factors"
            result["explanation"][key].append(text)
    except ImportError:
        logger.info("Lead SHAP explanation unavailable; prediction remains available")
    except Exception:
        logger.exception("Lead SHAP explanation failed; returning prediction without explanation")
    return result


def explain_lead_for_sales(pipeline, lead_df, index=0):
    return get_lead_explanation_json(pipeline, lead_df.iloc[index].to_dict())


if __name__ == "__main__":
    import json
    import joblib
    pipeline = joblib.load(Path(__file__).with_name("lead_scoring_v2_rf_pipeline.pkl"))
    example = {"Lead_Source": "google_ads", "Occupation": "student_y3_y4",
               "Study_Purpose": "study_abroad", "Last_Engagement_Status": "positive_interaction",
               "Call_Attempt_Count": 2, "Days_Since_Created": 3, "Course_Interested": "ielts"}
    print(json.dumps(get_lead_explanation_json(pipeline, example), ensure_ascii=True, indent=2))
