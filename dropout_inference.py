"""Inference contract for the single academic public-data baseline."""
import math

import joblib
import numpy as np
import pandas as pd

FEATURES = ["Average_Score", "Pass_Rate", "Not_Evaluated_Rate"]


def load_dropout_bundle(path):
    bundle = joblib.load(path)
    if bundle.get("variant") != "uci_academic" or bundle.get("features") != FEATURES:
        raise ValueError("Incompatible dropout bundle; expected uci_academic")
    if list(bundle["pipeline"].feature_names_in_) != FEATURES:
        raise ValueError("Fitted model feature contract mismatch")
    threshold = float(bundle["threshold"])
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid dropout threshold")
    return bundle


def score_dropout(bundle, values):
    if set(values) != set(FEATURES):
        raise ValueError("Expected exactly three academic features")
    if values['Pass_Rate'] + values['Not_Evaluated_Rate'] > 1 + 1e-9:
        raise ValueError("Passed and unevaluated assessments cannot exceed the total")
    row = {key: np.nan if value is None else value for key, value in values.items()}
    probability = float(bundle["pipeline"].predict_proba(pd.DataFrame([row], columns=FEATURES))[0, 1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid model output")
    flagged = probability >= bundle["threshold"]
    return {
        "model_version": "uci_academic_public_1",
        "model_flagged": flagged,
        "risk_level": "HIGH_RISK" if flagged else "LOW_RISK",
        "risk_score": round(probability * 100, 2),
        "threshold": float(bundle["threshold"]),
        "score_type": "source_model_score",
        "top_reasons": ["Model học tập vượt ngưỡng theo dõi"] if flagged else [],
        "action": "Liên hệ học viên để tìm hiểu khó khăn và tư vấn." if flagged else "Tiếp tục theo dõi học viên.",
    }
