"""UCI academic/finance inference with an explicit academic-only fallback."""
import math

import joblib
import numpy as np
import pandas as pd

ACADEMIC_FEATURES = ["Average_Score", "Pass_Rate", "Not_Evaluated_Rate"]
FEATURES = ACADEMIC_FEATURES + ["Has_Debt", "Tuition_Up_To_Date"]
VARIANTS = {"uci_academic": ACADEMIC_FEATURES, "uci_academic_finance": FEATURES}
MODEL_VERSIONS = {"uci_academic": "uci_academic_public_1",
                  "uci_academic_finance": "uci_academic_finance_public_1"}


def load_dropout_bundle(path, variant="uci_academic_finance"):
    bundle = joblib.load(path)
    if variant not in VARIANTS or bundle.get("variant") != variant or bundle.get("features") != VARIANTS[variant]:
        raise ValueError(f"Incompatible dropout bundle; expected {variant}")
    if list(bundle["pipeline"].feature_names_in_) != VARIANTS[variant]:
        raise ValueError("Fitted model feature contract mismatch")
    threshold = float(bundle["threshold"])
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid dropout threshold")
    return bundle


def score_dropout(bundle, values):
    variant = bundle["variant"]
    features = VARIANTS[variant]
    if bundle.get("features") != features or set(values) != set(features):
        raise ValueError(f"Expected exactly these features: {features}")
    limits = {"Average_Score": (0, 10), "Pass_Rate": (0, 1), "Not_Evaluated_Rate": (0, 1),
              "Has_Debt": (0, 1), "Tuition_Up_To_Date": (0, 1)}
    for key, value in values.items():
        if value is None and key == "Average_Score":
            continue
        lo, hi = limits[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"Invalid feature: {key}")
        if key in FEATURES[3:] and value not in (0, 1):
            raise ValueError(f"Expected binary flag: {key}")
    if values['Pass_Rate'] + values['Not_Evaluated_Rate'] > 1 + 1e-9:
        raise ValueError("Passed and unevaluated assessments cannot exceed the total")
    row = {key: np.nan if value is None else value for key, value in values.items()}
    probability = float(bundle["pipeline"].predict_proba(pd.DataFrame([row], columns=features))[0, 1])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid model output")
    flagged = probability >= bundle["threshold"]
    return {
        "model_version": MODEL_VERSIONS[variant],
        "includes_finance": variant == "uci_academic_finance",
        "model_flagged": flagged,
        "risk_level": "HIGH_RISK" if flagged else "LOW_RISK",
        "risk_score": round(probability * 100, 2),
        "threshold": float(bundle["threshold"]),
        "score_type": "source_model_score",
        "top_reasons": ["Model học tập và học phí vượt ngưỡng theo dõi" if variant == "uci_academic_finance"
                        else "Model học tập vượt ngưỡng theo dõi"] if flagged else [],
        "action": "Liên hệ học viên để tìm hiểu khó khăn và tư vấn." if flagged else "Tiếp tục theo dõi học viên.",
    }
