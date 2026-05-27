"""
QA tests for dropout risk model v2.
"""

import os
import sys
import joblib
import pandas as pd

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ai_models", "dropout_model_v2_pipeline.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "archive", "dropout_dataset_v2.csv")
RANDOM_SEED = 42

FEATURE_COLS = [
    "Attendance_Rate",
    "Unexcused_Absence_Count",
    "Unexcused_Absence_Rate",
    "Late_Count",
    "Consecutive_Unexcused_Absences",
    "Days_Since_Last_Attended",
    "Assignment_Missing_Rate",
    "Assignment_Late_Count",
    "Average_Score",
    "Score_Trend",
    "Has_Overdue_Invoice",
    "Days_Overdue",
    "Class_Progress_Ratio",
]

THRESHOLDS = {
    "roc_auc": 0.85,
    "recall": 0.80,
    "fn_rate": 0.20,
    "behavior_pass_rate": 0.95,
}


def predict_prob(model, data):
    return float(model.predict_proba(pd.DataFrame([data]))[0][1])


def record(results, name, passed, note=""):
    results.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {note}")


def main() -> None:
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Run train_dropout_model_v2.py first.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATASET_PATH)
    _, X_test, _, y_test = train_test_split(
        df[FEATURE_COLS], df["Is_Dropout"], test_size=0.2, random_state=RANDOM_SEED, stratify=df["Is_Dropout"]
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.40).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fn_rate = fn / (fn + tp)
    roc = roc_auc_score(y_test, y_prob)
    rec = recall_score(y_test, y_pred)

    print(f"ROC-AUC={roc:.4f} precision={precision_score(y_test, y_pred):.4f} recall={rec:.4f} f1={f1_score(y_test, y_pred):.4f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn} FN_rate={fn_rate:.1%}")

    results = []
    record(results, "ROC-AUC threshold", roc >= THRESHOLDS["roc_auc"])
    record(results, "Recall threshold at 40% action threshold", rec >= THRESHOLDS["recall"])
    record(results, "FN rate threshold", fn_rate <= THRESHOLDS["fn_rate"])

    base = {
        "Attendance_Rate": 0.95,
        "Unexcused_Absence_Count": 0,
        "Unexcused_Absence_Rate": 0.0,
        "Late_Count": 0,
        "Consecutive_Unexcused_Absences": 0,
        "Days_Since_Last_Attended": 1,
        "Assignment_Missing_Rate": 0.05,
        "Assignment_Late_Count": 0,
        "Average_Score": 8.2,
        "Score_Trend": 0.1,
        "Has_Overdue_Invoice": 0,
        "Days_Overdue": 0,
        "Class_Progress_Ratio": 0.5,
    }

    risky = base | {
        "Attendance_Rate": 0.45,
        "Unexcused_Absence_Count": 5,
        "Unexcused_Absence_Rate": 0.45,
        "Consecutive_Unexcused_Absences": 3,
        "Days_Since_Last_Attended": 14,
        "Assignment_Missing_Rate": 0.75,
        "Average_Score": 4.2,
        "Score_Trend": -1.8,
    }

    p_base = predict_prob(model, base)
    p_risky = predict_prob(model, risky)
    record(results, "strong student is low risk", p_base < 0.30, f"p={p_base:.3f}")
    record(results, "disengaged student is high risk", p_risky > 0.70, f"p={p_risky:.3f}")

    for feature, low, high in [
        ("Unexcused_Absence_Count", 0, 5),
        ("Consecutive_Unexcused_Absences", 0, 3),
        ("Assignment_Missing_Rate", 0.05, 0.8),
        ("Days_Since_Last_Attended", 1, 14),
    ]:
        a = base.copy()
        b = base.copy()
        a[feature] = low
        b[feature] = high
        if feature == "Unexcused_Absence_Count":
            b["Unexcused_Absence_Rate"] = 0.35
        record(results, f"{feature} increases risk", predict_prob(model, a) < predict_prob(model, b))

    pass_rate = sum(results) / len(results)
    print(f"Behavior/test pass rate: {pass_rate:.1%}")
    if pass_rate < THRESHOLDS["behavior_pass_rate"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
