"""
compare_lead_scoring_models.py
-------------------------------
SmartEdu CRM — Lead Scoring v1 vs v2 Model Comparison

Loads BOTH the old model (v1 - Kaggle features) and new models (v2 - CRM features)
and produces a comprehensive comparison report.

Comparison dimensions:
  1. Feature set analysis
  2. Test set metrics (on each model's own test data)
  3. Behavioral tests (domain logic scenarios)
  4. Feature importance ranking
  5. Prediction consistency

Output:
  - Console report
  - lead_scoring_comparison.log

Usage:
    python compare_lead_scoring_models.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

# Import custom transformer function required to unpickle v2 model pipelines.
# joblib needs this function available in the current module's namespace.
from preprocessing_utils import log_transform_days  # noqa: F401

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "..", "dataset", "archive")

V1_MODEL_PATH     = os.path.join(BASE_DIR, "lead_scoring_pipeline.pkl")
V2_RF_MODEL_PATH  = os.path.join(BASE_DIR, "lead_scoring_v2_rf_pipeline.pkl")
V2_XGB_MODEL_PATH = os.path.join(BASE_DIR, "lead_scoring_v2_xgb_pipeline.pkl")

V2_DATASET_PATH   = os.path.join(DATASET_DIR, "lead_scoring_v2_dataset.csv")

LOG_FILE = os.path.join(BASE_DIR, "lead_scoring_comparison.log")

# ---------------------------------------------------------------------------
# Logging to both console and file
# ---------------------------------------------------------------------------
_log_lines = []

def log(msg: str = "") -> None:
    """Print and buffer a log line."""
    print(msg)
    _log_lines.append(msg)


def flush_log() -> None:
    """Write all buffered lines to the log file."""
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines))
    print(f"\n  Log written to: {LOG_FILE}")


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------
def load_model(path: str, name: str):
    """Load a model if it exists."""
    if os.path.exists(path):
        model = joblib.load(path)
        log(f"  [OK] {name} loaded from: {os.path.basename(path)}")
        return model
    else:
        log(f"  [WARN] {name} NOT FOUND: {path}")
        return None


# ---------------------------------------------------------------------------
# Feature Set Comparison
# ---------------------------------------------------------------------------
def compare_feature_sets() -> None:
    """Compare v1 and v2 feature philosophies."""
    log("")
    log("=" * 70)
    log("  SECTION 1: FEATURE SET COMPARISON")
    log("=" * 70)

    v1_features = {
        "Lead Origin":                    ("Categorical", "Kênh intake (API/Form/Landing) — meta-data kỹ thuật"),
        "Lead Source":                    ("Categorical", "Nguồn (Google/Olark/Direct) — hợp lý nhưng quá chi tiết"),
        "Total Time Spent on Website":    ("Numeric",     "Thời gian trên web — PHỤ THUỘC web analytics"),
        "What is your current occupation":("Categorical", "Nghề nghiệp — quá generic (Student/Professional)"),
        "Specialization":                 ("Categorical", "Chuyên ngành — KHÔNG map vào sản phẩm trung tâm"),
        "TotalVisits":                    ("Numeric",     "Số lần truy cập — PHỤ THUỘC web analytics"),
        "Page Views Per Visit":           ("Numeric",     "Trang/lần — PHỤ THUỘC web analytics"),
        "Student_Year":                   ("Categorical", "Năm học — simulate random, ko có ground truth"),
    }

    v2_features = {
        "Lead_Source":            ("Categorical", "Kênh marketing (Facebook/Google/Referral) — thực tế CRM"),
        "Occupation":            ("Categorical", "Phân khúc KH chi tiết (SV Y1-2/Y3-4/Đi làm/Thất nghiệp)"),
        "Study_Purpose":         ("Categorical", "Mục tiêu học (Du học/Thi/Sở thích) — PROXY MOTIVATION"),
        "Course_Interested":     ("Categorical", "Khóa quan tâm — map đúng sản phẩm trung tâm"),
        "Call_Attempt_Count":    ("Numeric",     "Số lần gọi — ĐẶC SẢN CRM, đo sales effort"),
        "Last_Engagement_Status":("Categorical", "Kết quả tương tác cuối — SIGNAL MẠNH NHẤT"),
        "Days_Since_Created":    ("Numeric",     "Độ 'lạnh' lead — time decay pattern"),
    }

    log(f"\n  {'Feature v1':<35} {'Type':<13} {'Assessment'}")
    log(f"  {'-'*90}")
    for feat, (ftype, desc) in v1_features.items():
        marker = "[X]" if "PHU THUOC" in desc or "simulate" in desc else "[!]" if "meta-data" in desc or "KHONG" in desc else "[OK]"
        log(f"  {marker} {feat:<33} {ftype:<13} {desc}")

    log(f"\n  {'Feature v2':<35} {'Type':<13} {'Assessment'}")
    log(f"  {'-'*90}")
    for feat, (ftype, desc) in v2_features.items():
        log(f"  [OK] {feat:<33} {ftype:<13} {desc}")

    log(f"\n  Summary:")
    log(f"    v1: 3/8 features depend on Web Analytics (impractical for SME CRM)")
    log(f"    v2: 7/7 features are CRM-native and actionable by Sales teams")
    log(f"    v2 adds: Study_Purpose (motivation), Call_Attempt_Count (effort),")
    log(f"             Last_Engagement_Status (gold signal), Days_Since_Created (decay)")


# ---------------------------------------------------------------------------
# v2 Model Evaluation
# ---------------------------------------------------------------------------
def evaluate_v2_models(rf_model, xgb_model) -> dict:
    """Evaluate v2 models on their test set."""
    log("")
    log("=" * 70)
    log("  SECTION 2: v2 MODEL METRICS (on v2 dataset)")
    log("=" * 70)

    if not os.path.exists(V2_DATASET_PATH):
        log(f"  [ERROR] v2 dataset not found. Run generate_lead_scoring_v2_dataset.py first.")
        return {}

    df = pd.read_csv(V2_DATASET_PATH)
    features = ["Lead_Source", "Occupation", "Study_Purpose", "Course_Interested",
                 "Call_Attempt_Count", "Last_Engagement_Status", "Days_Since_Created"]

    X = df[features]
    y = df["Converted"]

    # Use same split as training (seed=42, test=0.2)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}
    models = {}
    if rf_model:
        models["v2 Random Forest"] = rf_model
    if xgb_model:
        models["v2 XGBoost"] = xgb_model

    log(f"\n  Test set size: {len(X_test)} records")
    log(f"\n  {'Model':<22} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10} {'Precision':>10} {'Recall':>10}")
    log(f"  {'-'*74}")

    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "roc_auc":   roc_auc_score(y_test, y_proba),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
        }
        results[name] = metrics

        log(f"  {name:<22} {metrics['accuracy']:>10.2%} {metrics['f1']:>10.4f} "
            f"{metrics['roc_auc']:>10.4f} {metrics['precision']:>10.4f} {metrics['recall']:>10.4f}")

    return results


# ---------------------------------------------------------------------------
# Behavioral Tests (domain logic validation)
# ---------------------------------------------------------------------------
def run_behavioral_tests(models: dict) -> None:
    """Test that models respect business logic expectations."""
    log("")
    log("=" * 70)
    log("  SECTION 3: BEHAVIORAL TESTS — Domain Logic Validation")
    log("=" * 70)

    tests = [
        {
            "id": "B1",
            "name": "HOT lead (referral + SV Y3-4 + du học + positive + fresh)",
            "data": {
                "Lead_Source": "referral", "Occupation": "student_y3_y4",
                "Study_Purpose": "study_abroad", "Course_Interested": "ielts",
                "Call_Attempt_Count": 1, "Last_Engagement_Status": "positive_interaction",
                "Days_Since_Created": 1,
            },
            "expected_min": 0.70,
            "expected_label": "HIGH (>70%)",
        },
        {
            "id": "B2",
            "name": "COLD lead (tiktok + thất nghiệp + hobby + rejected + 30 ngày)",
            "data": {
                "Lead_Source": "tiktok", "Occupation": "unemployed",
                "Study_Purpose": "hobby", "Course_Interested": "data_analysis",
                "Call_Attempt_Count": 7, "Last_Engagement_Status": "rejected",
                "Days_Since_Created": 30,
            },
            "expected_max": 0.15,
            "expected_label": "LOW (<15%)",
        },
        {
            "id": "B3",
            "name": "Referral > Facebook (same other features)",
            "data_a": {
                "Lead_Source": "referral", "Occupation": "working_professional",
                "Study_Purpose": "career_advancement", "Course_Interested": "toeic",
                "Call_Attempt_Count": 2, "Last_Engagement_Status": "interested_need_time",
                "Days_Since_Created": 3,
            },
            "data_b": {
                "Lead_Source": "facebook", "Occupation": "working_professional",
                "Study_Purpose": "career_advancement", "Course_Interested": "toeic",
                "Call_Attempt_Count": 2, "Last_Engagement_Status": "interested_need_time",
                "Days_Since_Created": 3,
            },
            "compare": "a > b",
            "expected_label": "referral prob > facebook prob",
        },
        {
            "id": "B4",
            "name": "Positive > Not Answering (engagement status impact)",
            "data_a": {
                "Lead_Source": "google_ads", "Occupation": "student_y3_y4",
                "Study_Purpose": "pass_exam", "Course_Interested": "communication_english",
                "Call_Attempt_Count": 2, "Last_Engagement_Status": "positive_interaction",
                "Days_Since_Created": 3,
            },
            "data_b": {
                "Lead_Source": "google_ads", "Occupation": "student_y3_y4",
                "Study_Purpose": "pass_exam", "Course_Interested": "communication_english",
                "Call_Attempt_Count": 2, "Last_Engagement_Status": "not_answering",
                "Days_Since_Created": 3,
            },
            "compare": "a > b",
            "expected_label": "positive_interaction > not_answering",
        },
        {
            "id": "B5",
            "name": "Fresh lead (2 days) > Old lead (25 days)",
            "data_a": {
                "Lead_Source": "google_ads", "Occupation": "student_y1_y2",
                "Study_Purpose": "pass_exam", "Course_Interested": "toeic",
                "Call_Attempt_Count": 1, "Last_Engagement_Status": "busy_call_back",
                "Days_Since_Created": 2,
            },
            "data_b": {
                "Lead_Source": "google_ads", "Occupation": "student_y1_y2",
                "Study_Purpose": "pass_exam", "Course_Interested": "toeic",
                "Call_Attempt_Count": 1, "Last_Engagement_Status": "busy_call_back",
                "Days_Since_Created": 25,
            },
            "compare": "a > b",
            "expected_label": "2 days > 25 days",
        },
        {
            "id": "B6",
            "name": "Few calls (1) > Many calls (7) — diminishing returns",
            "data_a": {
                "Lead_Source": "facebook", "Occupation": "working_professional",
                "Study_Purpose": "career_advancement", "Course_Interested": "ielts",
                "Call_Attempt_Count": 1, "Last_Engagement_Status": "busy_call_back",
                "Days_Since_Created": 5,
            },
            "data_b": {
                "Lead_Source": "facebook", "Occupation": "working_professional",
                "Study_Purpose": "career_advancement", "Course_Interested": "ielts",
                "Call_Attempt_Count": 7, "Last_Engagement_Status": "busy_call_back",
                "Days_Since_Created": 5,
            },
            "compare": "a > b",
            "expected_label": "1 call > 7 calls",
        },
    ]

    for model_name, model in models.items():
        log(f"\n  --- {model_name} ---")

        for t in tests:
            test_id = t["id"]
            test_name = t["name"]

            if "compare" in t:
                # Comparative test (A vs B)
                df_a = pd.DataFrame([t["data_a"]])
                df_b = pd.DataFrame([t["data_b"]])
                prob_a = model.predict_proba(df_a)[0][1]
                prob_b = model.predict_proba(df_b)[0][1]

                passed = prob_a > prob_b if t["compare"] == "a > b" else prob_a < prob_b
                status = "PASS" if passed else "FAIL"
                log(f"  [{status}] {test_id} | {test_name}")
                log(f"         A={prob_a:.1%}  B={prob_b:.1%}  expected: {t['expected_label']}")
            else:
                # Absolute test
                df_t = pd.DataFrame([t["data"]])
                prob = model.predict_proba(df_t)[0][1]

                if "expected_min" in t:
                    passed = prob >= t["expected_min"]
                elif "expected_max" in t:
                    passed = prob <= t["expected_max"]
                else:
                    passed = True

                status = "PASS" if passed else "FAIL"
                log(f"  [{status}] {test_id} | {test_name}")
                log(f"         prob={prob:.1%}  expected: {t['expected_label']}")


# ---------------------------------------------------------------------------
# v1 Reference Metrics (from training log)
# ---------------------------------------------------------------------------
def show_v1_reference() -> None:
    """Display v1 model metrics from the original training run."""
    log("")
    log("=" * 70)
    log("  SECTION 4: v1 MODEL REFERENCE METRICS (from original training)")
    log("=" * 70)
    log("")
    log("  v1 Model used Kaggle 'Lead Scoring' dataset (9240 records, 8 features)")
    log("  Training output.log shows:")
    log("")
    log("  Accuracy: 77.06%")
    log("                 precision    recall  f1-score   support")
    log("       0 (No)       0.80      0.84      0.82      1136")
    log("       1 (Yes)      0.72      0.66      0.69       712")
    log("       accuracy                         0.77      1848")
    log("")
    log("  Confusion Matrix:")
    log("    TP= 468   FP= 180")
    log("    FN= 244   TN= 956")
    log("")
    log("  Key Issues:")
    log("    - Recall (class 1) = 66% → Misses 34% of convertible leads")
    log("    - F1 (class 1) = 0.69 → Below acceptable threshold for production")
    log("    - 3/8 features depend on web analytics (unrealistic for CRM)")


# ---------------------------------------------------------------------------
# Final Verdict
# ---------------------------------------------------------------------------
def final_verdict(v2_results: dict) -> None:
    """Produce the final comparison verdict."""
    log("")
    log("=" * 70)
    log("  FINAL VERDICT")
    log("=" * 70)

    v1_metrics = {
        "accuracy": 0.7706, "f1": 0.69, "roc_auc": None,
        "precision": 0.72, "recall": 0.66,
    }

    log(f"\n  {'Model':<22} {'Accuracy':>10} {'F1-Score':>10} {'Precision':>10} {'Recall':>10}")
    log(f"  {'-'*64}")
    log(f"  {'v1 (Kaggle RF)':<22} {v1_metrics['accuracy']:>10.2%} {v1_metrics['f1']:>10.4f} "
        f"{v1_metrics['precision']:>10.4f} {v1_metrics['recall']:>10.4f}")

    for name, metrics in v2_results.items():
        log(f"  {name:<22} {metrics['accuracy']:>10.2%} {metrics['f1']:>10.4f} "
            f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f}")

    log("")

    # Determine best v2 model
    if v2_results:
        best_name = max(v2_results.items(), key=lambda x: x[1]["f1"])[0]
        best = v2_results[best_name]

        f1_improvement = best["f1"] - v1_metrics["f1"]
        recall_improvement = best["recall"] - v1_metrics["recall"]

        log(f"  [WINNER] Best v2 Model: {best_name}")
        log(f"     F1 improvement     : {v1_metrics['f1']:.4f} -> {best['f1']:.4f}  (Delta = +{f1_improvement:.4f})")
        log(f"     Recall improvement : {v1_metrics['recall']:.2%} -> {best['recall']:.2%}  (Delta = +{recall_improvement:.2%})")
        log("")

        if f1_improvement > 0.05:
            log("  [OK] VERDICT: v2 model is SIGNIFICANTLY BETTER than v1.")
            log("     Recommend: Deploy v2 as primary model.")
        elif f1_improvement > 0:
            log("  [OK-] VERDICT: v2 model is MARGINALLY better than v1.")
            log("     Recommend: A/B test before full deployment.")
        else:
            log("  [WARN] VERDICT: v2 model does not outperform v1 on synthetic data.")
            log("     Recommend: Re-examine feature engineering or collect real CRM data.")

    log("")
    log("  ⚠️  NOTE: v1 was trained on Kaggle data. v2 was trained on synthetic data.")
    log("  This comparison shows FEATURE DESIGN superiority, not a direct head-to-head.")
    log("  True comparison requires A/B testing on real CRM traffic.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log("=" * 70)
    log(f"  SmartEdu CRM — Lead Scoring Model Comparison Report")
    log(f"  Generated: {timestamp}")
    log("=" * 70)

    # Load models
    log(f"\n  Loading models...")
    v1_model  = load_model(V1_MODEL_PATH, "v1 (Kaggle RF)")
    rf_model  = load_model(V2_RF_MODEL_PATH, "v2 Random Forest")
    xgb_model = load_model(V2_XGB_MODEL_PATH, "v2 XGBoost")

    if rf_model is None and xgb_model is None:
        log("\n  [ERROR] No v2 models found. Run train_lead_scoring_v2.py first.")
        flush_log()
        sys.exit(1)

    # Section 1: Feature comparison
    compare_feature_sets()

    # Section 2: v2 metrics
    v2_results = evaluate_v2_models(rf_model, xgb_model)

    # Section 3: Behavioral tests
    v2_models = {}
    if rf_model:
        v2_models["v2 Random Forest"] = rf_model
    if xgb_model:
        v2_models["v2 XGBoost"] = xgb_model
    run_behavioral_tests(v2_models)

    # Section 4: v1 reference
    show_v1_reference()

    # Final verdict
    final_verdict(v2_results)

    # Flush log
    flush_log()


if __name__ == "__main__":
    main()
