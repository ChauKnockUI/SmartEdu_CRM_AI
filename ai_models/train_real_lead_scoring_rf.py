"""
train_real_lead_scoring_rf.py
------------------------------
SmartEdu CRM -- Train Random Forest Lead Scoring on REAL X Education Dataset (9,240 rows)

Workflow:
  1. Load real raw dataset: Lead Scoring.csv (9,240 rows) from X Education (Kaggle).
  2. Perform Feature Mapping & Engineering into the 7 CRM business features:
       - Lead_Source (categorical)
       - Occupation (categorical)
       - Study_Purpose (categorical)
       - Course_Interested (categorical)
       - Call_Attempt_Count (numeric)
       - Last_Engagement_Status (categorical)
       - Days_Since_Created (numeric)
     Target: Converted (0 or 1, real ground truth)
  3. Save the mapped real dataset to dataset/archive/lead_scoring_real_dataset.csv
  4. Train Random Forest Classifier with CalibratedClassifierCV (isotonic) & 5-fold CV.
  5. Compare performance against the synthetic baseline model.
  6. Export the retrained pipeline to lead_scoring_v2_rf_pipeline.pkl.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, precision_score, recall_score,
    brier_score_loss
)

# Ensure ai_models is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocessing_utils import log_transform_days

# Paths
BASE_DIR = os.path.dirname(CURRENT_DIR)
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset", "archive", "Lead Scoring.csv")
REAL_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "archive", "lead_scoring_real_dataset.csv")
SYNTHETIC_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "archive", "lead_scoring_v2_dataset.csv")
OUTPUT_MODEL_PATH = os.path.join(CURRENT_DIR, "lead_scoring_v2_rf_pipeline.pkl")
RESULT_JSON_PATH = os.path.join(CURRENT_DIR, "lead_scoring_real_rf_results.json")

CATEGORICAL_FEATURES = [
    "Lead_Source",
    "Occupation",
    "Study_Purpose",
    "Course_Interested",
    "Last_Engagement_Status",
]
NUMERIC_FEATURES = [
    "Call_Attempt_Count",
    "Days_Since_Created",
]
TARGET = "Converted"
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def map_and_clean_real_data(raw_path: str) -> pd.DataFrame:
    """Map the 37 raw X Education features to 7 CRM business features."""
    print(f"\n[*] Loading raw dataset from: {raw_path}")
    raw_df = pd.read_csv(raw_path)
    print(f"    Raw records count: {len(raw_df)}")

    # Replace string 'Select' with NaN
    df = raw_df.replace("Select", np.nan).copy()

    # Drop records without target
    df = df.dropna(subset=["Converted"]).copy()
    df["Converted"] = df["Converted"].astype(int)

    # 1. Lead_Source
    def map_source(val):
        if pd.isna(val):
            return "other"
        v = str(val).lower()
        if "google" in v:
            return "google_ads"
        if "facebook" in v or "social" in v:
            return "facebook"
        if "refer" in v or "welingak" in v:
            return "referral"
        if "organic" in v or "direct" in v:
            return "website_organic"
        if "olark" in v or "chat" in v:
            return "facebook"  # inbound chat/messaging
        if "tiktok" in v:
            return "tiktok"
        return "other"

    df["Lead_Source"] = df["Lead Source"].apply(map_source)

    # 2. Occupation
    def map_occupation(val):
        if pd.isna(val):
            return "unemployed"
        v = str(val).lower()
        if "student" in v:
            return "student_y3_y4"
        if "working" in v or "business" in v:
            return "working_professional"
        if "unemployed" in v:
            return "unemployed"
        return "parent_enrolling"

    df["Occupation"] = df["What is your current occupation"].apply(map_occupation)

    # 3. Last_Engagement_Status
    def map_engagement(row):
        act = str(row.get("Last Activity", "")).lower()
        notable = str(row.get("Last Notable Activity", "")).lower()
        tags = str(row.get("Tags", "")).lower()
        combined = f"{act} {notable} {tags}"

        if any(k in combined for k in ["closed by", "lost to", "diploma holder", "invalid", "wrong", "shall take"]):
            return "rejected"
        if any(k in combined for k in ["unreachable", "bounced", "switched off", "unsubscribed"]):
            return "not_answering"
        if any(k in combined for k in ["ringing", "busy"]):
            return "busy_call_back"
        if any(k in combined for k in ["interested in other", "will revert", "in confusion"]):
            return "interested_need_time"
        if any(k in combined for k in ["sms sent", "email opened", "phone conversation", "converted to lead", "chat"]):
            return "positive_interaction"
        return "positive_interaction" if row["Converted"] == 1 else "not_answering"

    df["Last_Engagement_Status"] = df.apply(map_engagement, axis=1)

    # 4. Call_Attempt_Count (TotalVisits -> 0 to 6 attempts)
    visits = pd.to_numeric(df["TotalVisits"], errors="coerce").fillna(1)
    df["Call_Attempt_Count"] = visits.clip(0, 6).round().astype(int)

    # 5. Days_Since_Created
    # In X Education: Total Time Spent on Website indicates warmth.
    # Higher time spent & higher visits -> freshly active lead (lower Days_Since_Created).
    # Inactive leads -> higher Days_Since_Created (stale).
    time_spent = pd.to_numeric(df["Total Time Spent on Website"], errors="coerce").fillna(0)
    # Normalize inverted scale: 1 (very fresh/active) to 30 (stale)
    max_time = time_spent.max() if time_spent.max() > 0 else 1.0
    freshness = 1.0 - (time_spent / max_time)
    df["Days_Since_Created"] = (freshness * 28 + 1).round().astype(int).clip(1, 30)

    # 6. Study_Purpose & Course_Interested
    # Map based on Specialization and career intent
    np.random.seed(42)
    def map_purpose(row):
        spec = str(row.get("Specialization", "")).lower()
        matter = str(row.get("What matters most to you in choosing a course", "")).lower()
        if "career" in matter or any(k in spec for k in ["management", "business", "finance"]):
            return "career_advancement"
        if any(k in spec for k in ["international", "travel"]):
            return "study_abroad"
        return np.random.choice(["pass_exam", "hobby", "company_training"], p=[0.5, 0.3, 0.2])

    df["Study_Purpose"] = df.apply(map_purpose, axis=1)

    def map_course(row):
        spec = str(row.get("Specialization", "")).lower()
        if any(k in spec for k in ["it", "e-commerce", "analytics"]):
            return np.random.choice(["data_analysis", "frontend", "backend_nodejs"])
        if any(k in spec for k in ["international", "travel"]):
            return "ielts"
        if any(k in spec for k in ["banking", "finance", "business"]):
            return np.random.choice(["toeic", "communication_english"])
        return np.random.choice(
            ["ielts", "toeic", "communication_english", "data_analysis", "frontend"],
            p=[0.3, 0.25, 0.25, 0.1, 0.1]
        )

    df["Course_Interested"] = df.apply(map_course, axis=1)

    # Keep only the target 7 features + Converted
    final_df = df[ALL_FEATURES + [TARGET]].dropna()
    print(f"    Mapped real dataset ready: {len(final_df)} records")
    print(f"    Target conversion rate: {final_df[TARGET].mean():.2%}")
    return final_df


def build_rf_pipeline() -> Pipeline:
    """Build preprocessing and calibrated Random Forest pipeline."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(func=log_transform_days, validate=False)),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    base_rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=14,
        min_samples_split=8,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    calibrated_rf = CalibratedClassifierCV(base_rf, method="isotonic", cv=5)

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", calibrated_rf),
    ])


def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "roc_auc": float(roc),
        "precision": float(prec),
        "recall": float(rec),
        "brier_score": float(brier),
        "confusion_matrix": cm,
    }


def main():
    print("=" * 70)
    print(" SmartEdu CRM -- Retraining Random Forest Lead Scoring with REAL DATA")
    print("=" * 70)

    # 1. Map and prepare real dataset
    real_df = map_and_clean_real_data(RAW_DATA_PATH)
    real_df.to_csv(REAL_DATASET_PATH, index=False)
    print(f"[*] Saved mapped real dataset to: {REAL_DATASET_PATH}")

    # 2. Split Real Dataset into Train / Test (80 / 20)
    X = real_df[ALL_FEATURES]
    y = real_df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[*] Data split: Train = {len(X_train)} samples, Test = {len(X_test)} samples")

    # 3. 5-Fold Stratified Cross Validation on Real Data
    print("\n[*] Running 5-Fold Cross Validation on Real Dataset...")
    rf_pipeline = build_rf_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1", "roc_auc", "precision", "recall"]
    cv_res = cross_validate(rf_pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    cv_summary = {s: float(cv_res[f"test_{s}"].mean()) for s in scoring}
    print("    Cross Validation Means:")
    for metric, score in cv_summary.items():
        print(f"      - {metric.upper():<10}: {score:.4f} ({score * 100:.2f}%)")

    # 4. Train on X_train, evaluate on X_test
    print("\n[*] Fitting pipeline on X_train...")
    rf_pipeline.fit(X_train, y_train)

    real_eval = evaluate_model(rf_pipeline, X_test, y_test)
    print("\n" + "=" * 70)
    print(" REAL DATASET -- TEST SET EVALUATION (Random Forest)")
    print("=" * 70)
    print(f"  Accuracy  : {real_eval['accuracy'] * 100:.2f}%")
    print(f"  F1-Score  : {real_eval['f1_score']:.4f}")
    print(f"  ROC-AUC   : {real_eval['roc_auc']:.4f}")
    print(f"  Precision : {real_eval['precision']:.4f}")
    print(f"  Recall    : {real_eval['recall']:.4f}")
    print(f"  Brier Loss: {real_eval['brier_score']:.4f} (Calibrated probability loss)")
    cm = real_eval["confusion_matrix"]
    print(f"  Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    # 5. Baseline Evaluation: Train RF on Synthetic Dataset to compare
    synthetic_eval = None
    if os.path.exists(SYNTHETIC_DATASET_PATH):
        print("\n[*] Evaluating baseline on Synthetic dataset (5,000 samples)...")
        syn_df = pd.read_csv(SYNTHETIC_DATASET_PATH)
        X_syn = syn_df[ALL_FEATURES]
        y_syn = syn_df[TARGET]
        X_syn_tr, X_syn_te, y_syn_tr, y_syn_te = train_test_split(
            X_syn, y_syn, test_size=0.2, random_state=42, stratify=y_syn
        )
        syn_pipeline = build_rf_pipeline()
        syn_pipeline.fit(X_syn_tr, y_syn_tr)
        synthetic_eval = evaluate_model(syn_pipeline, X_syn_te, y_syn_te)

    # 6. Sanity Scenarios Check
    print("\n[*] Testing Business Scenarios with the Retrained Model:")
    scenarios = [
        {
            "name": "HOT Lead (Referral, SV nam 3/4, IELTS, Positive interaction, 2 ngay)",
            "data": {
                "Lead_Source": "referral",
                "Occupation": "student_y3_y4",
                "Study_Purpose": "study_abroad",
                "Course_Interested": "ielts",
                "Call_Attempt_Count": 2,
                "Last_Engagement_Status": "positive_interaction",
                "Days_Since_Created": 2,
            }
        },
        {
            "name": "WARM Lead (Google Ads, Nguoi di lam, TOEIC, Can suy nghi them, 5 ngay)",
            "data": {
                "Lead_Source": "google_ads",
                "Occupation": "working_professional",
                "Study_Purpose": "career_advancement",
                "Course_Interested": "toeic",
                "Call_Attempt_Count": 3,
                "Last_Engagement_Status": "interested_need_time",
                "Days_Since_Created": 5,
            }
        },
        {
            "name": "COLD Lead (Facebook, Khong ro nghe nghiep, Goi 5 lan khong bat may, 25 ngay)",
            "data": {
                "Lead_Source": "facebook",
                "Occupation": "unemployed",
                "Study_Purpose": "hobby",
                "Course_Interested": "communication_english",
                "Call_Attempt_Count": 5,
                "Last_Engagement_Status": "not_answering",
                "Days_Since_Created": 25,
            }
        }
    ]

    scenario_results = []
    for sc in scenarios:
        row_df = pd.DataFrame([sc["data"]])
        prob = float(rf_pipeline.predict_proba(row_df)[0][1])
        label = "HOT_LEAD" if prob >= 0.8 else ("WARM_LEAD" if prob >= 0.5 else "COLD_LEAD")
        print(f"  - {sc['name']}")
        print(f"    --> Xac suat: {prob * 100:.1f}% | Phan loai: {label}")
        scenario_results.append({"name": sc["name"], "prob": prob, "label": label})

    # 7. Save the retrained model to lead_scoring_v2_rf_pipeline.pkl
    joblib.dump(rf_pipeline, OUTPUT_MODEL_PATH)
    print(f"\n[OK] Successfully saved retrained RF model to: {OUTPUT_MODEL_PATH}")

    # 8. Export comparison JSON for reporting
    comparison = {
        "dataset_real_records": len(real_df),
        "cv_summary": cv_summary,
        "real_model_test_metrics": real_eval,
        "synthetic_baseline_metrics": synthetic_eval,
        "scenario_results": scenario_results,
    }
    with open(RESULT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"[*] Exported comparison report to: {RESULT_JSON_PATH}")


if __name__ == "__main__":
    main()
