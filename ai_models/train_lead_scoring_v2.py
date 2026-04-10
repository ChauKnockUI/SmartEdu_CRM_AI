"""
train_lead_scoring_v2.py
-------------------------
SmartEdu CRM — Lead Scoring v2 Model Training Pipeline

Trains TWO models on the new 7-feature business-aligned dataset:
  1. Random Forest  (baseline, consistent with v1)
  2. XGBoost        (gradient boosting, typically better on tabular data)

Features (7 total):
  Categorical (5): Lead_Source, Occupation, Study_Purpose,
                   Course_Interested, Last_Engagement_Status
  Numeric (2):     Call_Attempt_Count, Days_Since_Created

Evaluation:
  - Stratified 5-Fold Cross Validation
  - Metrics: Accuracy, F1, ROC-AUC, Precision, Recall
  - Feature importance analysis
  - Confusion matrix

Outputs:
  - lead_scoring_v2_rf_pipeline.pkl   (Random Forest)
  - lead_scoring_v2_xgb_pipeline.pkl  (XGBoost)
  - Console log with full evaluation report

Usage:
    python train_lead_scoring_v2.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, precision_score, recall_score,
    brier_score_loss, precision_recall_curve
)
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARNING] XGBoost not installed. Will train Random Forest only.")
    print("          Install with: pip install xgboost")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "dataset", "archive", "lead_scoring_v2_dataset.csv"
)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

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


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Run generate_lead_scoring_v2_dataset.py first."
        )

    df = pd.read_csv(path)

    # Validate required columns
    missing = set(ALL_FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    print(f"  Data loaded: {len(X)} records")
    print(f"  Conversion rate: {y.mean():.1%}")
    print(f"  Features: {ALL_FEATURES}")

    return X, y

# ---------------------------------------------------------------------------
# Preprocessing Pipeline Builder
# ---------------------------------------------------------------------------
from preprocessing_utils import log_transform_days


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer that handles both feature types.
    
    Numeric:     Impute median -> log1p transform (Days_Since_Created) -> StandardScaler
    Categorical: Impute 'Unknown' -> OneHotEncoder
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(
            func=log_transform_days,
            validate=False
        )),
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

    return preprocessor


# ---------------------------------------------------------------------------
# Model Builders
# ---------------------------------------------------------------------------
def build_rf_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Build Calibrated Random Forest pipeline."""
    base_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    # Use Isotonic regression for RF to smooth out the steps
    calibrated_rf = CalibratedClassifierCV(base_rf, method='isotonic', cv=5)

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", calibrated_rf),
    ])


def build_xgb_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    """Build Calibrated XGBoost pipeline."""
    base_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.7,      # ~= (1 - 0.375) / 0.375 for imbalance
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )
    # Platt scaling (sigmoid) is usually best for gradient boosting trees
    calibrated_xgb = CalibratedClassifierCV(base_xgb, method='sigmoid', cv=5)

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", calibrated_xgb),
    ])


# ---------------------------------------------------------------------------
# Cross Validation
# ---------------------------------------------------------------------------
def run_cross_validation(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series,
                         model_name: str) -> dict:
    """Run stratified 5-fold CV and return aggregated metrics."""
    print(f"\n  Running 5-Fold Stratified Cross Validation for {model_name}...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy":  "accuracy",
        "f1":        "f1",
        "roc_auc":   "roc_auc",
        "precision": "precision",
        "recall":    "recall",
    }

    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    metrics = {}
    for metric_name in scoring:
        key = f"test_{metric_name}"
        values = cv_results[key]
        metrics[metric_name] = {
            "mean": values.mean(),
            "std":  values.std(),
            "values": values,
        }

    print(f"\n  {model_name} — Cross Validation Results:")
    print(f"  {'Metric':<12}  {'Mean':>8}  {'Std':>8}  {'Folds':}")
    print(f"  {'-'*60}")
    for name, data in metrics.items():
        fold_str = "  ".join(f"{v:.3f}" for v in data["values"])
        print(f"  {name:<12}  {data['mean']:>8.4f}  {data['std']:>8.4f}  [{fold_str}]")

    return metrics


# ---------------------------------------------------------------------------
# Full Evaluation on Test Set
# ---------------------------------------------------------------------------
def evaluate_on_test_set(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                         model_name: str) -> dict:
    """Evaluate a trained pipeline on held-out test data."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc       = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    brier     = brier_score_loss(y_test, y_proba)
    cm        = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*65}")
    print(f"  {model_name} — Test Set Evaluation")
    print(f"{'='*65}")
    print(f"  Accuracy  : {acc:.2%}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  Brier Loss: {brier:.4f} (Lower is better)")

    print(f"\n  Confusion Matrix:")
    print(f"    TP={cm[1][1]:4d}  FP={cm[0][1]:4d}")
    print(f"    FN={cm[1][0]:4d}  TN={cm[0][0]:4d}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Converted", "Converted"]))

    return {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
        "precision": precision,
        "recall": recall,
        "brier_score": brier,
        "y_proba": y_proba,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Feature Importance Analysis
# ---------------------------------------------------------------------------
def analyze_feature_importance(pipeline: Pipeline, model_name: str) -> None:
    """Extract and display feature importance from trained model."""
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    # If wrapped in CalibratedClassifierCV, get the base estimators
    if hasattr(classifier, "calibrated_classifiers_"):
        # Just grab the first calibrated fold for feature importance estimation
        base_estimator = classifier.calibrated_classifiers_[0].estimator
    else:
        base_estimator = classifier

    # Get feature names after preprocessing
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        print(f"  [SKIP] Cannot extract feature names for {model_name}")
        return

    # Get importances
    if hasattr(base_estimator, "feature_importances_"):
        importances = base_estimator.feature_importances_
    else:
        print(f"  [SKIP] {model_name} does not support feature_importances_")
        return

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    print(f"\n  {model_name} — Top 15 Feature Importances:")
    print(f"  {'Rank':<5} {'Feature':<45} {'Importance':>10}")
    print(f"  {'-'*62}")
    for rank, idx in enumerate(indices[:15], 1):
        name = feature_names[idx]
        # Clean up sklearn naming
        name = name.replace("cat__", "").replace("num__", "")
        print(f"  {rank:<5} {name:<45} {importances[idx]:>10.4f}")


# ---------------------------------------------------------------------------
# Prediction Simulation
# ---------------------------------------------------------------------------
def run_simulation(pipeline: Pipeline, model_name: str) -> None:
    """Test the model with realistic CRM scenarios."""
    scenarios = [
        {
            "name": "[HOT] Lead -- Referral, SV nam 4, du hoc, positive, 1 ngay",
            "data": {
                "Lead_Source": "referral",
                "Occupation": "student_y3_y4",
                "Study_Purpose": "study_abroad",
                "Course_Interested": "ielts",
                "Call_Attempt_Count": 1,
                "Last_Engagement_Status": "positive_interaction",
                "Days_Since_Created": 1,
            }
        },
        {
            "name": "[COLD] Lead -- TikTok, that nghiep, hobby, khong nghe may, 20 ngay",
            "data": {
                "Lead_Source": "tiktok",
                "Occupation": "unemployed",
                "Study_Purpose": "hobby",
                "Course_Interested": "communication_english",
                "Call_Attempt_Count": 6,
                "Last_Engagement_Status": "not_answering",
                "Days_Since_Created": 20,
            }
        },
        {
            "name": "[WARM] Lead -- Google Ads, nguoi di lam, thang tien, can thoi gian, 5 ngay",
            "data": {
                "Lead_Source": "google_ads",
                "Occupation": "working_professional",
                "Study_Purpose": "career_advancement",
                "Course_Interested": "toeic",
                "Call_Attempt_Count": 2,
                "Last_Engagement_Status": "interested_need_time",
                "Days_Since_Created": 5,
            }
        },
    ]

    print(f"\n{'='*65}")
    print(f"  {model_name} — Prediction Simulation")
    print(f"{'='*65}")

    for s in scenarios:
        input_df = pd.DataFrame([s["data"]])
        pred = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]

        if prob >= 0.80:
            label = "HOT_LEAD"
        elif prob >= 0.50:
            label = "WARM_LEAD"
        else:
            label = "COLD_LEAD"

        print(f"\n  {s['name']}")
        print(f"    Prediction  : {'Converted' if pred == 1 else 'Not Converted'}")
        print(f"    Probability : {prob:.1%}")
        print(f"    Label       : {label}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("  SmartEdu CRM — Lead Scoring v2 Training Pipeline")
    print("  Models: Random Forest + XGBoost")
    print("=" * 65)
    print()

    # --- Load Data ---
    try:
        X, y = load_data(DATASET_PATH)
    except Exception as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {len(X_train)} records | Test: {len(X_test)} records")

    # --- Build Preprocessors (separate instances for each model) ---
    results = {}

    # ===================== RANDOM FOREST =====================
    print(f"\n{'#'*65}")
    print(f"  MODEL 1: Random Forest")
    print(f"{'#'*65}")

    rf_pipeline = build_rf_pipeline(build_preprocessor())

    # Cross-validation
    rf_cv = run_cross_validation(rf_pipeline, X_train, y_train, "Random Forest")

    # Train on full training set
    print(f"\n  Training Random Forest on full training set...")
    rf_pipeline.fit(X_train, y_train)
    print(f"  Training complete.")

    # Evaluate
    rf_test = evaluate_on_test_set(rf_pipeline, X_test, y_test, "Random Forest")
    results["Random Forest"] = {"cv": rf_cv, "test": rf_test}

    # Feature importance
    analyze_feature_importance(rf_pipeline, "Random Forest")

    # Simulation
    run_simulation(rf_pipeline, "Random Forest")

    # Export
    rf_path = os.path.join(MODEL_DIR, "lead_scoring_v2_rf_pipeline.pkl")
    joblib.dump(rf_pipeline, rf_path)
    print(f"\n  Random Forest model saved: {rf_path}")

    # ===================== XGBOOST =====================
    if HAS_XGBOOST:
        print(f"\n{'#'*65}")
        print(f"  MODEL 2: XGBoost")
        print(f"{'#'*65}")

        xgb_pipeline = build_xgb_pipeline(build_preprocessor())

        # Cross-validation
        xgb_cv = run_cross_validation(xgb_pipeline, X_train, y_train, "XGBoost")

        # Train on full training set
        print(f"\n  Training XGBoost on full training set...")
        xgb_pipeline.fit(X_train, y_train)
        print(f"  Training complete.")

        # Evaluate
        xgb_test = evaluate_on_test_set(xgb_pipeline, X_test, y_test, "XGBoost")
        results["XGBoost"] = {"cv": xgb_cv, "test": xgb_test}

        # Feature importance
        analyze_feature_importance(xgb_pipeline, "XGBoost")

        # Simulation
        run_simulation(xgb_pipeline, "XGBoost")

        # Export
        xgb_path = os.path.join(MODEL_DIR, "lead_scoring_v2_xgb_pipeline.pkl")
        joblib.dump(xgb_pipeline, xgb_path)
        print(f"\n  XGBoost model saved: {xgb_path}")

    # ===================== COMPARISON SUMMARY =====================
    print(f"\n{'='*65}")
    print(f"  FINAL COMPARISON — Test Set Results")
    print(f"{'='*65}")
    print(f"\n  {'Model':<18} {'Accuracy':>10} {'F1':>10} {'ROC-AUC':>10} {'BrierLoss':>10}")
    print(f"  {'-'*65}")
    for name, data in results.items():
        t = data["test"]
        print(f"  {name:<18} {t['accuracy']:>10.2%} {t['f1']:>10.4f} {t['roc_auc']:>10.4f} "
              f"{t['brier_score']:>10.4f}")

    # Plot Calibration Curve
    plot_path = os.path.join(MODEL_DIR, "calibration_reliability_curve.png")
    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name, data in results.items():
        probas = data["test"]["y_proba"]
        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probas, n_bins=10)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{name}")
        ax2.hist(probas, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title("Calibration Plots (Reliability Curve)")

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\n  [INFO] Calibration Curve plotted to: {plot_path}")

    # Winner
    if len(results) > 1:
        best_model = max(results.items(), key=lambda x: x[1]["test"]["roc_auc"])
        print(f"\n  [WINNER] Best Model (by ROC-AUC): {best_model[0]}")

    # ===================== THRESHOLD ANALYSIS & PR CURVE =====================
    print(f"\n{'='*65}")
    print(f"  THRESHOLD OPTIMIZATION (Maximize Recall)")
    print(f"{'='*65}")
    
    # Plot Precision-Recall Curve
    pr_plot_path = os.path.join(MODEL_DIR, "precision_recall_curve.png")
    plt.figure(figsize=(8, 6))
    
    for name, data in results.items():
        y_proba = data["test"]["y_proba"]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, lw=2, label=f"{name}")
        
        # Analyze discrete thresholds for this model
        print(f"\n  Model: {name}")
        print(f"  {'Threshold':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
        print(f"  {'-'*50}")
        for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
            y_pred_t = (y_proba >= t).astype(int)
            p = precision_score(y_test, y_pred_t, zero_division=0)
            r = recall_score(y_test, y_pred_t, zero_division=0)
            f1 = f1_score(y_test, y_pred_t, zero_division=0)
            mark = "  <-- MẶC ĐỊNH" if t == 0.5 else ""
            print(f"  {t:<12.2f} {p:>10.4f} {r:>10.4f} {f1:>10.4f}{mark}")
            
    plt.xlabel("Recall (Tỷ lệ bắt được KH tiềm năng)")
    plt.ylabel("Precision (Độ chính xác khi dự đoán Chốt)")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(pr_plot_path)
    print(f"\n  [INFO] Precision-Recall Curve plotted to: {pr_plot_path}")

    print(f"\n{'='*65}")
    print(f"  Training pipeline complete. Models ready for deployment.")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
