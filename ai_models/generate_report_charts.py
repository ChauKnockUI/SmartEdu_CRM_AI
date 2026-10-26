"""
generate_report_charts.py
--------------------------
SmartEdu CRM -- Generate Professional Academic & Industrial Report Charts (English)

Model: Random Forest Classifier (Trained on 9,240 Real X Education Records)
Outputs saved to: ai_models/reports/
  1. 1_roc_curve.png                    -- ROC-AUC Evaluation Curve (AUC = 0.91)
  2. 2_confusion_matrix.png             -- Confusion Matrix Heatmap (Test Set)
  3. 3_precision_recall_curve.png        -- Precision-Recall Curve & Baseline
  4. 4_feature_importance.png            -- Top 12 Influential Features (MDI / Gini)
  5. 5_calibration_curve.png             -- Probability Reliability Diagram (Brier Score)
  6. lead_scoring_dashboard_summary.png  -- All-in-one 4-in-1 Dashboard for Slides/Reports
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    brier_score_loss, accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocessing_utils import log_transform_days  # noqa: F401

BASE_DIR = os.path.dirname(CURRENT_DIR)
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "archive", "lead_scoring_real_dataset.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "lead_scoring_v2_rf_pipeline.pkl")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clean, modern publication styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8


def load_model_and_test_data():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    pipeline = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATASET_PATH)

    features = [
        "Lead_Source", "Occupation", "Study_Purpose", "Course_Interested",
        "Last_Engagement_Status", "Call_Attempt_Count", "Days_Since_Created"
    ]
    X = df[features]
    y = df["Converted"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    return pipeline, X_test, y_test, y_prob, y_pred


def plot_roc_curve(y_test, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"Random Forest (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#7f7f7f", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.5000)")
    plt.fill_between(fpr, tpr, alpha=0.1, color="#1f77b4")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    plt.title("RECEIVER OPERATING CHARACTERISTIC (ROC) CURVE", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_confusion_matrix(y_test, y_pred, save_path):
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6.5, 5.5), dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("CONFUSION MATRIX (TEST SET)", fontsize=13, fontweight="bold", pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)

    classes = ["Not Converted (0)", "Converted (1)"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10, fontweight="bold")
    plt.yticks(tick_marks, classes, fontsize=10, fontweight="bold")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm[i, j]:,}\n({cm_norm[i, j]:.1%})"
            plt.text(j, i, val_str,
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=12, fontweight="bold",
                     color="white" if cm[i, j] > thresh else "#222222")

    plt.ylabel("True Label (Ground Truth)", fontsize=11, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_precision_recall_curve(y_test, y_prob, save_path):
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot(recall, precision, color="#2ca02c", lw=2.5, label=f"Precision-Recall (PR-AUC = {pr_auc:.4f})")
    plt.axhline(y=y_test.mean(), color="#d62728", linestyle="--", label=f"Baseline (Positive Rate = {y_test.mean():.1%})")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall", fontsize=11, fontweight="bold")
    plt.ylabel("Precision", fontsize=11, fontweight="bold")
    plt.title("PRECISION-RECALL (PR) CURVE", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_feature_importance(pipeline, save_path):
    preprocessor = pipeline.named_steps["preprocessor"]
    calibrated_clf = pipeline.named_steps["classifier"]

    # Average feature importances across all 5 calibrated estimator folds
    importances = np.mean(
        [clf.estimator.feature_importances_ for clf in calibrated_clf.calibrated_classifiers_],
        axis=0
    )

    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "cat":
            ohe = trans.named_steps["onehot"]
            feats = ohe.get_feature_names_out(cols)
            feature_names.extend(feats)
        elif name == "num":
            feature_names.extend(cols)

    # Professional English feature descriptions
    english_labels = {
        "Last_Engagement_Status_positive_interaction": "Engagement: Positive Interaction",
        "Last_Engagement_Status_rejected": "Engagement: Rejected / No Intent",
        "Last_Engagement_Status_not_answering": "Engagement: Not Answering",
        "Last_Engagement_Status_busy_call_back": "Engagement: Busy / Call Back Later",
        "Last_Engagement_Status_interested_need_time": "Engagement: Interested / Needs Time",
        "Occupation_working_professional": "Occupation: Working Professional",
        "Occupation_student_y3_y4": "Occupation: Senior Student (Y3-Y4)",
        "Occupation_unemployed": "Occupation: Unemployed",
        "Occupation_parent_enrolling": "Occupation: Parent Enrolling",
        "Days_Since_Created": "Lead Staleness (Days Since Created)",
        "Call_Attempt_Count": "Sales Intensity (Call Attempt Count)",
        "Lead_Source_referral": "Lead Source: Referral / Word-of-Mouth",
        "Lead_Source_google_ads": "Lead Source: Google Search Ads",
        "Lead_Source_website_organic": "Lead Source: Organic Search / Website",
        "Lead_Source_facebook": "Lead Source: Facebook Ads",
        "Study_Purpose_career_advancement": "Study Goal: Career Advancement",
        "Study_Purpose_study_abroad": "Study Goal: Study Abroad (IELTS)",
        "Course_Interested_ielts": "Target Course: IELTS Preparation",
        "Course_Interested_toeic": "Target Course: TOEIC Preparation",
    }

    clean_names = [english_labels.get(f, f.replace("_", " ")) for f in feature_names]

    # Top 12 most influential features
    indices = np.argsort(importances)[::-1][:12]
    top_importances = importances[indices][::-1]
    top_names = [clean_names[i] for i in indices][::-1]

    plt.figure(figsize=(9, 6.5), dpi=300)
    bars = plt.barh(range(len(top_names)), top_importances, color="#17becf", edgecolor="#0e8a96", height=0.65)
    plt.yticks(range(len(top_names)), top_names, fontsize=10, fontweight="bold")
    plt.xlabel("Feature Importance Score (MDI / Gini Impurity)", fontsize=11, fontweight="bold")
    plt.title("TOP 12 MOST INFLUENTIAL FEATURES (RANDOM FOREST)", fontsize=13, fontweight="bold", pad=15)

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width * 100:.1f}%",
                 va='center', ha='left', fontsize=9, fontweight="bold", color="#333333")

    plt.xlim(0, max(top_importances) * 1.18)
    plt.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_calibration_curve(y_test, y_prob, save_path):
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    brier = brier_score_loss(y_test, y_prob)

    plt.figure(figsize=(7, 6), dpi=300)
    plt.plot([0, 1], [0, 1], "k:", lw=1.5, label="Perfect Calibration (Ideal)")
    plt.plot(prob_pred, prob_true, "s-", color="#ff7f0e", lw=2.5,
             label=f"Calibrated RF (Isotonic, Brier = {brier:.4f})")

    plt.xlabel("Mean Predicted Probability", fontsize=11, fontweight="bold")
    plt.ylabel("Empirical Fraction of Positives", fontsize=11, fontweight="bold")
    plt.title("PROBABILITY CALIBRATION CURVE (RELIABILITY DIAGRAM)", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def plot_summary_dashboard(y_test, y_prob, y_pred, pipeline, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)
    fig.suptitle("SMARTEDU CRM -- LEAD SCORING MODEL EVALUATION DASHBOARD",
                 fontsize=18, fontweight="bold", y=0.98)

    # 1. ROC
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1f77b4")
    ax.set_title("A. RECEIVER OPERATING CHARACTERISTIC (ROC)", fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    # 2. Confusion Matrix
    ax = axes[0, 1]
    cm = confusion_matrix(y_test, y_pred)
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("B. CONFUSION MATRIX (TEST SET, N=1,848)", fontsize=13, fontweight="bold")
    classes = ["Not Converted", "Converted"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontweight="bold")
    ax.set_yticklabels(classes, fontweight="bold")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("Ground Truth", fontweight="bold")
    ax.set_xlabel("Predicted Label", fontweight="bold")

    # 3. Precision-Recall
    ax = axes[1, 0]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec, prec, color="#2ca02c", lw=2.5, label=f"PR Curve (AUC = {auc(rec, prec):.4f})")
    ax.axhline(y=y_test.mean(), color="#d62728", linestyle="--", label=f"Baseline ({y_test.mean():.1%})")
    ax.set_title("C. PRECISION-RECALL (PR) CURVE", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)

    # 4. Calibration
    ax = axes[1, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ax.plot([0, 1], [0, 1], "k:", lw=1.5, label="Ideal")
    ax.plot(prob_pred, prob_true, "s-", color="#ff7f0e", lw=2.5,
            label=f"Calibrated RF (Brier = {brier_score_loss(y_test, y_prob):.4f})")
    ax.set_title("D. PROBABILITY CALIBRATION (RELIABILITY)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Empirical Fraction of Positives", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"  [OK] Saved: {save_path}")


def main():
    print("=" * 70)
    print(" GENERATING LEAD SCORING EVALUATION CHARTS (ENGLISH)")
    print("=" * 70)

    pipeline, X_test, y_test, y_prob, y_pred = load_model_and_test_data()

    print(f"[*] Test Set: {len(X_test)} samples")
    print(f"    - Accuracy : {accuracy_score(y_test, y_pred):.2%}")
    print(f"    - F1-Score : {f1_score(y_test, y_pred):.4f}")
    print(f"    - Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"    - Recall   : {recall_score(y_test, y_pred):.4f}")

    print("\n[*] Rendering high-resolution figures (300 DPI)...")
    plot_roc_curve(y_test, y_prob, os.path.join(OUTPUT_DIR, "1_roc_curve.png"))
    plot_confusion_matrix(y_test, y_pred, os.path.join(OUTPUT_DIR, "2_confusion_matrix.png"))
    plot_precision_recall_curve(y_test, y_prob, os.path.join(OUTPUT_DIR, "3_precision_recall_curve.png"))
    plot_feature_importance(pipeline, os.path.join(OUTPUT_DIR, "4_feature_importance.png"))
    plot_calibration_curve(y_test, y_prob, os.path.join(OUTPUT_DIR, "5_calibration_curve.png"))
    plot_summary_dashboard(y_test, y_prob, y_pred, pipeline, os.path.join(OUTPUT_DIR, "lead_scoring_dashboard_summary.png"))

    print("\n" + "=" * 70)
    print(f" COMPLETE! All 6 English figures (300 DPI) are saved at:")
    print(f" >> {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
