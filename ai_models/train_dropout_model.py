"""
train_dropout_model.py
-----------------------
Trains the Student Dropout Risk prediction model for the SmartEdu CRM system.

This model identifies students at risk of dropping out of an English language
course based on attendance and homework engagement signals.

Pipeline:
    1. Load dataset (dropout_dataset.csv)
    2. Split into train / test (80 / 20, stratified)
    3. Build sklearn Pipeline: Imputer -> Scaler -> RandomForestClassifier
    4. Train and evaluate: Accuracy, F1, ROC-AUC, Confusion Matrix
    5. Report feature importances
    6. Run three end-to-end simulation predictions
    7. Export trained pipeline to dropout_model_pipeline.pkl

Output:
    ai_models/dropout_model_pipeline.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import StandardScaler
from sklearn.ensemble          import RandomForestClassifier
from sklearn.metrics           import (accuracy_score, f1_score,
                                        classification_report,
                                        confusion_matrix, roc_auc_score)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "archive", "dropout_dataset.csv")
MODEL_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH   = os.path.join(MODEL_DIR, "dropout_model_pipeline.pkl")

RANDOM_SEED  = 42

FEATURE_COLS = [
    "Excused_Absences",
    "Unexcused_Absences",
    "Consecutive_Absences",
    "Missed_Homework_Ratio",
]
TARGET_COL = "Is_Dropout"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the dropout dataset and perform basic validation checks.

    Args:
        path: Absolute path to the CSV file.

    Returns:
        Tuple of (X, y) — feature DataFrame and target Series.

    Raises:
        FileNotFoundError: If the dataset CSV does not exist.
        ValueError: If any required column is missing.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Run generate_dropout_dataset.py first."
        )

    df = pd.read_csv(path)

    required = FEATURE_COLS + [TARGET_COL]
    missing  = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = df[required].isnull().sum()
    if null_counts.any():
        # Null values will be handled inside the pipeline (SimpleImputer)
        print(f"  [INFO] Null values detected (will be imputed):\n{null_counts[null_counts > 0]}")

    total     = len(df)
    n_dropout = int(df[TARGET_COL].sum())
    print(f"  Dataset loaded: {total:,} records  |  "
          f"Dropout: {n_dropout:,} ({n_dropout / total * 100:.1f}%)  |  "
          f"Safe: {total - n_dropout:,} ({(total - n_dropout) / total * 100:.1f}%)")

    return df[FEATURE_COLS], df[TARGET_COL]


# ---------------------------------------------------------------------------
# Model pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Construct the sklearn preprocessing + classification pipeline.

    Preprocessing:
        - SimpleImputer  (strategy='median'): handles any NaN values robustly
        - StandardScaler : normalizes feature ranges for consistent behavior

    Classifier:
        - RandomForestClassifier with class_weight='balanced' to automatically
          compensate for class imbalance (fewer dropouts than non-dropouts).
          200 estimators and max_depth=12 balance accuracy vs. overfitting risk.

    Returns:
        An unfitted sklearn Pipeline ready for training.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, FEATURE_COLS)],
        remainder="drop",
    )

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators     = 200,
            max_depth        = 12,
            min_samples_leaf = 10,  # prevents overfitting on small leaf nodes
            class_weight     = "balanced",
            random_state     = RANDOM_SEED,
            n_jobs           = -1,
        )),
    ])


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _print_confusion_matrix(y_true, y_pred) -> None:
    """Display a formatted confusion matrix with labeled cells."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(
        f"\n  Confusion Matrix:\n"
        f"  {'':20} {'Predicted: Safe':>16} {'Predicted: Dropout':>18}\n"
        f"  {'Actual: Safe':20} {'TN =':>8} {tn:<8} {'FP =':>8} {fp:<8}\n"
        f"  {'Actual: Dropout':20} {'FN =':>8} {fn:<8} {'TP =':>8} {tp:<8}\n"
        f"\n"
        f"  FP = False alarm (called a safe student 'at risk')\n"
        f"  FN = Missed detection (a dropout student not flagged) <-- minimize this"
    )


def _print_feature_importances(pipeline: Pipeline) -> None:
    """Print the relative importance of each input feature learned by the forest."""
    importances = pipeline.named_steps["classifier"].feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    print("\n  Feature Importances (higher = more influential):")
    for rank, i in enumerate(sorted_idx, 1):
        bar = "#" * int(importances[i] * 50)
        print(f"    {rank}. {FEATURE_COLS[i]:<30}: {importances[i]:.4f}  {bar}")


# ---------------------------------------------------------------------------
# Simulation predictions
# ---------------------------------------------------------------------------

def _run_simulations(pipeline: Pipeline) -> None:
    """
    Run three hand-crafted scenario predictions to sanity-check the trained model.

    Kịch bản kiểm tra nhanh sau khi train để xác nhận mô hình hoạt động đúng.
    """
    scenarios = [
        {
            "label": "Model student — attends regularly, completes all homework",
            "data" : {"Excused_Absences": 0, "Unexcused_Absences": 0,
                      "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.05},
            "expected": "LOW_RISK",
        },
        {
            "label": "Average student — 1 excused, 1 unexcused, some missed hw",
            "data" : {"Excused_Absences": 1, "Unexcused_Absences": 1,
                      "Consecutive_Absences": 1, "Missed_Homework_Ratio": 0.20},
            "expected": "MEDIUM_RISK",
        },
        {
            "label": "Red alert — 3 consecutive unexcused absences, high hw skip",
            "data" : {"Excused_Absences": 0, "Unexcused_Absences": 4,
                      "Consecutive_Absences": 3, "Missed_Homework_Ratio": 0.70},
            "expected": "HIGH_RISK",
        },
    ]

    print(f"\n{'=' * 60}")
    print("  SIMULATION PREDICTIONS")
    print(f"{'=' * 60}")

    for i, sc in enumerate(scenarios, 1):
        df_input = pd.DataFrame([sc["data"]])
        prob     = pipeline.predict_proba(df_input)[0][1] * 100
        level    = "HIGH_RISK" if prob >= 70 else ("MEDIUM_RISK" if prob >= 40 else "LOW_RISK")

        print(f"\n  [{i}] {sc['label']}")
        print(f"        Input    : {sc['data']}")
        print(f"        Result   : {level} ({prob:.1f}%)")
        print(f"        Expected : {sc['expected']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{'=' * 60}")
    print("  DROPOUT RISK MODEL — TRAINING PIPELINE")
    print(f"{'=' * 60}\n")

    # Step 1: Load data
    print("[1/6] Loading dataset...")
    X, y = load_data(DATASET_PATH)

    # Step 2: Train / test split (stratified to preserve class ratio)
    print("\n[2/6] Splitting data (80% train / 20% test, stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Step 3: Build pipeline
    print("\n[3/6] Building pipeline...")
    pipeline = build_pipeline()
    print("  Pipeline: SimpleImputer(median) -> StandardScaler -> "
          "RandomForestClassifier(n=200)")

    # Step 4: Train
    print("\n[4/6] Training model...")
    pipeline.fit(X_train, y_train)
    print("  Training complete.")

    # Step 5: Cross-validation (5-fold, F1 metric)
    print("\n[5/6] Running 5-fold cross-validation (scoring='f1')...")
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1", n_jobs=-1)
    print(f"  Fold F1 scores : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean F1        : {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")

    # Step 6: Evaluate on holdout test set
    print("\n[6/6] Evaluating on test set...")
    y_pred  = pipeline.predict(X_test)
    y_prob  = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_prob)

    print(f"\n  Accuracy : {accuracy * 100:.2f}%")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Safe (0)", "Dropout (1)"],
    ))

    _print_confusion_matrix(y_test, y_pred)
    _print_feature_importances(pipeline)

    # Export trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n  Model saved to: {MODEL_PATH}")

    # Simulation predictions
    _run_simulations(pipeline)

    print(f"\n{'=' * 60}")
    print("  Training pipeline completed successfully.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
