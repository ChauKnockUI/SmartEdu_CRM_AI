"""
Train v2 dropout risk model using business-aligned CRM/LMS features.
"""

import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "archive", "dropout_dataset_v2.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dropout_model_v2_pipeline.pkl")
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
TARGET_COL = "Is_Dropout"


def build_pipeline() -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([("numeric", numeric, FEATURE_COLS)])
    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=14,
        min_samples_leaf=8,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


def main() -> None:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError("Run generate_dropout_dataset_v2.py first.")

    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipeline = build_pipeline()
    cv = cross_val_score(pipeline, X, y, cv=5, scoring="f1", n_jobs=-1)
    print(f"CV F1: {[round(x, 4) for x in cv]} mean={cv.mean():.4f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(classification_report(y_test, y_pred, target_names=["Safe", "Dropout"], digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1: {f1_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"FN rate: {fn / (fn + tp):.2%}  TP={tp} FP={fp} TN={tn} FN={fn}")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
