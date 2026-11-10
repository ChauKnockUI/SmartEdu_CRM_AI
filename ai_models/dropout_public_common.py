"""Source-specific public baseline features, estimators and evaluation splits."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
    fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SCHEMA_VERSION = "public_center_baseline_1"
SEED = 42
N_FOLDS = 5
TEST_FRAC = 0.20
TARGET_COL = "Is_Dropout"
ACADEMIC_FEATURES = ["Average_Score", "Pass_Rate", "Not_Evaluated_Rate"]
FINANCE_FEATURES = ACADEMIC_FEATURES + ["Has_Debt", "Tuition_Up_To_Date"]
OULAD_FEATURES = ["Average_Score", "Pass_Rate", "Assignment_Missing_Rate",
                  "Assignment_Late_Count", "Score_Trend"]

def make_pipeline(clf, features):
    """Always builds a FRESH preprocessor -> no state shared between models."""
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("sc", StandardScaler()),
        ]), features)
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


def pipeline_signature(pipe):
    """Compare estimator and preprocessing settings without fitted state."""
    params = {key: value for key, value in pipe.get_params(deep=True).items()
              if value is None or isinstance(value, (str, bool, int, float))}
    # NaN != NaN after deserialization; normalize the imputer's missing-value sentinel.
    params = {key: "NaN" if isinstance(value, float) and np.isnan(value) else value
              for key, value in params.items()}
    params["feature_columns"] = list(pipe.named_steps["prep"].transformers[0][2])
    params["steps"] = [name for name, _ in pipe.steps]
    return params


def model_zoo():
    return {
        "Dummy (prior)": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05,
                                                               max_iter=300, l2_regularization=1.0,
                                                               class_weight="balanced", random_state=SEED),
    }


def group_holdout(df):
    strat = df[TARGET_COL].astype(str) + "_" + df["Data_Source"]
    sgkf = StratifiedGroupKFold(n_splits=int(round(1 / TEST_FRAC)), shuffle=True, random_state=SEED)
    tr, te = next(sgkf.split(df, strat, groups=df["student_id"]))
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    if not set(train["student_id"]).isdisjoint(test["student_id"]):
        raise ValueError("Student leakage between train/test")
    return train, test


def best_threshold(y, p, beta=2.0):
    """Recall-oriented (F2) threshold from out-of-fold probabilities."""
    pr, rc, th = precision_recall_curve(y, p)
    pr, rc = pr[:-1], rc[:-1]
    f = (1 + beta ** 2) * pr * rc / np.maximum(beta ** 2 * pr + rc, 1e-12)
    return float(th[int(np.argmax(f))]) if len(th) else 0.5


def evaluate(y, p, thr):
    y = np.asarray(y)
    p = np.asarray(p)
    pred = (p >= thr).astype(int)
    one_class = len(np.unique(y)) < 2
    return {
        "n": int(len(y)),
        "base_rate": round(float(y.mean()), 4),
        "roc_auc": None if one_class else round(float(roc_auc_score(y, p)), 4),
        "pr_auc": None if one_class else round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "threshold": float(thr),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(fbeta_score(y, pred, beta=1, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y, pred, beta=2, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
    }


def make_cv_splits(X, y, groups, sources=None):
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    strata = np.asarray(y).astype(str)
    if sources is not None:
        strata = np.char.add(np.char.add(strata, "_"), np.asarray(sources).astype(str))
    return list(cv.split(X, strata, groups=groups))


def oof_probs(pipe, X, y, groups, sources=None):
    splits = make_cv_splits(X, y, groups, sources)
    return cross_val_predict(pipe, X, y, cv=splits,
                             method="predict_proba", n_jobs=1)[:, 1]
