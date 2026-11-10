"""Build aligned public/CRM features and select a calibrated RF on train-only CV.

Run: python ai_models/train_real_lead_scoring_rf.py
The source is approximate; source-domain metrics do not establish CRM accuracy.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
from lead_contract import (CORE_FEATURES, FEATURES, MODEL_VERSION,
                           engineer_lead_features, engineer_source_features)
from staged_lead_scoring import StagedLeadModel

BASE_DIR = CURRENT_DIR.parent
RAW_DATA_PATH = BASE_DIR / "dataset/archive/Lead Scoring.csv"
REAL_DATASET_PATH = BASE_DIR / "dataset/archive/lead_scoring_real_dataset.csv"
OUTPUT_MODEL_PATH = CURRENT_DIR / "lead_scoring_v2_rf_pipeline.pkl"
RESULT_JSON_PATH = CURRENT_DIR / "lead_scoring_real_rf_results.json"
REPORT_PATH = CURRENT_DIR / "lead_scoring_report.md"
TARGET = "Converted"
CANDIDATES = {
    "intake": CORE_FEATURES,
    "with_engagement": CORE_FEATURES + ["Engagement_Group"],
}


def map_and_clean_real_data(raw_path):
    raw = pd.read_csv(raw_path)
    required = {"Lead Source", "What is your current occupation", TARGET,
                "What matters most to you in choosing a course",
                "Last Activity", "Last Notable Activity"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")
    raw = raw.dropna(subset=[TARGET]).copy()
    target = pd.to_numeric(raw[TARGET], errors="raise")
    if not target.isin([0, 1]).all():
        raise ValueError("Converted must contain binary 0/1 labels")
    for identifier in ["Prospect ID", "Lead Number"]:
        if identifier in raw and (raw[identifier].isna().any() or raw[identifier].duplicated().any()):
            raise ValueError(f"Missing or duplicate {identifier}; resolve identities before splitting")
    mapped = engineer_source_features(raw)
    mapped[TARGET] = target.astype(int)
    return mapped.reset_index(drop=True)


def build_rf_pipeline(features=None, n_estimators=250):
    features = list(FEATURES if features is None else features)
    # Fit the encoder separately within EACH inner calibration training fold.
    estimator = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), features),
        ])),
        ("forest", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=14, min_samples_split=8,
            min_samples_leaf=4, class_weight="balanced", random_state=42, n_jobs=1,
        )),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_lead_features, validate=False)),
        ("classifier", CalibratedClassifierCV(
            estimator, method="isotonic",
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=1,
        )),
    ])


def evaluate_model(pipeline, X_test, y_test):
    return evaluate_probabilities(y_test, pipeline.predict_proba(X_test)[:, 1])


def evaluate_probabilities(y, probabilities, threshold=0.5):
    predicted = probabilities >= threshold
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "f1_score": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def select_candidate(results, minimum_gain=0.005):
    """Prefer fewer proxies unless the train-only mean AUC improves by 0.005."""
    selected = next(iter(results))
    for name in list(results)[1:]:
        if results[name]["mean"]["roc_auc"] > results[selected]["mean"]["roc_auc"] + minimum_gain:
            selected = name
    return selected


def write_markdown(report):
    metadata = report["metadata"]
    metrics = report["real_model_test_metrics"]
    prior = report["same_test_prior_baseline_metrics"]
    lines = [
        "# Lead conversion: aligned feature evaluation", "",
        f"Model: `{MODEL_VERSION}`. Source: X Education, {report['dataset_real_records']:,} records.", "",
        "Public source and CRM records are processed into a shared, coarser feature vocabulary.",
        "No target-derived fallback, random assignment, or full-dataset fitted statistic is used.", "",
        "| Feature | Public source | CRM |", "|---|---|---|",
        "| Source_Group | Lead Source grouped into search/social/referral/website/other/unknown | Same grouping of lead_source |",
        "| Occupation_Group | Student/working/unemployed/other/unknown | Both student year groups merge into student |",
        "| Goal_Group | Explicit career motivation; other source answers unknown | career/company -> career; exam/study abroad -> academic; hobby -> personal |",
        "| Engagement_Group | Recorded email/phone/chat response -> active; unreachable/bounced -> unreachable; otherwise unknown | Explicit contact result; used only after actual contact |", "",
        "Initial model uses Source_Group, Occupation_Group and Goal_Group only.",
        "After actual contact, the response model may add Engagement_Group.",
        "A contact without a recorded result retains the intake model until response evidence exists.",
        "TotalVisits and call counts are not features of either public-data model.", "",
        "Course choice and lead age are excluded: specialization is not a course, and website seconds are not lead age.",
        "Tags are excluded; SMS Sent and Converted to Lead do not establish customer response.",
        "Missing occupation is unknown, not unemployed. No student year is inferred.", "",
        "## Model selection", "",
        "Split 80/20 with stratification and seed 42 before model selection.",
        "All candidates use the same five CV folds within the 80% training partition.",
        "Every calibration fold fits its own encoder and forest.",
        "Prefer fewer proxies unless an added candidate improves mean CV ROC-AUC by more than 0.005.",
        "The test set is not used to select features or tune thresholds.", "",
        "| Candidate | Train CV AUC | Train CV F1 |", "|---|---:|---:|",
    ]
    for name, result in report["candidate_cv"].items():
        lines.append(f"| {name} | {result['mean']['roc_auc']:.4f} | {result['mean']['f1']:.4f} |")
    lines += ["", f"Selected: `{metadata['selected_candidate']}`.",
              f"Stage feature sets: `{metadata['stage_features']}`.", "",
              "## Stage evaluation on the same public held-out partition", "",
              "These are source-domain comparisons, not evidence of performance on new CRM leads.",
              "Source data has no contact chronology; routed evaluation uses recorded response availability as a proxy.", "",
              "| Model | Accuracy | ROC-AUC | PR-AUC | Brier |", "|---|---:|---:|---:|---:|",
              *[f"| {stage} | {values['accuracy']:.4f} | {values['roc_auc']:.4f} | {values['pr_auc']:.4f} | {values['brier_score']:.4f} |"
                for stage, values in report['stage_test_metrics'].items()], "",
              "## Held-out test", "",
              f"Train: {metadata['split']['train_records']:,}; test: {metadata['split']['test_records']:,}.",
              "Prior baseline uses exactly the same train/test split.", "",
              "| Metric | Selected RF | Prior baseline |", "|---|---:|---:|",
    ]
    for name in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc", "brier_score"]:
        lines.append(f"| {name} | {metrics[name]:.4f} | {prior[name]:.4f} |")
    lines += ["", f"Confusion matrix [TN, FP; FN, TP]: `{metrics['confusion_matrix']}`.", "",
              "HOT >= 0.80; WARM >= 0.50; binary conversion prediction >= 0.50.",
              "Threshold analysis is descriptive; thresholds are not tuned on test outcomes.", "",
              "## Limits", "", *[f"- {v}" for v in report["limitations"]], "",
              "## Next data collection", "",
              "Export lead ID, scoring-time snapshots, real features, conversion timestamp and an explicit outcome horizon.",
              "Keep snapshots of one lead in the same partition and use later dates for final temporal testing.",
              "Incomplete outcome windows must not be labeled as negative.", "",
              "The artifact bundles two trained pipelines; stage routing and SHAP use the same selected pipeline.", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    data = map_and_clean_real_data(RAW_DATA_PATH)
    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"Train={len(X_train)}; held-out test={len(X_test)}", flush=True)
    scores = ["accuracy", "f1", "roc_auc", "precision", "recall"]
    candidates = {}
    for name, features in CANDIDATES.items():
        print(f"Train-only CV: {name}, {features}", flush=True)
        results = cross_validate(
            build_rf_pipeline(features), X_train, y_train,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring=scores, n_jobs=1, error_score="raise",
        )
        candidates[name] = {
            "features": features,
            "mean": {s: float(results[f"test_{s}"].mean()) for s in scores},
            "std": {s: float(results[f"test_{s}"].std()) for s in scores},
            "folds": {s: results[f"test_{s}"].tolist() for s in scores},
        }
        print(candidates[name]["mean"], flush=True)
    selected = select_candidate(candidates)
    active = CANDIDATES[selected]
    print(f"Selected by training CV: {selected}", flush=True)
    intake = build_rf_pipeline(CORE_FEATURES).fit(X_train, y_train)
    engaged = build_rf_pipeline(active).fit(X_train, y_train)
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    metadata = {
        "model_version": MODEL_VERSION, "selected_candidate": selected,
        "active_features": active, "engineered_features": FEATURES,
        "observed_categories": {name: sorted(X_train[name].unique().tolist()) for name in active},
        "proxy_features": [f for f in active if f == "Engagement_Group"],
        "stage_features": {"initial": CORE_FEATURES, "post_contact": active},
        "source": "X Education public dataset", "crm_validated": False,
        "raw_data_sha256": hashlib.sha256(RAW_DATA_PATH.read_bytes()).hexdigest(),
        "split": {"random_state": 42, "test_size": 0.2, "stratified": True,
                  "train_records": len(X_train), "test_records": len(X_test)},
    }
    intake.lead_metadata_ = {**metadata, "active_features": CORE_FEATURES, "training_stage": "initial", "proxy_features": []}
    engaged.lead_metadata_ = {**metadata, "training_stage": "post_contact"}
    pipeline = StagedLeadModel(intake, engaged, metadata)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_probabilities(y_test, probabilities)
    pipeline.lead_metadata_ = metadata
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_real_records": len(data), "metadata": metadata,
        "candidate_cv": candidates, "cv_summary": candidates[selected]["mean"],
        "real_model_test_metrics": metrics,
        "stage_test_metrics": {"initial": evaluate_model(intake, X_test, y_test),
                               "post_contact": evaluate_model(engaged, X_test, y_test),
                               "routed_response_proxy": metrics},
        "same_test_prior_baseline_metrics": evaluate_model(baseline, X_test, y_test),
        "threshold_analysis": {str(t): evaluate_probabilities(y_test, probabilities, t)
                               for t in [0.3, 0.5, 0.8]},
        "limitations": [
            "Public-domain accuracy and probability calibration do not establish CRM performance.",
            "The initial model uses profile information only; call counts and TotalVisits are excluded.",
            "After contact, public response categories remain approximate; missing results retain the profile-only model.",
            "Source motivation contains career/unknown only; academic/personal goals lack source examples.",
            "Last activities lack timestamps; pre-outcome availability remains unverified despite excluding Tags.",
            "No conversion timestamps or outcome horizon exist, so temporal validation is unavailable.",
            "Coarse profiles repeat across splits; unique source lead IDs were checked before splitting.",
        ],
    }
    data.to_csv(REAL_DATASET_PATH, index=False)
    model_tmp = OUTPUT_MODEL_PATH.with_suffix(".tmp")
    joblib.dump(pipeline, model_tmp)
    os.replace(model_tmp, OUTPUT_MODEL_PATH)
    report["model_sha256"] = hashlib.sha256(OUTPUT_MODEL_PATH.read_bytes()).hexdigest()
    report["mapped_data_sha256"] = hashlib.sha256(REAL_DATASET_PATH.read_bytes()).hexdigest()
    RESULT_JSON_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report)
    print("Held-out test:", metrics, flush=True)
    print(f"Saved model and report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
