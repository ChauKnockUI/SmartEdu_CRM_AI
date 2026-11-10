"""Train a real 30-day CRM conversion model from the admin snapshot export.

python ai_models/train_crm_lead_scoring.py --dataset path/to/snapshots.json
Add --activate only after reviewing the report. Public source data cannot be
passed to this trainer, and incomplete outcome windows are never negatives.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from lead_contract import CRM_CATEGORICAL, CRM_NUMERIC, CRM_FEATURES, CRM_MODEL_VERSION, engineer_crm_features
from staged_lead_scoring import has_contact


def load_snapshots(path):
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if document.get("dataset_version") != "lead_snapshot_1":
        raise ValueError("Expected an admin export of lead_snapshot_1; public proxy data is not CRM training data")
    data = pd.DataFrame(document.get("rows", []))
    required = set(CRM_FEATURES + ["lead_key", "snapshot_at", "lead_created_at", "observed_until",
                                  "outcome_window_days", "converted_30d", "eligible_for_training", "outcome_status"])
    if required - set(data.columns):
        raise ValueError(f"No usable CRM snapshots or missing columns: {sorted(required - set(data.columns))}")
    data = data.loc[data.eligible_for_training.eq(True) & data.outcome_status.eq("observed")].copy()
    if data.empty:
        raise ValueError("No completed 30-day observation windows; continue collecting snapshots")
    for name in ["snapshot_at", "lead_created_at", "observed_until"]:
        data[name] = pd.to_datetime(data[name], utc=True, errors="raise")
    if not data.outcome_window_days.eq(30).all() or not data.converted_30d.isin([0, 1]).all():
        raise ValueError("Expected observed binary converted_30d labels with a 30-day horizon")
    if data.lead_key.isna().any() or data.lead_key.eq("").any():
        raise ValueError("Missing stable lead identity")
    if ((data.snapshot_at + pd.Timedelta(days=30) > data.observed_until) |
        (data.snapshot_at < data.lead_created_at)).any():
        raise ValueError("Invalid snapshot time or incomplete observation window")
    if data.groupby(["lead_key", "snapshot_at"]).converted_30d.nunique().gt(1).any():
        raise ValueError("Conflicting labels for the same snapshot")
    # Serving uses this history model after real contact only. Train and
    # validate on that same cohort; initial snapshots remain in the export.
    data = data.loc[data.apply(has_contact, axis=1)].copy()
    if data.empty:
        raise ValueError("No post-contact snapshots; continue recording real calls/emails/SMS/meetings")
    # Cap repeated rescoring to one observation per lead/day without looking at outcomes.
    data = data.sort_values("snapshot_at").assign(snapshot_day=lambda d: d.snapshot_at.dt.floor("D"))
    data = data.drop_duplicates(["lead_key", "snapshot_day"], keep="first").reset_index(drop=True)
    data["converted_30d"] = data.converted_30d.astype(int)
    return data


def temporal_partitions(data):
    first = data.groupby("lead_key").snapshot_at.min().sort_values()
    if len(first) < 30:
        raise ValueError("Need at least 30 distinct leads before attempting temporal partitions")
    n = len(first)
    validation_start, test_start = first.iloc[int(n * .6)], first.iloc[int(n * .8)]
    train_ids = set(first[first < validation_start].index)
    validation_ids = set(first[(first >= validation_start) & (first < test_start)].index)
    test_ids = set(first[first >= test_start].index)
    horizon = data.snapshot_at + pd.Timedelta(days=30)
    train = data[data.lead_key.isin(train_ids) & horizon.le(validation_start)].copy()
    validation = data[data.lead_key.isin(validation_ids) & horizon.le(test_start)].copy()
    test = data[data.lead_key.isin(test_ids) & data.snapshot_at.ge(test_start)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError("Not enough time coverage: collect separated cohorts with full 30-day windows")
    for name, part in [("train", train), ("validation", validation), ("test", test)]:
        minimum = 20 if name == "train" else 5
        counts = part.converted_30d.value_counts()
        if len(counts) != 2 or counts.min() < minimum:
            raise ValueError(f"{name} needs both outcomes with at least {minimum} examples per class")
    return train, validation, test


def build_model(train, depth):
    X, y = train[CRM_FEATURES], train.converted_30d
    splits = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(X, y, train.lead_key))
    for tr, cal in splits:
        if y.iloc[tr].nunique() != 2 or y.iloc[cal].nunique() != 2:
            raise ValueError("Insufficient class coverage across distinct leads for calibration")
    estimator = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CRM_CATEGORICAL),
            ("num", SimpleImputer(strategy="constant", fill_value=-1, keep_empty_features=True), CRM_NUMERIC),
        ])),
        ("forest", RandomForestClassifier(n_estimators=250, max_depth=depth, min_samples_leaf=5,
                                          class_weight="balanced", random_state=42, n_jobs=1)),
    ])
    return Pipeline([
        ("engineer", FunctionTransformer(engineer_crm_features, validate=False)),
        ("classifier", CalibratedClassifierCV(estimator, method="sigmoid", cv=splits, n_jobs=1)),
    ])


def metrics(y, p, lead_keys):
    high = p >= .8
    top = np.argsort(-p, kind="stable")[:max(1, int(np.ceil(len(p) * .2)))]
    return {
        "accuracy": float(np.mean((p >= .5) == y)),
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "precision_50": float(precision_score(y, p >= .5, zero_division=0)),
        "recall_50": float(recall_score(y, p >= .5, zero_division=0)),
        "precision_hot": float(precision_score(y, high, zero_division=0)),
        "hot_count": int(high.sum()), "hot_leads": int(pd.Series(np.asarray(lead_keys)[high]).nunique()),
        "recall_follow_up_30": float(recall_score(y, p >= .3, zero_division=0)),
        "precision_top_20_percent": float(np.asarray(y)[top].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    data = load_snapshots(args.dataset)
    train, validation, test = temporal_partitions(data)
    candidates = []
    for depth in [6, 12]:
        model = build_model(train, depth).fit(train[CRM_FEATURES], train.converted_30d)
        score = metrics(validation.converted_30d, model.predict_proba(validation[CRM_FEATURES])[:, 1], validation.lead_key)
        candidates.append((model, score, depth))
    # Select on validation only. The final test has not been consulted here.
    model, validation_metrics, depth = max(candidates, key=lambda item: item[1]["pr_auc"])
    test_metrics = metrics(test.converted_30d, model.predict_proba(test[CRM_FEATURES])[:, 1], test.lead_key)
    # Predeclared acceptance gate; fixed 80% policy is never tuned using test outcomes.
    approved = all(m["precision_hot"] >= .85 and m["hot_leads"] >= 20
                   for m in [validation_metrics, test_metrics])
    engineered = engineer_crm_features(train[CRM_FEATURES])
    model.lead_metadata_ = {
        "model_version": CRM_MODEL_VERSION, "active_features": CRM_FEATURES,
        "observed_categories": {f: sorted(engineered[f].unique().tolist()) for f in CRM_CATEGORICAL},
        "outcome_window_days": 30, "temporal_validated": True, "crm_validated": True,
        "approved_for_serving": approved, "selected_depth": depth,
        "training_stage": "post_contact",
    }
    report = {
        "model_version": CRM_MODEL_VERSION,
        "training_stage": "post_contact",
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "selection": "Validation PR-AUC; grouped temporal holdout with 30-day label purge",
        "candidate_validation": [{"depth": d, "metrics": s} for _, s, d in candidates],
        "partitions": {name: {"rows": len(part), "leads": int(part.lead_key.nunique()),
                              "start": part.snapshot_at.min().isoformat(), "end": part.snapshot_at.max().isoformat()}
                       for name, part in [("train", train), ("validation", validation), ("test", test)]},
        "validation_metrics": validation_metrics, "test_metrics": test_metrics,
        "approved_for_serving": approved,
        "acceptance": "Precision >= .85 and >= 20 distinct HOT leads on both validation and test",
        "limitations": ["Care priority is a separate CRM rule, not a learned causal treatment effect.",
                        "Repeated snapshots are limited to one per lead/day; calibration groups by lead."],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate = args.output_dir / "lead_scoring_crm_candidate.pkl"
    joblib.dump(model, candidate)
    (args.output_dir / "lead_scoring_crm_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.activate:
        if not approved:
            raise SystemExit("Candidate saved, but activation refused: high-precision acceptance gate not met")
        active = args.output_dir / "lead_scoring_crm_pipeline.pkl"
        temporary = active.with_suffix(".tmp")
        joblib.dump(model, temporary)
        os.replace(temporary, active)
        print(f"Activated artifact: {active}; restart the AI service to load it")


if __name__ == "__main__":
    main()
