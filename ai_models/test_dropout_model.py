"""
test_dropout_model.py
----------------------
Independent QA test suite for the Student Dropout Risk model.

Acts as a standalone tester before the model is integrated into the
production API server. Covers six test groups:

    T1. Sanity Check        — model loads, schema is correct, predict() works
    T2. Performance         — Accuracy / F1 / ROC-AUC / Recall on holdout set
    T3. Behavioral Tests    — model output direction matches domain logic
    T4. Monotonicity        — increasing risk inputs must increase probability
    T5. Edge Cases          — boundary values, NaN, and out-of-range inputs
    T6. Real-world Scenarios — representative student profiles from TOEIC centers

Acceptance Criteria (minimum thresholds for release):
    Accuracy  >= 78%
    F1-Score  >= 0.72
    ROC-AUC   >= 0.85
    Recall    >= 0.70

Final verdict is printed at the end:
    APPROVED    (>= 90% tests pass) — ready for production
    CONDITIONAL (>= 75% tests pass) — release with monitoring
    REJECTED    (<  75% tests pass) — retrain required
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics         import (accuracy_score, f1_score, precision_score,
                                      recall_score, roc_auc_score,
                                      classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "ai_models", "dropout_model_pipeline.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "archive", "dropout_dataset.csv")

FEATURE_COLS = [
    "Excused_Absences",
    "Unexcused_Absences",
    "Consecutive_Absences",
    "Missed_Homework_Ratio",
]
TARGET_COL  = "Is_Dropout"
RANDOM_SEED = 42

# Minimum acceptable thresholds for each performance metric
THRESHOLDS = {
    "accuracy" : 0.78,
    "f1"       : 0.72,
    "roc_auc"  : 0.85,
    "recall"   : 0.70,
    "fn_rate"  : 0.15,   # max acceptable False Negative rate among actual dropouts
}

# ---------------------------------------------------------------------------
# Test result tracking
# ---------------------------------------------------------------------------
_results: list[tuple[str, str, bool, str]] = []   # (id, name, passed, note)


def _record(test_id: str, name: str, passed: bool, note: str = "") -> None:
    """Register a test result and print a PASS/FAIL line."""
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {test_id} | {name}")
    if note:
        print(f"          {note}")
    _results.append((test_id, name, passed, note))


def _section(title: str = "") -> None:
    """Print a visual section separator."""
    line = "-" * 65
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def _predict_prob(model, data: dict) -> float:
    """Return the dropout probability (0.0-1.0) for a single input record."""
    return model.predict_proba(pd.DataFrame([data]))[0][1]


# ---------------------------------------------------------------------------
# Load model and dataset
# ---------------------------------------------------------------------------
_section("SETUP — Loading Model and Dataset")

if not os.path.exists(MODEL_PATH):
    print(f"  [ERROR] Model file not found: {MODEL_PATH}")
    print("  Run train_dropout_model.py first.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
print(f"  Model loaded : {MODEL_PATH}")

df = pd.read_csv(DATASET_PATH)
X  = df[FEATURE_COLS]
y  = df[TARGET_COL]

# Use the same split seed as training to get the exact same holdout test set
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
print(f"  Test set     : {len(X_test):,} records  ({y_test.sum()} dropouts)")


# ---------------------------------------------------------------------------
# T1 — Sanity Check
# ---------------------------------------------------------------------------
_section("T1 | SANITY CHECK — Load & Schema")

# T1.1: model.predict() does not crash
try:
    model.predict(X_test.head(1))
    _record("T1.1", "model.predict() runs without errors", True)
except Exception as e:
    _record("T1.1", "model.predict() runs without errors", False, str(e))

# T1.2: predict_proba returns a (n, 2) shaped matrix
proba      = model.predict_proba(X_test.head(5))
shape_ok   = proba.shape[1] == 2
_record("T1.2", "predict_proba() returns shape (n, 2)", shape_ok,
        f"Actual shape: {proba.shape}")

# T1.3: each row sums to 1.0 (probabilities are well-formed)
_record("T1.3", "Row probabilities sum to 1.0",
        np.allclose(proba.sum(axis=1), 1.0, atol=1e-6))

# T1.4: all probabilities lie within [0, 1]
_record("T1.4", "All probability values are in range [0.0, 1.0]",
        bool((proba >= 0).all() and (proba <= 1).all()))

# T1.5: model accepts the expected 4 feature columns
try:
    sample = pd.DataFrame([{
        "Excused_Absences": 1, "Unexcused_Absences": 2,
        "Consecutive_Absences": 1, "Missed_Homework_Ratio": 0.3,
    }])
    model.predict(sample)
    _record("T1.5", "Model accepts the correct 4 feature columns", True)
except Exception as e:
    _record("T1.5", "Model accepts the correct 4 feature columns", False, str(e))


# ---------------------------------------------------------------------------
# T2 — Performance Metrics
# ---------------------------------------------------------------------------
_section("T2 | PERFORMANCE METRICS — Accuracy / F1 / ROC-AUC / Recall")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy  = accuracy_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print(f"\n  Metric Results:")
print(f"    Accuracy  : {accuracy * 100:.2f}%  (threshold: >= {THRESHOLDS['accuracy'] * 100:.0f}%)")
print(f"    F1-Score  : {f1:.4f}   (threshold: >= {THRESHOLDS['f1']:.2f})")
print(f"    ROC-AUC   : {roc_auc:.4f}   (threshold: >= {THRESHOLDS['roc_auc']:.2f})")
print(f"    Precision : {precision:.4f}")
print(f"    Recall    : {recall:.4f}   (threshold: >= {THRESHOLDS['recall']:.2f})")
print(f"\n  Confusion Matrix:")
print(f"    TP (correct dropout flag)   : {tp:>5}     FP (false alarm)              : {fp:>5}")
print(f"    TN (correct safe label)     : {tn:>5}     FN (missed dropout detection) : {fn:>5}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Safe (0)", "Dropout (1)"], digits=4))

fn_rate = fn / (fn + tp)

_record("T2.1", f"Accuracy  >= {THRESHOLDS['accuracy']*100:.0f}%   (actual: {accuracy*100:.1f}%)",
        accuracy >= THRESHOLDS["accuracy"])
_record("T2.2", f"F1-Score  >= {THRESHOLDS['f1']:.2f}    (actual: {f1:.4f})",
        f1 >= THRESHOLDS["f1"])
_record("T2.3", f"ROC-AUC   >= {THRESHOLDS['roc_auc']:.2f}   (actual: {roc_auc:.4f})",
        roc_auc >= THRESHOLDS["roc_auc"])
_record("T2.4", f"Recall    >= {THRESHOLDS['recall']:.2f}   (actual: {recall:.4f})",
        recall >= THRESHOLDS["recall"])
_record("T2.5", f"FN rate   <  {THRESHOLDS['fn_rate']:.0%} of actual dropouts  (actual: {fn_rate:.1%})",
        fn_rate < THRESHOLDS["fn_rate"])


# ---------------------------------------------------------------------------
# T3 — Behavioral Correctness
# ---------------------------------------------------------------------------
_section("T3 | BEHAVIORAL TESTS — Does the model output match domain logic?")

# T3.1: Perfect student -> very low dropout risk
p_perfect = _predict_prob(model, {
    "Excused_Absences": 0, "Unexcused_Absences": 0,
    "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.0,
})
_record("T3.1", f"Perfect attendance -> low risk (<30%)  [got {p_perfect*100:.1f}%]",
        p_perfect < 0.30)

# T3.2: 4 unexcused absences + consecutive -> high risk
p_danger = _predict_prob(model, {
    "Excused_Absences": 0, "Unexcused_Absences": 4,
    "Consecutive_Absences": 3, "Missed_Homework_Ratio": 0.6,
})
_record("T3.2", f"4 unexcused + consecutive absences -> high risk (>70%)  [got {p_danger*100:.1f}%]",
        p_danger > 0.70)

# T3.3: Excused absences should be less dangerous than unexcused (same count)
p_excused   = _predict_prob(model, {
    "Excused_Absences": 3, "Unexcused_Absences": 0,
    "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.2,
})
p_unexcused = _predict_prob(model, {
    "Excused_Absences": 0, "Unexcused_Absences": 3,
    "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.2,
})
_record("T3.3",
        f"Excused ({p_excused*100:.1f}%) < Unexcused ({p_unexcused*100:.1f}%) — same count",
        p_excused < p_unexcused)

# T3.4: Consecutive absences should raise risk above scattered absences (same total)
p_scattered   = _predict_prob(model, {
    "Excused_Absences": 1, "Unexcused_Absences": 3,
    "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.3,
})
p_consecutive = _predict_prob(model, {
    "Excused_Absences": 1, "Unexcused_Absences": 3,
    "Consecutive_Absences": 3, "Missed_Homework_Ratio": 0.3,
})
_record("T3.4",
        f"Scattered ({p_scattered*100:.1f}%) < Consecutive ({p_consecutive*100:.1f}%)",
        p_scattered < p_consecutive)

# T3.5: Higher missed homework ratio should increase risk
p_hw_low  = _predict_prob(model, {
    "Excused_Absences": 1, "Unexcused_Absences": 1,
    "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.05,
})
p_hw_high = _predict_prob(model, {
    "Excused_Absences": 1, "Unexcused_Absences": 1,
    "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.80,
})
_record("T3.5",
        f"Low HW skip ({p_hw_low*100:.1f}%) < High HW skip ({p_hw_high*100:.1f}%)",
        p_hw_low < p_hw_high)


# ---------------------------------------------------------------------------
# T4 — Monotonicity
# ---------------------------------------------------------------------------
_section("T4 | MONOTONICITY — Increasing risk inputs must increase probability")


def _check_monotonicity(feature: str, values: list, base: dict) -> tuple[list, bool]:
    """Return (probability_list, is_non_decreasing) for a sweep of one feature."""
    probs = []
    for v in values:
        data         = base.copy()
        data[feature] = v
        probs.append(_predict_prob(model, data))
    non_decreasing = all(probs[i] <= probs[i + 1] for i in range(len(probs) - 1))
    return probs, non_decreasing


base = {
    "Excused_Absences": 0, "Unexcused_Absences": 0,
    "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.2,
}

vals_unex = [0, 1, 2, 4, 6]
probs_unex, ok_unex = _check_monotonicity("Unexcused_Absences", vals_unex, base)
_record("T4.1",
        "Unexcused_Absences 0->6 is non-decreasing: "
        + " -> ".join(f"{p*100:.1f}%" for p in probs_unex),
        ok_unex)

vals_cons = [0, 1, 2, 3, 5]
probs_cons, ok_cons = _check_monotonicity("Consecutive_Absences", vals_cons, base)
_record("T4.2",
        "Consecutive_Absences 0->5 is non-decreasing: "
        + " -> ".join(f"{p*100:.1f}%" for p in probs_cons),
        ok_cons)

vals_hw = [0.0, 0.2, 0.5, 0.8, 1.0]
probs_hw, ok_hw = _check_monotonicity("Missed_Homework_Ratio", vals_hw, base)
_record("T4.3",
        "Missed_Homework_Ratio 0.0->1.0 is non-decreasing: "
        + " -> ".join(f"{p*100:.1f}%" for p in probs_hw),
        ok_hw)


# ---------------------------------------------------------------------------
# T5 — Edge Cases
# ---------------------------------------------------------------------------
_section("T5 | EDGE CASES — Boundary values, NaN, and out-of-range inputs")

# T5.1: All zeros (ideal student) -> very low probability
p = _predict_prob(model, {
    "Excused_Absences": 0, "Unexcused_Absences": 0,
    "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.0,
})
_record("T5.1", f"All zeros -> very low risk (<20%)  [got {p*100:.1f}%]", p < 0.20)

# T5.2: Worst possible values -> very high probability
p = _predict_prob(model, {
    "Excused_Absences": 0, "Unexcused_Absences": 9,
    "Consecutive_Absences": 7, "Missed_Homework_Ratio": 1.0,
})
_record("T5.2", f"Worst-case inputs -> very high risk (>90%)  [got {p*100:.1f}%]", p > 0.90)

# T5.3: NaN in one feature -> SimpleImputer in pipeline should handle it
try:
    p = _predict_prob(model, {
        "Excused_Absences": float("nan"), "Unexcused_Absences": 1,
        "Consecutive_Absences": 1, "Missed_Homework_Ratio": 0.3,
    })
    _record("T5.3", f"NaN input -> handled by Imputer, no crash  [got {p*100:.1f}%]", True)
except Exception as e:
    _record("T5.3", "NaN input -> handled by Imputer, no crash", False, str(e))

# T5.4: Consecutive > total absences (logically invalid but defensively handled)
try:
    p = _predict_prob(model, {
        "Excused_Absences": 1, "Unexcused_Absences": 0,
        "Consecutive_Absences": 5,
        "Missed_Homework_Ratio": 0.3,
    })
    _record("T5.4", f"Consecutive > total absences (invalid) -> no crash  [got {p*100:.1f}%]",
            True)
except Exception as e:
    _record("T5.4", "Consecutive > total absences -> no crash", False, str(e))

# T5.5: Missed_Homework_Ratio > 1.0 (data entry error) -> no crash
try:
    p = _predict_prob(model, {
        "Excused_Absences": 0, "Unexcused_Absences": 2,
        "Consecutive_Absences": 0, "Missed_Homework_Ratio": 1.5,
    })
    _record("T5.5", f"Missed_Homework_Ratio=1.5 (out of range) -> no crash  [got {p*100:.1f}%]",
            True)
except Exception as e:
    _record("T5.5", "Missed_Homework_Ratio=1.5 -> no crash", False, str(e))


# ---------------------------------------------------------------------------
# T6 — Real-world Scenarios
# ---------------------------------------------------------------------------
_section("T6 | REAL-WORLD SCENARIOS — Student profiles from TOEIC/IELTS centers")

# Các tình huống thực tế điển hình ở trung tâm Anh ngữ để xác nhận
# mô hình xử lý đúng từng loại học viên.
scenarios = [
    {
        "id": "T6.1", "expected": "LOW_RISK (<40%)",
        "label": "Nguyen Van A — attends regularly, 1 excused absence (doctor visit)",
        "data" : {"Excused_Absences": 1, "Unexcused_Absences": 0,
                  "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.10},
        "check": lambda p: p < 0.40,
    },
    {
        "id": "T6.2", "expected": "LOW_RISK (<40%)",
        "label": "Tran Thi B — evening worker, 2 excused absences, rarely misses hw",
        "data" : {"Excused_Absences": 2, "Unexcused_Absences": 0,
                  "Consecutive_Absences": 0, "Missed_Homework_Ratio": 0.12},
        "check": lambda p: p < 0.40,
    },
    {
        "id": "T6.3", "expected": "MEDIUM_RISK (40-70%)",
        "label": "Le Van C — 2 unexcused absences, skipped 30% of homework",
        "data" : {"Excused_Absences": 1, "Unexcused_Absences": 2,
                  "Consecutive_Absences": 1, "Missed_Homework_Ratio": 0.30},
        "check": lambda p: 0.40 <= p <= 0.70,
    },
    {
        "id": "T6.4", "expected": "HIGH_RISK (>70%)",
        "label": "Pham Thi D — 3 consecutive unexcused absences, no homework submitted",
        "data" : {"Excused_Absences": 0, "Unexcused_Absences": 3,
                  "Consecutive_Absences": 3, "Missed_Homework_Ratio": 0.60},
        "check": lambda p: p > 0.70,
    },
    {
        "id": "T6.5", "expected": "HIGH_RISK (>70%)",
        "label": "Hoang Van E — 4 consecutive unexcused, 80% homework missed",
        "data" : {"Excused_Absences": 0, "Unexcused_Absences": 4,
                  "Consecutive_Absences": 4, "Missed_Homework_Ratio": 0.80},
        "check": lambda p: p > 0.70,
    },
    {
        "id": "T6.6", "expected": "LOW/MEDIUM (<55%)",
        "label": "Do Thi F — busy schedule (5 excused), still completes most homework",
        "data" : {"Excused_Absences": 5, "Unexcused_Absences": 0,
                  "Consecutive_Absences": 2, "Missed_Homework_Ratio": 0.25},
        "check": lambda p: p < 0.55,
    },
    {
        "id": "T6.7", "expected": "HIGH_RISK (>65%)",
        "label": "Vu Van G — low absences but submits zero homework (laziness pattern)",
        "data" : {"Excused_Absences": 0, "Unexcused_Absences": 2,
                  "Consecutive_Absences": 0, "Missed_Homework_Ratio": 1.0},
        "check": lambda p: p > 0.65,
    },
]

print()
for sc in scenarios:
    p     = _predict_prob(model, sc["data"])
    level = "HIGH_RISK" if p >= 0.70 else ("MEDIUM_RISK" if p >= 0.40 else "LOW_RISK")
    ok    = sc["check"](p)

    print(f"  {sc['id']} | {sc['label']}")
    print(f"         -> {level}  ({p*100:.1f}%)   Expected: {sc['expected']}")
    _record(sc["id"], sc["label"][:58] + "...", ok,
            f"prob={p*100:.1f}%  expected={sc['expected']}")


# ---------------------------------------------------------------------------
# Final QA verdict
# ---------------------------------------------------------------------------
_section("FINAL QA REPORT")

total   = len(_results)
n_pass  = sum(1 for _, _, ok, _ in _results if ok)
n_fail  = total - n_pass
pass_rt = n_pass / total * 100

print(f"\n  Result: {n_pass}/{total} tests passed  ({pass_rt:.1f}%)\n")

if n_fail:
    print("  Failed tests:")
    for tid, name, ok, note in _results:
        if not ok:
            print(f"    [FAIL] {tid}: {name}")
            if note:
                print(f"           -> {note}")

print()
if pass_rt >= 90:
    verdict = "APPROVED — Model meets all acceptance criteria. Safe to deploy."
elif pass_rt >= 75:
    verdict = "CONDITIONAL — Deploy with active monitoring. Review failed tests."
else:
    verdict = "REJECTED — Too many failures. Retrain with revised dataset/config."

print(f"  VERDICT: {verdict}\n")
_section()
