"""
process_and_train_dropout_v3.py
--------------------------------
SmartEdu CRM -- Dropout early-warning model (v3), improved version of v2.

Key fixes vs v2
  * Fresh, cloned Pipeline per model (v2 shared one preprocessor -> exported .pkl was corrupted).
  * No source-identifying shortcuts (constant Class_Progress_Ratio, all-NaN finance columns removed
    from the feature set); a "source-only" baseline is reported to prove it.
  * Features come from real signals; unavailable ones are NaN, never fabricated or duplicated.
  * StratifiedGroupKFold everywhere (hold-out AND CV) grouped by student.
  * Model selection on out-of-fold PR-AUC (train only); threshold chosen on OOF (recall-oriented, F2).
  * Per-source test metrics, grouped bootstrap CIs, permutation importance, honest cross-domain test
    on SHARED features only, data-driven report (no hardcoded conclusions).
  * Final model refit on ALL data and saved with metadata.

Milestones (no temporal leakage)
  UCI   : end of semester 1  (only 1st-sem columns; 2nd-sem never touched)
  OULAD : Day 30             (VLE/assessments up to Day 30; students unregistered by Day 30 removed)
Target: 1 = dropout (UCI Dropout / OULAD Withdrawn after Day 30), 0 = Graduate / Pass / Distinction / Fail.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
    fbeta_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

UCI_PATH = os.path.join(BASE_DIR, "dataset", "archive", "dropout_data_uci.csv")
OULAD_DIR = os.path.join(BASE_DIR, "dataset", "archive", "dropout_data_OULAD")
FINAL_DATASET_PATH = os.path.join(BASE_DIR, "dataset", "archive", "dropout_real_final_dataset_v3.csv")
MODEL_OUTPUT_PATH = os.path.join(CURRENT_DIR, "dropout_model_v3_bundle.pkl")
BENCHMARK_JSON_PATH = os.path.join(CURRENT_DIR, "dropout_benchmark_results_v3.json")
BENCHMARK_MD_PATH = os.path.join(CURRENT_DIR, "dropout_benchmark_summary_v3.md")
REPORTS_DIR = os.path.join(CURRENT_DIR, "reports_v3")
os.makedirs(REPORTS_DIR, exist_ok=True)

SEED = 42
CUTOFF_DAY = 30          # OULAD early-warning day
N_FOLDS = 5
TEST_FRAC = 0.20
KEYS = ["code_module", "code_presentation", "id_student"]
TARGET_COL = "Is_Dropout"

# Features actually fed to the model. NaN = not observable in that source.
FEATURE_COLS = [
    "Attendance_Rate",                  # OULAD: active VLE days / days in window      | UCI: NaN
    "Consecutive_Unexcused_Absences",   # OULAD: longest run of inactive days          | UCI: NaN
    "Days_Since_Last_Attended",         # OULAD: days since last VLE activity          | UCI: NaN
    "Log_Total_Activity",               # OULAD: log1p(total clicks)                   | UCI: NaN
    "Assignment_Missing_Rate",          # share of due assessments/units not evaluated | both
    "Assignment_Late_Count",            # OULAD: late submissions                      | UCI: NaN
    "Average_Score",                    # 0-10 scale                                   | both
    "Score_Trend",                      # OULAD: last - first early score (0-10)       | UCI: NaN
    "Pass_Rate",                        # UCI: approved/enrolled; OULAD: share score>=40 | both
    "Has_Overdue_Invoice",              # UCI: debtor / tuition not up to date         | OULAD: NaN
]
# Features with comparable meaning in both sources -> used for honest cross-domain test
SHARED_FEATURES = ["Average_Score", "Assignment_Missing_Rate", "Pass_Rate"]


# =============================================================================
# 1. UCI  (end of semester 1)
# =============================================================================
def process_uci(path: str) -> pd.DataFrame:
    print(f"\n[1/5] UCI (milestone: end of semester 1): {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df[df["Target"].isin(["Dropout", "Graduate"])].copy().reset_index(drop=True)

    num = lambda col: pd.to_numeric(df[col], errors="coerce")
    enrolled = num("Curricular units 1st sem (enrolled)")
    approved = num("Curricular units 1st sem (approved)")
    grade = num("Curricular units 1st sem (grade)")
    without_eval = num("Curricular units 1st sem (without evaluations)")

    # Students with no enrolled unit in sem 1 have no observable behaviour -> excluded (documented)
    keep = enrolled > 0
    print(f"      Dropped {int((~keep).sum())} rows with 0 enrolled units in semester 1")
    df, enrolled, approved, grade, without_eval = (x[keep].reset_index(drop=True) for x in
                                                   (df, enrolled, approved, grade, without_eval))

    out = pd.DataFrame(index=df.index)
    out["Average_Score"] = np.where(approved > 0, np.clip(grade / 2.0, 0, 10), np.nan)  # 0-20 -> 0-10
    out["Pass_Rate"] = np.clip(approved / enrolled, 0, 1)
    out["Assignment_Missing_Rate"] = np.clip(without_eval / enrolled, 0, 1)
    debtor = (num("Debtor") == 1) | (num("Tuition fees up to date") == 0)
    out["Has_Overdue_Invoice"] = debtor.astype(float)
    for col in FEATURE_COLS:
        if col not in out:
            out[col] = np.nan   # genuinely unobservable in UCI

    out[TARGET_COL] = (df["Target"] == "Dropout").astype(int)
    out["id_student"] = "UCI_" + df.index.astype(str)
    out["Data_Source"] = "UCI"
    print(f"      -> {len(out):,} records | dropout rate {out[TARGET_COL].mean():.2%}")
    return out[FEATURE_COLS + [TARGET_COL, "id_student", "Data_Source"]]


# =============================================================================
# 2. OULAD  (Day 30)  -- fully vectorised
# =============================================================================
def process_oulad(oulad_dir: str, cutoff: int = CUTOFF_DAY) -> pd.DataFrame:
    print(f"\n[2/5] OULAD (milestone: Day {cutoff}): {oulad_dir}")
    p = lambda f: os.path.join(oulad_dir, f)
    for f in ["studentInfo.csv", "studentRegistration.csv", "assessments.csv",
              "studentAssessment.csv", "studentVle.csv"]:
        if not os.path.exists(p(f)):
            raise FileNotFoundError(p(f))

    info = pd.read_csv(p("studentInfo.csv"))
    reg = pd.read_csv(p("studentRegistration.csv"))
    enr = info.merge(reg[KEYS + ["date_unregistration"]], on=KEYS, how="inner")
    enr["date_unregistration"] = pd.to_numeric(enr["date_unregistration"], errors="coerce")
    # Only students still registered after the cutoff are scored
    enr = enr[enr["date_unregistration"].isna() | (enr["date_unregistration"] > cutoff)]
    enr[TARGET_COL] = (enr["final_result"] == "Withdrawn").astype(int)
    enr = enr[KEYS + [TARGET_COL]].drop_duplicates(KEYS).reset_index(drop=True)
    print(f"      {len(enr):,} active enrollments at Day {cutoff} | dropout rate {enr[TARGET_COL].mean():.2%}")

    # ---- assessments (known by Day `cutoff`) ----
    asm = pd.read_csv(p("assessments.csv"))[["code_module", "code_presentation", "id_assessment", "date"]]
    asm = asm.rename(columns={"date": "date_due"})
    asm["date_due"] = pd.to_numeric(asm["date_due"], errors="coerce")
    n_due = (asm[asm["date_due"] <= cutoff].groupby(["code_module", "code_presentation"])
             .size().rename("n_due").reset_index())

    sa = pd.read_csv(p("studentAssessment.csv"))
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
    sa = sa.merge(asm, on="id_assessment", how="left")
    sa = sa[sa["date_submitted"] <= cutoff].merge(enr[KEYS], on=KEYS, how="inner")
    sa = sa.sort_values("date_submitted")
    sa["is_late"] = ((sa["date_submitted"] > sa["date_due"]) & sa["date_due"].notna()).astype(int)
    sa["passed"] = np.where(sa["score"].notna(), (sa["score"] >= 40).astype(float), np.nan)

    g = sa.groupby(KEYS)
    ass = pd.DataFrame({
        "Average_Score": g["score"].mean() / 10.0,
        "Pass_Rate": g["passed"].mean(),
        "Assignment_Late_Count": g["is_late"].sum(),
        "_first": g["score"].first(),
        "_last": g["score"].last(),
        "_n_scored": g["score"].count(),
    })
    ass["Score_Trend"] = np.where(ass["_n_scored"] >= 2, (ass["_last"] - ass["_first"]) / 10.0, np.nan)
    ass["_n_due_submitted"] = sa[sa["date_due"] <= cutoff].groupby(KEYS).size()
    ass = ass.reset_index()

    # ---- VLE activity: days 0..cutoff only, aggregated per chunk then combined ----
    print("      Streaming VLE logs...")
    parts = []
    for chunk in pd.read_csv(p("studentVle.csv"),
                             usecols=KEYS + ["date", "sum_click"], chunksize=1_000_000):
        chunk = chunk[(chunk["date"] >= 0) & (chunk["date"] <= cutoff)]
        if not chunk.empty:
            parts.append(chunk.groupby(KEYS + ["date"], as_index=False)["sum_click"].sum())
    vle = pd.concat(parts).groupby(KEYS + ["date"], as_index=False)["sum_click"].sum()
    vle = vle.merge(enr[KEYS], on=KEYS, how="inner").sort_values(KEYS + ["date"])
    prev = vle.groupby(KEYS)["date"].shift(1).fillna(-1)
    vle["gap"] = vle["date"] - prev - 1            # inactive days right before this active day
    gv = vle.groupby(KEYS)
    act = pd.DataFrame({
        "n_active": gv["date"].nunique(),
        "last_active": gv["date"].max(),
        "clicks": gv["sum_click"].sum(),
        "max_gap": gv["gap"].max(),
    }).reset_index()

    # ---- assemble ----
    window = cutoff + 1
    df = enr.merge(act, on=KEYS, how="left").merge(ass, on=KEYS, how="left")
    df = df.merge(n_due, on=["code_module", "code_presentation"], how="left")
    # No VLE rows = genuinely zero activity (observed), not missing
    df["n_active"] = df["n_active"].fillna(0)
    df["clicks"] = df["clicks"].fillna(0)
    df["Attendance_Rate"] = df["n_active"] / window
    df["Consecutive_Unexcused_Absences"] = np.where(
        df["n_active"] > 0, np.maximum(df["max_gap"], cutoff - df["last_active"]), window)
    df["Days_Since_Last_Attended"] = np.where(df["n_active"] > 0, cutoff - df["last_active"], window)
    df["Log_Total_Activity"] = np.log1p(df["clicks"])

    df["Assignment_Late_Count"] = df["Assignment_Late_Count"].fillna(0)
    sub = df["_n_due_submitted"].fillna(0)
    df["Assignment_Missing_Rate"] = np.where(
        df["n_due"] > 0, np.clip(1.0 - sub / df["n_due"].replace(0, np.nan), 0, 1), np.nan)
    df["Has_Overdue_Invoice"] = np.nan      # OULAD has no finance data

    df["id_student"] = df["id_student"].astype(str)
    df["Data_Source"] = "OULAD"
    out = df[FEATURE_COLS + [TARGET_COL, "id_student", "Data_Source"]].reset_index(drop=True)
    print(f"      -> {len(out):,} records | dropout rate {out[TARGET_COL].mean():.2%}")
    return out


# =============================================================================
# 3. Blend
# =============================================================================
def blend(uci: pd.DataFrame, oulad: pd.DataFrame, path: str) -> pd.DataFrame:
    df = pd.concat([uci, oulad], ignore_index=True)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"\n[3/5] Blended {len(df):,} rows (dropout {df[TARGET_COL].mean():.2%}) -> {path}")
    return df


# =============================================================================
# Helpers
# =============================================================================
def make_pipeline(clf, features):
    """Always builds a FRESH preprocessor -> no state shared between models."""
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("sc", StandardScaler()),
        ]), features)
    ])
    return Pipeline([("prep", prep), ("clf", clf)])


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
    tr, te = next(sgkf.split(df, strat, groups=df["id_student"]))
    train, test = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
    assert set(train["id_student"]).isdisjoint(test["id_student"]), "student leakage between train/test!"
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
        "threshold": round(float(thr), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(fbeta_score(y, pred, beta=1, zero_division=0)), 4),
        "f2": round(float(fbeta_score(y, pred, beta=2, zero_division=0)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
    }


def grouped_bootstrap_ci(y, p, groups, metric, n_boot=300):
    """95% CI resampling STUDENTS (not rows)."""
    y, p = np.asarray(y), np.asarray(p)
    codes, uniq = pd.factorize(pd.Series(groups))
    idx_by_g = [np.where(codes == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_g[i] for i in pick])
        if len(np.unique(y[idx])) == 2:
            vals.append(metric(y[idx], p[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def oof_probs(pipe, X, y, groups):
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return cross_val_predict(pipe, X, y, cv=cv, groups=groups,
                             method="predict_proba", n_jobs=-1)[:, 1]


# =============================================================================
# 4. Benchmark + production training
# =============================================================================
def run_benchmark_and_train(df: pd.DataFrame):
    print("\n[4/5] Benchmark (StratifiedGroupKFold by student, OOF model selection)...")
    train, test = group_holdout(df)
    Xtr, ytr, gtr = train[FEATURE_COLS], train[TARGET_COL].values, train["id_student"]
    Xte, yte = test[FEATURE_COLS], test[TARGET_COL].values
    print(f"      Train {len(train):,} rows / {gtr.nunique():,} students | "
          f"Test {len(test):,} rows / {test['id_student'].nunique():,} students")

    results, fitted, test_prob, oof_ap = {}, {}, {}, {}
    for name, clf in model_zoo().items():
        pipe = make_pipeline(clf, FEATURE_COLS)
        oof = oof_probs(pipe, Xtr, ytr, gtr)
        thr = best_threshold(ytr, oof)
        oof_ap[name] = float(average_precision_score(ytr, oof))
        pipe.fit(Xtr, ytr)
        prob = pipe.predict_proba(Xte)[:, 1]
        fitted[name], test_prob[name] = pipe, prob
        results[name] = {
            "oof_pr_auc_train": round(oof_ap[name], 4),
            "test_overall": evaluate(yte, prob, thr),
            "test_by_source": {
                src: evaluate(yte[m], prob[m], thr)
                for src in sorted(test["Data_Source"].unique())
                for m in [(test["Data_Source"] == src).values]
            },
        }

    candidates = {k: v for k, v in oof_ap.items() if not k.startswith("Dummy")}
    best_name = max(candidates, key=candidates.get)
    best_thr = results[best_name]["test_overall"]["threshold"]
    best_prob = test_prob[best_name]
    print(f"      Selected model (best OOF PR-AUC on TRAIN): {best_name}")

    hdr = f"      {'Model':<22}|{'OOF PR':>8}|{'ROC-AUC':>9}|{'PR-AUC':>8}|{'Recall':>8}|{'Prec':>7}|{'F2':>7}"
    print("      " + "-" * (len(hdr) - 6)); print(hdr); print("      " + "-" * (len(hdr) - 6))
    for n, r in results.items():
        t = r["test_overall"]
        print(f"      {n:<22}|{r['oof_pr_auc_train']:>8.4f}|{t['roc_auc']:>9.4f}|{t['pr_auc']:>8.4f}|"
              f"{t['recall']:>8.4f}|{t['precision']:>7.4f}|{t['f2']:>7.4f}")

    ci = {
        "roc_auc_95ci": grouped_bootstrap_ci(yte, best_prob, test["id_student"], roc_auc_score),
        "pr_auc_95ci": grouped_bootstrap_ci(yte, best_prob, test["id_student"], average_precision_score),
    }
    print(f"      Best model 95% CI (student bootstrap): ROC-AUC {ci['roc_auc_95ci']} | PR-AUC {ci['pr_auc_95ci']}")

    # ---- Shortcut sanity check: how well does knowing ONLY the data source predict? ----
    src_rate = train.groupby("Data_Source")[TARGET_COL].mean()
    src_prob = test["Data_Source"].map(src_rate).values
    source_only_auc = round(float(roc_auc_score(yte, src_prob)), 4)
    print(f"      Source-only baseline ROC-AUC: {source_only_auc} "
          f"(blended AUC far above per-source AUC => source shortcut still present)")

    # ---- Cross-domain on SHARED features only ----
    print("\n[5/5] Cross-domain test on SHARED features:", SHARED_FEATURES)
    uci, oul = df[df["Data_Source"] == "UCI"], df[df["Data_Source"] == "OULAD"]
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)

    def transfer(src, dst):
        pipe = make_pipeline(clone(rf), SHARED_FEATURES).fit(src[SHARED_FEATURES], src[TARGET_COL])
        pr = pipe.predict_proba(dst[SHARED_FEATURES])[:, 1]
        return {"roc_auc": round(float(roc_auc_score(dst[TARGET_COL], pr)), 4),
                "pr_auc": round(float(average_precision_score(dst[TARGET_COL], pr)), 4),
                "target_base_rate": round(float(dst[TARGET_COL].mean()), 4)}

    def in_domain(d):
        pr = oof_probs(make_pipeline(clone(rf), SHARED_FEATURES), d[SHARED_FEATURES],
                       d[TARGET_COL].values, d["id_student"])
        return {"roc_auc": round(float(roc_auc_score(d[TARGET_COL], pr)), 4),
                "pr_auc": round(float(average_precision_score(d[TARGET_COL], pr)), 4)}

    cross = {
        "UCI_to_OULAD": transfer(uci, oul),
        "OULAD_to_UCI": transfer(oul, uci),
        "in_domain_reference_UCI": in_domain(uci),
        "in_domain_reference_OULAD": in_domain(oul),
    }
    for k, v in cross.items():
        print(f"      {k:<26}: {v}")

    # ---- Permutation importance on held-out test (raw features, order-safe) ----
    pi = permutation_importance(fitted[best_name], Xte, yte, scoring="average_precision",
                                n_repeats=5, random_state=SEED, n_jobs=-1)
    importance = dict(sorted(zip(FEATURE_COLS, pi.importances_mean.round(4).tolist()),
                             key=lambda kv: -kv[1]))

    # ---- Final production model: refit on ALL data, keep OOF-based threshold ----
    final_pipe = make_pipeline(clone(model_zoo()[best_name]), FEATURE_COLS).fit(df[FEATURE_COLS], df[TARGET_COL])
    bundle = {
        "pipeline": final_pipe,
        "features": FEATURE_COLS,
        "threshold": best_thr,
        "model_name": best_name,
        "version": "v3",
        "milestones": {"UCI": "end of semester 1", "OULAD": f"day {CUTOFF_DAY}"},
        "note": "Trained on proxy features from UCI/OULAD; recalibrate on real SmartEdu data before relying on it.",
    }
    joblib.dump(bundle, MODEL_OUTPUT_PATH)
    print(f"\n[OK] Model bundle saved: {MODEL_OUTPUT_PATH}")

    out = {
        "selected_model": best_name,
        "models": results,
        "selected_model_ci": ci,
        "source_only_baseline_roc_auc": source_only_auc,
        "cross_domain_shared_features": cross,
        "permutation_importance_pr_auc_drop": importance,
        "test_confusion_matrix": confusion_matrix(yte, (best_prob >= best_thr).astype(int)).tolist(),
    }
    with open(BENCHMARK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    write_report(out, BENCHMARK_MD_PATH)
    make_charts(yte, test_prob, best_name, best_thr, importance)


# =============================================================================
# Charts (one dashboard + individual ROC/PR)
# =============================================================================
def make_charts(y, probs, best_name, thr, importance):
    print("\n[*] Generating charts...")
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    fig, ax = plt.subplots(2, 2, figsize=(14, 11), dpi=200)

    for name, p in probs.items():
        if len(np.unique(p)) > 1:
            fpr, tpr, _ = roc_curve(y, p)
            ax[0, 0].plot(fpr, tpr, lw=2.2 if name == best_name else 1.3,
                          label=f"{name} ({roc_auc_score(y, p):.3f})")
            pr, rc, _ = precision_recall_curve(y, p)
            ax[0, 1].plot(rc, pr, lw=2.2 if name == best_name else 1.3,
                          label=f"{name} ({average_precision_score(y, p):.3f})")
    ax[0, 0].plot([0, 1], [0, 1], "k:")
    ax[0, 0].set(title="(A) ROC", xlabel="FPR", ylabel="TPR"); ax[0, 0].legend(loc="lower right", fontsize=8)
    ax[0, 1].axhline(np.mean(y), color="gray", ls=":", label=f"Base rate {np.mean(y):.2f}")
    ax[0, 1].set(title="(B) Precision-Recall", xlabel="Recall", ylabel="Precision")
    ax[0, 1].legend(loc="upper right", fontsize=8)

    cm = confusion_matrix(y, (probs[best_name] >= thr).astype(int))
    ax[1, 0].imshow(cm, cmap="Reds"); ax[1, 0].grid(False)
    ax[1, 0].set(title=f"(C) Confusion matrix - {best_name} @thr={thr:.2f}",
                 xticks=[0, 1], yticks=[0, 1], xticklabels=["Safe", "Dropout"],
                 yticklabels=["Safe", "Dropout"], xlabel="Predicted", ylabel="Actual")
    for i in range(2):
        for j in range(2):
            ax[1, 0].text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                          color="white" if cm[i, j] > cm.max() / 2 else "black")

    names, vals = list(importance.keys())[::-1], list(importance.values())[::-1]
    ax[1, 1].barh(names, vals, color="#2ca02c", alpha=0.85)
    ax[1, 1].set(title="(D) Permutation importance (drop in PR-AUC, test)", xlabel="Mean drop")

    plt.suptitle("Dropout risk v3 - evaluation dashboard", fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(REPORTS_DIR, "dropout_v3_dashboard.png")
    plt.savefig(path); plt.close()
    print(f"  [OK] {path}")


# =============================================================================
# Data-driven report (no hardcoded conclusions)
# =============================================================================
def write_report(o: dict, path: str):
    f = lambda v: "n/a" if v is None else f"{v:.4f}"
    rows = "\n".join(
        f"| {n} | {f(r['oof_pr_auc_train'])} | {f(r['test_overall']['roc_auc'])} | {f(r['test_overall']['pr_auc'])} | "
        f"{f(r['test_overall']['recall'])} | {f(r['test_overall']['precision'])} | {f(r['test_overall']['f2'])} | "
        f"{f(r['test_overall']['brier'])} |"
        for n, r in o["models"].items())
    best = o["selected_model"]
    src_rows = "\n".join(
        f"| {s} | {m['n']:,} | {m['base_rate']:.2%} | {f(m['roc_auc'])} | {f(m['pr_auc'])} | {f(m['recall'])} | {f(m['precision'])} |"
        for s, m in o["models"][best]["test_by_source"].items())
    cd = o["cross_domain_shared_features"]
    cm = o["test_confusion_matrix"]
    md = f"""# Báo cáo mô hình Dropout Risk v3 (SmartEdu CRM)

## 1. Thiết lập
- UCI: mốc cuối học kỳ 1 (không dùng dữ liệu học kỳ 2). OULAD: mốc Day {CUTOFF_DAY}.
- Chia Train/Test và CV bằng `StratifiedGroupKFold` theo `id_student` (không sinh viên nào xuất hiện ở cả hai tập).
- Đặc trưng không quan sát được ở một nguồn để NaN (không bịa), được impute trung vị + indicator trong pipeline.
- Mô hình được chọn bằng PR-AUC out-of-fold trên **tập train**: **{best}**. Ngưỡng chọn theo F2 trên OOF.

## 2. Benchmark (test set)
| Mô hình | OOF PR-AUC (train) | ROC-AUC | PR-AUC | Recall | Precision | F2 | Brier |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
{rows}

Mô hình chọn: ROC-AUC 95% CI {o['selected_model_ci']['roc_auc_95ci']}, PR-AUC 95% CI {o['selected_model_ci']['pr_auc_95ci']} (bootstrap theo sinh viên).

## 3. Kết quả theo từng nguồn ({best})
| Nguồn | N | Tỉ lệ dropout | ROC-AUC | PR-AUC | Recall | Precision |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
{src_rows}

Kiểm tra shortcut: ROC-AUC của baseline "chỉ biết nguồn dữ liệu" = **{o['source_only_baseline_roc_auc']}**.
Nếu AUC gộp cao hơn rõ rệt so với AUC từng nguồn, mô hình đang tận dụng khác biệt giữa hai dataset chứ không chỉ hành vi sinh viên.

## 4. Confusion matrix ({best})
TN={cm[0][0]:,} | FP={cm[0][1]:,} | FN={cm[1][0]:,} | TP={cm[1][1]:,}

## 5. Chuyển miền (chỉ dùng đặc trưng chung: {', '.join(SHARED_FEATURES)})
| Kịch bản | ROC-AUC | PR-AUC | Base rate đích |
|:--|:-:|:-:|:-:|
| UCI → OULAD | {cd['UCI_to_OULAD']['roc_auc']} | {cd['UCI_to_OULAD']['pr_auc']} | {cd['UCI_to_OULAD']['target_base_rate']} |
| OULAD → UCI | {cd['OULAD_to_UCI']['roc_auc']} | {cd['OULAD_to_UCI']['pr_auc']} | {cd['OULAD_to_UCI']['target_base_rate']} |
| In-domain UCI (OOF, cùng đặc trưng) | {cd['in_domain_reference_UCI']['roc_auc']} | {cd['in_domain_reference_UCI']['pr_auc']} | - |
| In-domain OULAD (OOF, cùng đặc trưng) | {cd['in_domain_reference_OULAD']['roc_auc']} | {cd['in_domain_reference_OULAD']['pr_auc']} | - |

Khoảng chênh giữa chuyển miền và in-domain cho thấy mức độ khái quát hóa thực sự.

## 6. Hạn chế
- Hai tập có định nghĩa "bỏ học" khác nhau (UCI: Dropout vs Graduate; OULAD: Withdrawn sau Day {CUTOFF_DAY} vs còn lại).
- Đặc trưng là proxy (VLE ≠ điểm danh thực; UCI không có dữ liệu chuyên cần). Cần hiệu chỉnh/huấn luyện lại bằng dữ liệu thật của SmartEdu.
- Chưa đánh giá công bằng (fairness) theo nhóm nhân khẩu học.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"  [OK] {path}")


# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("SMARTEDU CRM -- DROPOUT MODEL v3 TRAINING PIPELINE")
    print("=" * 80)
    uci_df = process_uci(UCI_PATH)
    oulad_df = process_oulad(OULAD_DIR)
    final_df = blend(uci_df, oulad_df, FINAL_DATASET_PATH)
    run_benchmark_and_train(final_df)
    print("\nDone.")