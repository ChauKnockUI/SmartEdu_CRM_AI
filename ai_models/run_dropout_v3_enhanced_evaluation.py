"""
run_dropout_v3_enhanced_evaluation.py
--------------------------------------
SmartEdu CRM -- Đánh giá Bổ sung Nâng cao cho Mô hình Dropout Risk v3.
Chạy độc lập để sinh toàn bộ các bảng chỉ số, biểu đồ độ tin cậy và báo cáo chuyên sâu.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "archive")
FINAL_DATASET_PATH = os.path.join(DATASET_DIR, "dropout_real_final_dataset_v3.csv")
OULAD_DIR = os.path.join(DATASET_DIR, "dropout_data_OULAD")
REPORTS_DIR = os.path.join(CURRENT_DIR, "reports_v3")
os.makedirs(REPORTS_DIR, exist_ok=True)

SEED = 42
CUTOFF_DAY = 30
N_FOLDS = 5
TEST_FRAC = 0.20
KEYS = ["code_module", "code_presentation", "id_student"]
TARGET_COL = "Is_Dropout"

FEATURE_COLS = [
    "Attendance_Rate",
    "Consecutive_Unexcused_Absences",
    "Days_Since_Last_Attended",
    "Log_Total_Activity",
    "Assignment_Missing_Rate",
    "Assignment_Late_Count",
    "Average_Score",
    "Score_Trend",
    "Pass_Rate",
    "Has_Overdue_Invoice",
]

SRC_FEATURES = {
    "OULAD": [c for c in FEATURE_COLS if c != "Has_Overdue_Invoice"],
    "UCI": ["Average_Score", "Assignment_Missing_Rate", "Pass_Rate", "Has_Overdue_Invoice"],
}

rng = np.random.default_rng(SEED)
extra_results = {}

def make_pipeline(classifier, feature_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, feature_cols)],
        remainder="drop"
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])

def topk_table(y, p, ks=(0.05, 0.10, 0.20, 0.30)):
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p)
    base = y.mean()
    rows = []
    for k in ks:
        n = max(1, int(round(len(y) * k)))
        tp = y[order[:n]].sum()
        prec = tp / n
        rows.append({
            "Gắn cờ top": f"{k:.0%}",
            "Số HV được gắn cờ": n,
            "Precision": round(prec, 3),
            "Recall": round(tp / y.sum(), 3) if y.sum() else np.nan,
            "Lift (x lần ngẫu nhiên)": round(prec / base, 2),
        })
    return pd.DataFrame(rows)

def bootstrap_ci(y, p, groups, metric, n_boot=500):
    y, p = np.asarray(y), np.asarray(p)
    gidx = pd.Series(np.arange(len(y))).groupby(np.asarray(groups)).apply(np.array)
    gidx = list(gidx.values)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(gidx), len(gidx))
        idx = np.concatenate([gidx[i] for i in pick])
        if y[idx].min() == y[idx].max():
            continue
        vals.append(metric(y[idx], p[idx]))
    lo, mid, hi = np.percentile(vals, [2.5, 50, 97.5])
    return f"{mid:.3f} [{lo:.3f} - {hi:.3f}]"

def src_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=SEED
        ),
    }

def main():
    print("=" * 80)
    print("SMARTEDU CRM -- DROPOUT RISK v3: ĐÁNH GIÁ BỔ SUNG NÂNG CAO")
    print("=" * 80)

    # 1. Đọc dữ liệu
    print(f"\n[1/7] Đang nạp dữ liệu: {FINAL_DATASET_PATH}")
    final_df = pd.read_csv(FINAL_DATASET_PATH)
    final_df["id_student"] = final_df["id_student"].astype(str)

    # Split train/test (group by id_student)
    sgkf = StratifiedGroupKFold(n_splits=int(round(1.0 / TEST_FRAC)), shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(final_df, final_df[TARGET_COL], groups=final_df["id_student"]))
    train_df = final_df.iloc[train_idx].reset_index(drop=True)
    test_df = final_df.iloc[test_idx].reset_index(drop=True)
    yte = test_df[TARGET_COL].values
    print(f"[*] Train set: {len(train_df):,} dòng | Test set: {len(test_df):,} dòng (0% rò rỉ id_student)")

    # 2. Huấn luyện Mô hình Gộp (Pooled HistGradientBoosting)
    print("\n[2/7] Huấn luyện Mô hình Gộp (Pooled HistGradientBoosting)...")
    best_name = "HistGradientBoosting"
    pool_clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=SEED)
    pool_pipe = make_pipeline(pool_clf, FEATURE_COLS)
    pool_pipe.fit(train_df[FEATURE_COLS], train_df[TARGET_COL])
    test_prob = {best_name: pool_pipe.predict_proba(test_df[FEATURE_COLS])[:, 1]}
    best_thr = 0.3401

    # Đánh giá riêng từng nguồn kèm khoảng tin cậy 95%
    rows = []
    for src in ["OULAD", "UCI"]:
        m = (test_df["Data_Source"] == src).values
        y_s, p_s, g_s = yte[m], test_prob[best_name][m], test_df.loc[m, "id_student"].values
        rows.append({
            "Nguồn": src,
            "n": int(m.sum()),
            "Tỉ lệ bỏ học thật": round(y_s.mean(), 3),
            "ROC-AUC [95% CI]": bootstrap_ci(y_s, p_s, g_s, roc_auc_score),
            "PR-AUC [95% CI]": bootstrap_ci(y_s, p_s, g_s, average_precision_score),
        })
    per_source_ci = pd.DataFrame(rows)
    print("\n=== ĐÁNH GIÁ MÔ HÌNH GỘP THEO TỪNG NGUỒN KÈM 95% CI ===")
    print(per_source_ci.to_string(index=False))
    extra_results["pooled_per_source_ci"] = per_source_ci

    # 3. Tách mô hình theo nguồn (Separate Models)
    print("\n[3/7] Huấn luyện mô hình riêng biệt theo nguồn (OULAD riêng / UCI riêng)...")
    src_fit, src_oof, src_test_prob, sep_rows = {}, {}, {}, []
    cv_src = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for src, feats in SRC_FEATURES.items():
        tr = train_df[train_df["Data_Source"] == src].reset_index(drop=True)
        te = test_df[test_df["Data_Source"] == src].reset_index(drop=True)
        for name, clf in src_models().items():
            pipe = make_pipeline(clf, feats)
            oof = cross_val_predict(pipe, tr[feats], tr[TARGET_COL], cv=cv_src,
                                    groups=tr["id_student"], method="predict_proba", n_jobs=1)[:, 1]
            pipe.fit(tr[feats], tr[TARGET_COL])
            prob = pipe.predict_proba(te[feats])[:, 1]
            src_fit[(src, name)], src_oof[(src, name)], src_test_prob[(src, name)] = pipe, oof, prob
            sep_rows.append({
                "Nguồn": src, "Mô hình": name,
                "OOF PR-AUC": round(average_precision_score(tr[TARGET_COL], oof), 4),
                "Test ROC-AUC": round(roc_auc_score(te[TARGET_COL], prob), 4),
                "Test PR-AUC": round(average_precision_score(te[TARGET_COL], prob), 4),
                "Base rate": round(te[TARGET_COL].mean(), 3),
                "Brier": round(brier_score_loss(te[TARGET_COL], prob), 4),
            })

    for src in SRC_FEATURES:
        m = (test_df["Data_Source"] == src).values
        sep_rows.append({
            "Nguồn": src, "Mô hình": f"[Gộp] {best_name}",
            "OOF PR-AUC": np.nan,
            "Test ROC-AUC": round(roc_auc_score(yte[m], test_prob[best_name][m]), 4),
            "Test PR-AUC": round(average_precision_score(yte[m], test_prob[best_name][m]), 4),
            "Base rate": round(yte[m].mean(), 3),
            "Brier": round(brier_score_loss(yte[m], test_prob[best_name][m]), 4),
        })
    sep_table = pd.DataFrame(sep_rows).sort_values(["Nguồn", "Mô hình"]).reset_index(drop=True)
    print("\n=== ĐỐI CHIẾU MÔ HÌNH RIÊNG BIỆT VS MÔ HÌNH GỘP ===")
    print(sep_table.to_string(index=False))
    extra_results["separate_vs_pooled"] = sep_table

    # 4. Ngưỡng theo Năng lực Can thiệp CRM (Precision@k, Recall@k, Lift)
    print("\n[4/7] Đo lường hiệu quả can thiệp CRM theo năng lực Cố vấn học tập (Top-k)...")
    oul_te = test_df[test_df["Data_Source"] == "OULAD"].reset_index(drop=True)
    y_oul = oul_te[TARGET_COL].values
    tk_sep = topk_table(y_oul, src_test_prob[("OULAD", "HistGradientBoosting")])
    m_oul = (test_df["Data_Source"] == "OULAD").values
    tk_pool = topk_table(y_oul, test_prob[best_name][m_oul])
    uci_te = test_df[test_df["Data_Source"] == "UCI"].reset_index(drop=True)
    tk_uci = topk_table(uci_te[TARGET_COL].values, src_test_prob[("UCI", "HistGradientBoosting")])

    print("\n--- OULAD (Mô hình riêng) ---")
    print(tk_sep.to_string(index=False))
    print("\n--- OULAD (Mô hình gộp) ---")
    print(tk_pool.to_string(index=False))
    print("\n--- UCI (Mô hình riêng) ---")
    print(tk_uci.to_string(index=False))

    extra_results["topk_oulad_separate"] = tk_sep
    extra_results["topk_oulad_pooled"] = tk_pool
    extra_results["topk_uci_separate"] = tk_uci

    CAPACITY = 0.15
    oof_oul = src_oof[("OULAD", "HistGradientBoosting")]
    thr_capacity = float(np.quantile(oof_oul, 1 - CAPACITY))
    print(f"\n[*] Ngưỡng tự động cho Cố vấn học tập gọi 15% học viên: {thr_capacity:.4f}")

    # 5. Hiệu chỉnh Xác suất (Isotonic Calibration)
    print("\n[5/7] Hiệu chỉnh xác suất (Isotonic Regression) & Vẽ Reliability Curve...")
    key = ("OULAD", "HistGradientBoosting")
    tr_oul = train_df[train_df["Data_Source"] == "OULAD"].reset_index(drop=True)
    iso = IsotonicRegression(out_of_bounds="clip").fit(src_oof[key], tr_oul[TARGET_COL])
    p_raw = src_test_prob[key]
    p_cal = iso.predict(p_raw)

    print(f"[*] Brier score trước hiệu chỉnh : {brier_score_loss(y_oul, p_raw):.4f}")
    print(f"[*] Brier score sau hiệu chỉnh   : {brier_score_loss(y_oul, p_cal):.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for p, lab in [(p_raw, "Trước hiệu chỉnh"), (p_cal, "Sau hiệu chỉnh (isotonic)")]:
        frac_pos, mean_pred = calibration_curve(y_oul, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=lab)
    ax.plot([0, 1], [0, 1], "k:", label="Hiệu chỉnh hoàn hảo")
    ax.set(title="Reliability Curve -- OULAD (Holdout Test)", xlabel="Xác suất dự đoán", ylabel="Tỉ lệ bỏ học thực tế")
    ax.legend()
    plt.tight_layout()
    cal_png = os.path.join(REPORTS_DIR, "calibration_oulad.png")
    plt.savefig(cal_png, dpi=150)
    plt.close()
    print(f"[OK] Đã lưu biểu đồ: {cal_png}")

    # 6. Kiểm tra Suy biến Thời gian (2013 -> 2014)
    print("\n[6/7] Kiểm tra suy biến theo thời gian (Train 2013 -> Test 2014)...")
    # Lấy thông tin kỳ học từ studentRegistration
    reg_path = os.path.join(OULAD_DIR, "studentRegistration.csv")
    if os.path.exists(reg_path):
        reg = pd.read_csv(reg_path)
        reg["id_student"] = reg["id_student"].astype(str)
        oulad_rows = final_df[final_df["Data_Source"] == "OULAD"].copy()
        oulad_rows = oulad_rows.merge(reg[KEYS].drop_duplicates(), on="id_student", how="left")
        oulad_rows["year"] = oulad_rows["code_presentation"].astype(str).str[:4]
        oulad_rows["year"] = pd.to_numeric(oulad_rows["year"], errors="coerce")
        tr_t = oulad_rows[oulad_rows["year"] == 2013].reset_index(drop=True)
        te_t = oulad_rows[oulad_rows["year"] == 2014].reset_index(drop=True)
        tr_t = tr_t[~tr_t["id_student"].isin(set(te_t["id_student"]))].reset_index(drop=True)

        t_feats = SRC_FEATURES["OULAD"]
        t_rows = []
        for name, clf in src_models().items():
            pipe = make_pipeline(clf, t_feats).fit(tr_t[t_feats], tr_t[TARGET_COL])
            prob = pipe.predict_proba(te_t[t_feats])[:, 1]
            t_rows.append({
                "Mô hình": name,
                "n_train (2013)": len(tr_t), "n_test (2014)": len(te_t),
                "Base rate test": round(te_t[TARGET_COL].mean(), 3),
                "ROC-AUC": round(roc_auc_score(te_t[TARGET_COL], prob), 4),
                "PR-AUC": round(average_precision_score(te_t[TARGET_COL], prob), 4),
            })
        temporal = pd.DataFrame(t_rows)
        print("\n=== KẾT QUẢ KIỂM THỬ THỜI GIAN (TRAIN 2013 -> TEST 2014) ===")
        print(temporal.to_string(index=False))
        extra_results["temporal_oulad"] = temporal

    # 7. Lưu tất cả kết quả
    print("\n[7/7] Đang lưu toàn bộ kết quả vào reports_v3...")
    for name, tbl in extra_results.items():
        csv_file = os.path.join(REPORTS_DIR, f"extra_{name}.csv")
        tbl.to_csv(csv_file, index=False, encoding="utf-8-sig")
        print(f"  [+] {csv_file}")

    print(f"\n[HOÀN TẤT] Toàn bộ đánh giá bổ sung v3 đã hoàn tất thành công!")

if __name__ == "__main__":
    main()
