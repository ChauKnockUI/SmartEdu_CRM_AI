"""Build separate public CSVs and evaluate the UCI center-oriented baseline."""
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from dropout_public_common import (
    ACADEMIC_FEATURES, FINANCE_FEATURES, OULAD_FEATURES, SCHEMA_VERSION, SEED,
    TARGET_COL, best_threshold, evaluate, group_holdout, make_pipeline, model_zoo,
    oof_probs,
)
from dropout_public_data import process_uci, process_oulad, validate_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "archive"
PROCESSED = ROOT / "dataset" / "processed"
OUTPUT = ROOT / "ai_models" / "dropout_public"
VARIANTS = {
    "uci_academic": ("UCI", ACADEMIC_FEATURES, "Đối chứng / dự phòng: học tập"),
    "uci_academic_finance": ("UCI", FINANCE_FEATURES, "Model tích hợp: học tập + tài chính"),
    "oulad_assessment": ("OULAD", OULAD_FEATURES, "Bổ sung: bài nộp và điểm"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bootstrap_intervals(y, p, groups):
    y, p = np.asarray(y), np.asarray(p)
    indices = list(pd.DataFrame({"group": np.asarray(groups)}).groupby("group").indices.values())
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(200):
        idx = np.concatenate([indices[i] for i in rng.integers(0, len(indices), len(indices))])
        if np.unique(y[idx]).size == 2:
            values.append([roc_auc_score(y[idx], p[idx]), average_precision_score(y[idx], p[idx])])
    limits = np.percentile(values, [2.5, 97.5], axis=0)
    return {"roc_auc_95ci": limits[:, 0].round(4).tolist(),
            "pr_auc_95ci": limits[:, 1].round(4).tolist()}


def benchmark(name, df, features, protocol):
    train, test = group_holdout(df)
    X, y = train[features], train[TARGET_COL]
    results, fitted, probabilities = {}, {}, {}
    for model_name, estimator in model_zoo().items():
        print(f"  {name}: {model_name}", flush=True)
        pipe = make_pipeline(estimator, features)
        oof = oof_probs(pipe, X, y, train.student_id)
        threshold = best_threshold(y, oof)
        pipe.fit(X, y)
        p = pipe.predict_proba(test[features])[:, 1]
        results[model_name] = {"oof_average_precision": float(average_precision_score(y, oof)),
                               "test": evaluate(test[TARGET_COL], p, threshold)}
        fitted[model_name], probabilities[model_name] = pipe, p
    selected = max((n for n in results if not n.startswith("Dummy")),
                   key=lambda n: results[n]["oof_average_precision"])
    threshold = results[selected]["test"]["threshold"]
    probability = probabilities[selected]
    metadata = {**protocol, "variant": name, "features": features, "selected_model": selected,
                "threshold": threshold, "research_only": True,
                "selection": "Train OOF average precision; threshold: train OOF F2"}
    evaluation_bundle = {**metadata, "pipeline": fitted[selected], "training_scope": "train_only"}
    joblib.dump(evaluation_bundle, OUTPUT / f"{name}_evaluation.pkl")
    final = make_pipeline(clone(model_zoo()[selected]), features).fit(df[features], df[TARGET_COL])
    joblib.dump({**metadata, "pipeline": final, "training_scope": "all_data_refit",
                 "metric_artifact": f"{name}_evaluation.pkl"}, OUTPUT / f"{name}_bundle.pkl")
    pd.DataFrame({"record_id": test.record_id, "student_id": test.student_id,
                  "actual": test[TARGET_COL], "probability": probability,
                  "predicted": (probability >= threshold).astype(int)}).to_csv(
                      OUTPUT / f"{name}_holdout_predictions.csv", index=False)
    return {"features": features, "selected_model": selected, "models": results,
            "selected_ci": bootstrap_intervals(test[TARGET_COL], probability, test.student_id),
            "confusion_matrix": confusion_matrix(test[TARGET_COL], probability >= threshold).tolist(),
            "train_rows": len(train), "test_rows": len(test),
            "protocol": protocol}


def report(results, audits):
    rows = []
    for key, result in results.items():
        t = result["models"][result["selected_model"]]["test"]
        rows.append(f"| {VARIANTS[key][2]} | {result['selected_model']} | {t['n']} | {t['base_rate']:.2%} | "
                    f"{t['roc_auc']:.4f} | {t['pr_auc']:.4f} | {t['recall']:.2%} | {t['precision']:.2%} | {t['brier']:.4f} |")
    lines = ["# Báo cáo dropout: baseline UCI cho nhóm học tập–tài chính", "",
             "UCI là nguồn huấn luyện chính. OULAD là thực nghiệm bổ sung riêng về bài nộp và điểm; không trộn hai nguồn, không dùng VLE hoặc giả lập điểm danh.", "",
             "## Dữ liệu và đặc trưng", "",
             "- `dataset/processed/dropout_uci_center_baseline.csv`: Average_Score, Pass_Rate, Not_Evaluated_Rate, Has_Debt, Tuition_Up_To_Date.",
             "- UCI: điểm kỳ 1 chia 2 về 0–10 khi có học phần đạt; Pass_Rate = học phần đạt / đăng ký; Not_Evaluated_Rate = học phần không đánh giá / đăng ký. Không đồng nhất thiếu đánh giá với không nộp bài.",
             "- Has_Debt và Tuition_Up_To_Date giữ riêng hai cờ gốc; không suy ra số ngày quá hạn hoặc hóa đơn.",
             "- `dataset/processed/dropout_oulad_assessment.csv`: Average_Score, Pass_Rate, Assignment_Missing_Rate, Assignment_Late_Count, Score_Trend; giữ khóa môn/kỳ/sinh viên.",
             "- OULAD: học viên đã đăng ký và chưa hủy tại ngày 30; bài nộp đến ngày 30; thiếu bài chỉ tính trên bài đã đến hạn; nhãn Withdrawn vs Pass/Distinction/Fail.",
             "- CSV giữ NaN và thang đo nghiệp vụ. Giá trị sai miền được ghi audit và chuyển thành NaN, không clip. Median imputation + missing indicators + StandardScaler được fit trong pipeline của từng fold.", "",
             "## Protocol", "",
             "Holdout 1/5 bằng StratifiedGroupKFold theo sinh viên; OOF 5 fold chỉ trên train. Hai cấu hình UCI dùng cùng train/test. Chọn model bằng average precision OOF và ngưỡng bằng F2 OOF. Test chỉ dùng để báo cáo.",
             "UCI không có ID sinh viên gốc; student_id là ID dòng ổn định trong file nguồn. Không thể xác minh trùng danh tính thực. OULAD giữ ID sinh viên gốc để ngăn chồng lấn giữa các kỳ.", "",
             "## Kết quả holdout", "",
             "PR-AUC trong báo cáo được tính bằng average_precision_score; ROC-AUC không phải tỷ lệ dự đoán đúng.", "",
             "| Cấu hình | Model | N test | Tỷ lệ dropout | ROC-AUC | PR-AUC | Recall | Precision | Brier |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|", *rows, ""]
    for key, result in results.items():
        selected = result["selected_model"]
        metrics = result["models"][selected]["test"]
        cm = result["confusion_matrix"]
        baseline = result["models"]["Dummy (prior)"]["test"]
        lines += [f"### {VARIANTS[key][2]}", "",
                  f"Train/test: {result['train_rows']}/{result['test_rows']}. Feature: {', '.join(result['features'])}.",
                  f"Model: {selected}; ngưỡng F2 OOF: {metrics['threshold']:.6f}. ROC-AUC CI 95% {result['selected_ci']['roc_auc_95ci']}; AP CI 95% {result['selected_ci']['pr_auc_95ci']} (200 bootstrap theo sinh viên).",
                  f"TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}; gắn cờ {(cm[0][1] + cm[1][1]) / metrics['n']:.2%} test.",
                  f"Baseline prior: AP={baseline['pr_auc']:.4f}, Brier={baseline['brier']:.4f}. Brier thấp hơn là tốt hơn.", "",
                  "| Model | AP OOF train | ROC-AUC test | AP test | F2 test |",
                  "|---|---:|---:|---:|---:|"]
        for model, r in result["models"].items():
            lines.append(f"| {model} | {r['oof_average_precision']:.4f} | {r['test']['roc_auc']:.4f} | {r['test']['pr_auc']:.4f} | {r['test']['f2']:.4f} |")
        lines.append("")
    lines += ["## Audit và phạm vi sử dụng", "",
              f"UCI: {audits['UCI']['raw_rows']} dòng gốc; loại {audits['UCI']['excluded_unresolved_target']} dòng chưa có nhãn Dropout/Graduate, {audits['UCI']['excluded_invalid_or_zero_enrollment']} dòng không có mẫu số đăng ký hợp lệ; còn {audits['UCI']['output_rows']} dòng.",
              f"OULAD: {audits['OULAD']['raw_rows']} lượt gốc; còn {audits['OULAD']['output_rows']} lượt tại ngày 30. Chi tiết lỗi/loại mẫu: data_audit.json; missingness: missing_rates.csv.", "",
              "- API dùng `uci_academic_finance_bundle.pkl` với năm feature khi có học phí; `uci_academic_bundle.pkl` là đối chứng và dự phòng khi học phí chưa quan sát được. Thuật toán của mỗi cấu hình vẫn được chọn theo AP OOF train, không theo test.",
              "- UCI là dữ liệu đại học cuối kỳ 1, không chứng minh hiệu quả cảnh báo tuần 3–4 hoặc hiệu quả tại trung tâm. Không có chuyên cần thực trong các CSV này.",
              "- Chưa xác minh thời điểm snapshot tài chính UCI và công bố điểm OULAD; kết quả không chứng minh đã loại hết leakage thời gian. Nhãn UCI là kết quả cuối cùng, không có ngày bỏ học để xác minh mọi ca đều xảy ra sau mốc kỳ 1.",
              "- Điểm test thuộc artifact *_evaluation.pkl (chỉ fit train); *_bundle.pkl được refit toàn bộ nguồn sau đánh giá, không có phép kiểm thử độc lập riêng.",
              "- Backend ánh xạ bài đánh giá và hóa đơn của đúng học viên–lớp thành proxy UCI; AI chạy từ điểm hợp lệ đầu tiên. Has_Debt là còn dư nợ, kể cả chưa đến hạn; Tuition_Up_To_Date là không còn khoản quá hạn. Thiếu hóa đơn hoặc thiếu hạn thanh toán cần thiết thì dùng model học tập. Các proxy tài chính không được khẳng định tương đương hoàn toàn với hai cờ UCI. Xem ../DROPOUT_INTEGRATION.md.",
              "- UI dùng điểm tổng hợp hybrid_risk_2: khi AI có học phí, AI 70% + chuyên cần 30%, không cộng trọng số học phí riêng. Khi AI chỉ có học tập hoặc không sẵn sàng, dùng chính sách 50/30/20 cho AI học tập/chuyên cần/học phí với trọng số phân bổ lại. Mức sàn học phí vẫn là quy tắc ưu tiên hỗ trợ. Đây không phải xác suất đã hiệu chỉnh; benchmark chỉ đánh giá model công khai.",
              "- Cần snapshot và nhãn thực tế của trung tâm để huấn luyện/kiểm chứng chuyên cần, học phí và cảnh báo trong khóa.", ""]
    (OUTPUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def dashboard():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for name, (_, _, label) in VARIANTS.items():
        prediction = pd.read_csv(OUTPUT / f"{name}_holdout_predictions.csv")
        row = 0 if name.startswith("uci") else 1
        y, p = prediction.actual, prediction.probability
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        axes[row, 0].plot(fpr, tpr, label=name)
        axes[row, 1].plot(rec, prec, label=name)
    for i, source in enumerate(["UCI", "OULAD assessments"]):
        axes[i, 0].plot([0, 1], [0, 1], "k:")
        axes[i, 0].set(title=f"{source}: ROC", xlabel="False positive rate", ylabel="Recall")
        axes[i, 1].set(title=f"{source}: Precision / Recall", xlabel="Recall", ylabel="Precision")
        for ax in axes[i]:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT / "dashboard.png", dpi=160)
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    PROCESSED.mkdir(exist_ok=True)
    uci, uci_audit = process_uci(RAW / "dropout_data_uci.csv")
    oulad, oulad_audit = process_oulad(RAW / "dropout_data_OULAD")
    paths = {"UCI": PROCESSED / "dropout_uci_center_baseline.csv",
             "OULAD": PROCESSED / "dropout_oulad_assessment.csv"}
    data = {}
    for source, frame in [("UCI", uci), ("OULAD", oulad)]:
        frame.to_csv(paths[source], index=False)
        data[source] = pd.read_csv(paths[source], dtype={"student_id": str}, float_precision="round_trip")
        validate_dataset(data[source], source)
    audits = {"UCI": uci_audit, "OULAD": oulad_audit}
    write_json(OUTPUT / "data_audit.json", audits)
    missing = [pd.DataFrame({"source": source, "feature": features,
                             "missing_rate": [data[source][f].isna().mean() for f in features]})
               for source, features in [("UCI", FINANCE_FEATURES), ("OULAD", OULAD_FEATURES)]]
    pd.concat(missing).to_csv(OUTPUT / "missing_rates.csv", index=False)
    protocols = {}
    for source, df in data.items():
        train, test = group_holdout(df)
        assignments = pd.concat([train[["record_id", "student_id"]].assign(split="train"),
                                 test[["record_id", "student_id"]].assign(split="test")], ignore_index=True)
        split_path = OUTPUT / f"{source.lower()}_split.csv"
        assignments.to_csv(split_path, index=False)
        raw_files = [RAW / "dropout_data_uci.csv"] if source == "UCI" else [RAW / "dropout_data_OULAD" / f
            for f in ["studentInfo.csv", "studentRegistration.csv", "assessments.csv", "studentAssessment.csv"]]
        protocols[source] = {"schema_version": SCHEMA_VERSION, "source": source, "seed": SEED,
                             "dataset_sha256": digest(paths[source]), "split_sha256": digest(split_path),
                             "raw_sha256": {p.name: digest(p) for p in raw_files},
                             "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__,
                             "split": "StratifiedGroupKFold by student, 1/5 holdout, 5-fold train OOF"}
    results = {}
    for name, (source, features, _) in VARIANTS.items():
        results[name] = benchmark(name, data[source], features, protocols[source])
    write_json(OUTPUT / "benchmark.json", results)
    report(results, audits)
    dashboard()
    print(f"Done. Canonical report: {OUTPUT / 'report.md'}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
