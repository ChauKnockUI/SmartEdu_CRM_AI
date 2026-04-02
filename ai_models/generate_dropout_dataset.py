"""
generate_dropout_dataset.py
----------------------------
Generates a synthetic dataset for training the Student Dropout Risk model.

This script follows a Persona-Based Generation approach — instead of purely
random sampling, it defines 5 realistic student archetype groups (Personas)
that reflect actual behavior patterns observed in English language centers.
Mixing these personas produces a dataset with meaningful, real-world
correlations between features and the dropout label.

Personas:
    1. Dedicated         (22%) — Dropout rate ~4%
    2. Busy Professional (20%) — Dropout rate ~12%
    3. Average Student   (25%) — Dropout rate ~22%
    4. Wavering          (20%) — Dropout rate ~65%
    5. About to Drop     (13%) — Dropout rate ~88%

Features (Model Inputs):
    - Excused_Absences      : Sessions missed with prior notice
    - Unexcused_Absences    : Sessions missed without any notice (highest risk signal)
    - Consecutive_Absences  : Current streak of consecutive missed sessions (red flag)
    - Missed_Homework_Ratio : Ratio of homework assignments not submitted (0.0-1.0)

Target (Model Output):
    - Is_Dropout : 1 = dropped / high risk of dropping, 0 = continuing normally

Output:
    dataset/archive/dropout_dataset.csv
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_SEED   = 42
TOTAL_RECORDS = 3000
OUTPUT_PATH   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "archive", "dropout_dataset.csv"
)

np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------
# Each persona captures a distinct student archetype common in English centers.
# Distribution parameters are calibrated to reflect realistic center statistics.

PERSONAS = [

    # Persona 1: DEDICATED
    # Học viên gương mẫu — đặt mục tiêu rõ ràng (thi TOEIC ra trường, IELTS du học),
    # hiếm khi vắng, có vắng thì chủ động xin phép trước.
    {
        "name"  : "Dedicated",
        "ratio" : 0.22,
        "features": {
            "Excused_Absences"    : {"dist": "poisson", "lam": 0.5,  "clip": (0, 3)},
            "Unexcused_Absences"  : {"dist": "poisson", "lam": 0.1,  "clip": (0, 1)},
            "Consecutive_Absences": {"dist": "poisson", "lam": 0.1,  "clip": (0, 1)},
            "Missed_Homework_Ratio": {"dist": "beta",   "a": 1.5, "b": 12, "clip": (0.0, 0.25)},
        },
        "dropout_base_prob": 0.04,
    },

    # Persona 2: BUSY PROFESSIONAL
    # Người đi làm toàn thời gian — thỉnh thoảng bận đột xuất nhưng vẫn chủ động
    # báo vắng, cố gắng theo kịp bài. Ít vắng không phép.
    {
        "name"  : "Busy_Professional",
        "ratio" : 0.20,
        "features": {
            "Excused_Absences"    : {"dist": "poisson", "lam": 2.5, "clip": (1, 7)},
            "Unexcused_Absences"  : {"dist": "poisson", "lam": 0.4, "clip": (0, 2)},
            "Consecutive_Absences": {"dist": "poisson", "lam": 0.3, "clip": (0, 2)},
            "Missed_Homework_Ratio": {"dist": "beta",   "a": 2.0, "b": 7, "clip": (0.05, 0.45)},
        },
        "dropout_base_prob": 0.12,
    },

    # Persona 3: AVERAGE STUDENT
    # Học viên trung bình — đi học khá đều, thỉnh thoảng nghỉ không báo,
    # làm bài tập ở mức vừa phải. Dễ mất động lực nếu không được khích lệ.
    {
        "name"  : "Average_Student",
        "ratio" : 0.25,
        "features": {
            "Excused_Absences"    : {"dist": "poisson", "lam": 1.5, "clip": (0, 5)},
            "Unexcused_Absences"  : {"dist": "poisson", "lam": 1.2, "clip": (0, 4)},
            "Consecutive_Absences": {"dist": "poisson", "lam": 0.6, "clip": (0, 2)},
            "Missed_Homework_Ratio": {"dist": "beta",   "a": 3.0, "b": 7, "clip": (0.05, 0.55)},
        },
        "dropout_base_prob": 0.22,
    },

    # Persona 4: WAVERING
    # Học viên đang chùn bước — nghỉ nhiều không báo, lỡ nhiều bài tập.
    # Đang ở giai đoạn mất phương hướng, tinh thần giảm sút rõ rệt.
    # Nhóm này CSKH cần ưu tiên tiếp cận.
    {
        "name"  : "Wavering",
        "ratio" : 0.20,
        "features": {
            "Excused_Absences"    : {"dist": "poisson", "lam": 1.5, "clip": (0, 5)},
            "Unexcused_Absences"  : {"dist": "poisson", "lam": 2.8, "clip": (1, 7)},
            "Consecutive_Absences": {"dist": "poisson", "lam": 1.8, "clip": (0, 4)},
            "Missed_Homework_Ratio": {"dist": "beta",   "a": 5.0, "b": 4, "clip": (0.25, 0.85)},
        },
        "dropout_base_prob": 0.65,
    },

    # Persona 5: ABOUT TO DROP
    # Học viên gần bỏ học — đã nghỉ nhiều buổi liên tiếp không báo,
    # gần như không nộp bài tập. Cần gọi điện cảnh báo ngay lập tức.
    {
        "name"  : "About_to_Drop",
        "ratio" : 0.13,
        "features": {
            "Excused_Absences"    : {"dist": "poisson", "lam": 0.8, "clip": (0, 3)},
            "Unexcused_Absences"  : {"dist": "poisson", "lam": 4.5, "clip": (2, 9)},
            "Consecutive_Absences": {"dist": "poisson", "lam": 3.5, "clip": (2, 7)},
            "Missed_Homework_Ratio": {"dist": "beta",   "a": 8.0, "b": 2, "clip": (0.5, 1.0)},
        },
        "dropout_base_prob": 0.88,
    },
]


# ---------------------------------------------------------------------------
# Feature sampling
# ---------------------------------------------------------------------------

def _sample(cfg: dict, n: int) -> np.ndarray:
    """
    Draw `n` samples from the distribution specified in a persona feature config.

    Supported distributions:
        'poisson' — for discrete count variables (number of absences, etc.)
        'beta'    — for continuous ratio variables bounded between 0 and 1
    """
    dist = cfg["dist"]
    lo, hi = cfg["clip"]

    if dist == "poisson":
        values = np.random.poisson(lam=cfg["lam"], size=n).astype(float)
    elif dist == "beta":
        values = np.random.beta(a=cfg["a"], b=cfg["b"], size=n)
    else:
        raise ValueError(f"Unsupported distribution type: '{dist}'")

    return np.clip(values, lo, hi)


# ---------------------------------------------------------------------------
# Dropout probability computation
# ---------------------------------------------------------------------------

def _compute_dropout_prob(
    base_prob         : float,
    excused_abs       : np.ndarray,
    unexcused_abs     : np.ndarray,
    consecutive_abs   : np.ndarray,
    missed_hw_ratio   : np.ndarray,
) -> np.ndarray:
    """
    Refine the dropout probability around each persona's base rate by applying
    additive risk adjustments based on observed behavioral signals.

    Risk factor weights (ordered by severity):
        +30pp — Consecutive absences >= 3  (strongest single dropout signal)
        +25pp — Unexcused absences >= 4
        +12pp — Unexcused absences >= 2
        +15pp — Missed homework ratio >= 0.6
        +10pp — Consecutive >= 2 AND Unexcused >= 2 simultaneously (interaction)
        +8pp  — Excused absences >= 5 (losing momentum even with valid reasons)
        +8pp  — Excused >= 3 AND homework ratio >= 0.4 (compound disengagement)
        -5pp  — All-clear bonus: consecutive=0, unexcused=0, missed_hw<=0.1
    """
    prob = np.full(len(excused_abs), base_prob, dtype=float)

    # Primary risk signals
    prob = np.where(consecutive_abs >= 3, prob + 0.30, prob)
    prob = np.where(unexcused_abs >= 4,   prob + 0.25,
           np.where(unexcused_abs >= 2,   prob + 0.12, prob))
    prob = np.where(missed_hw_ratio >= 0.6, prob + 0.15, prob)

    # Interaction effect: simultaneous consecutive + unexcused absences
    double_danger = (consecutive_abs >= 2) & (unexcused_abs >= 2)
    prob = np.where(double_danger, prob + 0.10, prob)

    # Momentum loss from too many excused absences
    prob = np.where(excused_abs >= 5, prob + 0.08, prob)

    # Compounding: moderate excused absences + high homework skipping
    compound_disengagement = (excused_abs >= 3) & (missed_hw_ratio >= 0.4)
    prob = np.where(compound_disengagement, prob + 0.08, prob)

    # Positive reinforcement: strong engagement pattern
    strong_engagement = (
        (consecutive_abs == 0) & (unexcused_abs == 0) & (missed_hw_ratio <= 0.1)
    )
    prob = np.where(strong_engagement, prob - 0.05, prob)

    return np.clip(prob, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate(total: int = TOTAL_RECORDS) -> pd.DataFrame:
    """
    Generate a synthetic student dataset by sampling from each defined persona
    and combining them into a single shuffled DataFrame.

    Args:
        total: Total number of records to generate across all personas.

    Returns:
        A DataFrame containing all features, the dropout label, and a
        temporary '_Persona' column for internal quality reporting.
    """
    all_records = []

    for persona in PERSONAS:
        n   = int(total * persona["ratio"])
        cfg = persona["features"]

        excused     = _sample(cfg["Excused_Absences"],    n)
        unexcused   = _sample(cfg["Unexcused_Absences"],  n)
        consecutive = _sample(cfg["Consecutive_Absences"], n)
        missed_hw   = _sample(cfg["Missed_Homework_Ratio"], n)

        # Enforce logic constraint: consecutive absences cannot
        # exceed the total number of sessions missed (excused + unexcused)
        consecutive = np.minimum(consecutive, excused + unexcused)

        # Compute per-sample dropout probability and draw Bernoulli label
        dropout_prob = _compute_dropout_prob(
            base_prob       = persona["dropout_base_prob"],
            excused_abs     = excused,
            unexcused_abs   = unexcused,
            consecutive_abs = consecutive,
            missed_hw_ratio = missed_hw,
        )
        is_dropout = (np.random.rand(n) < dropout_prob).astype(int)

        records = pd.DataFrame({
            "Excused_Absences"     : np.round(excused).astype(int),
            "Unexcused_Absences"   : np.round(unexcused).astype(int),
            "Consecutive_Absences" : np.round(consecutive).astype(int),
            "Missed_Homework_Ratio": np.round(missed_hw, 4),
            "Is_Dropout"           : is_dropout,
            "_Persona"             : persona["name"],
        })
        all_records.append(records)

    df = pd.concat(all_records, ignore_index=True)

    # Shuffle to avoid the model learning persona-order artifacts
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def _print_quality_report(df: pd.DataFrame) -> None:
    """Print a distribution summary to verify dataset quality before exporting."""
    total     = len(df)
    n_dropout = int(df["Is_Dropout"].sum())
    n_safe    = total - n_dropout

    print(f"\n{'=' * 60}")
    print("  DATASET QUALITY REPORT")
    print(f"{'=' * 60}")
    print(f"  Total records : {total:,}")
    print(f"  Dropout  (1)  : {n_dropout:,}  ({n_dropout / total * 100:.1f}%)")
    print(f"  Safe     (0)  : {n_safe:,}  ({n_safe / total * 100:.1f}%)")

    print("\n  Feature Statistics:")
    stats = df[["Excused_Absences", "Unexcused_Absences",
                "Consecutive_Absences", "Missed_Homework_Ratio"]].describe()
    print(stats.round(3).to_string())

    print("\n  Dropout Rate by Persona:")
    persona_stats = (
        df.groupby("_Persona")["Is_Dropout"]
        .agg(count="count", dropout_count="sum", dropout_rate="mean")
    )
    persona_stats["dropout_rate"] = persona_stats["dropout_rate"].map("{:.1%}".format)
    print(persona_stats.to_string())

    print("\n  Correlation with Is_Dropout:")
    corr = (
        df[["Excused_Absences", "Unexcused_Absences",
            "Consecutive_Absences", "Missed_Homework_Ratio", "Is_Dropout"]]
        .corr()["Is_Dropout"]
        .drop("Is_Dropout")
    )
    for col, val in corr.items():
        bar  = "#" * int(abs(val) * 30)
        sign = "+" if val > 0 else "-"
        print(f"    {col:<30}: {sign}{abs(val):.4f}  {bar}")

    print(f"\n{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nGenerating Dropout Risk dataset...")

    df = generate(TOTAL_RECORDS)
    _print_quality_report(df)

    # Export: drop the internal _Persona column before saving
    df_export = df.drop(columns=["_Persona"])
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_export.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Shape: {df_export.shape[0]:,} rows x {df_export.shape[1]} columns\n")


if __name__ == "__main__":
    main()
