"""
Generate a synthetic v2 dataset for SmartEdu dropout early warning.

The dataset is rule-assisted synthetic data. It is meant to bootstrap the
workflow until enough real AIPrediction + outcome data exists for retraining.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, "dataset", "archive", "dropout_dataset_v2.csv")
RANDOM_SEED = 42
TOTAL_RECORDS = 6000

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

PERSONAS = [
    ("dedicated", 0.22, 0.03),
    ("busy_responsible", 0.18, 0.10),
    ("average", 0.24, 0.22),
    ("academic_struggle", 0.14, 0.42),
    ("financially_at_risk", 0.10, 0.48),
    ("disengaged", 0.12, 0.72),
]


def _clip(values, lo, hi):
    return np.clip(values, lo, hi)


def _persona_features(name: str, n: int) -> dict:
    if name == "dedicated":
        attendance = _clip(np.random.normal(0.94, 0.04, n), 0.75, 1.0)
        missing = _clip(np.random.beta(1.2, 12, n), 0, 0.22)
        avg = _clip(np.random.normal(8.2, 0.8, n), 5.5, 10)
        overdue = np.random.binomial(1, 0.03, n)
    elif name == "busy_responsible":
        attendance = _clip(np.random.normal(0.84, 0.08, n), 0.55, 1.0)
        missing = _clip(np.random.beta(2, 8, n), 0.03, 0.45)
        avg = _clip(np.random.normal(7.4, 0.9, n), 4.8, 10)
        overdue = np.random.binomial(1, 0.08, n)
    elif name == "average":
        attendance = _clip(np.random.normal(0.76, 0.10, n), 0.42, 1.0)
        missing = _clip(np.random.beta(3, 6, n), 0.05, 0.62)
        avg = _clip(np.random.normal(6.6, 1.0, n), 3.5, 9.5)
        overdue = np.random.binomial(1, 0.15, n)
    elif name == "academic_struggle":
        attendance = _clip(np.random.normal(0.70, 0.12, n), 0.32, 0.98)
        missing = _clip(np.random.beta(4, 5, n), 0.10, 0.78)
        avg = _clip(np.random.normal(4.8, 1.0, n), 1.5, 7.0)
        overdue = np.random.binomial(1, 0.18, n)
    elif name == "financially_at_risk":
        attendance = _clip(np.random.normal(0.78, 0.12, n), 0.35, 1.0)
        missing = _clip(np.random.beta(3, 6, n), 0.06, 0.70)
        avg = _clip(np.random.normal(6.3, 1.2, n), 2.8, 9.2)
        overdue = np.random.binomial(1, 0.85, n)
    else:
        attendance = _clip(np.random.normal(0.50, 0.16, n), 0.05, 0.88)
        missing = _clip(np.random.beta(6, 3, n), 0.28, 0.98)
        avg = _clip(np.random.normal(4.5, 1.4, n), 0.5, 8.0)
        overdue = np.random.binomial(1, 0.35, n)

    held = np.random.randint(4, 31, n)
    unexcused_count = np.rint((1 - attendance) * held * np.random.uniform(0.45, 0.9, n)).astype(int)
    unexcused_rate = _clip(unexcused_count / held, 0, 1)
    late_count = np.random.poisson(np.maximum(0.1, (1 - attendance) * 3), n).astype(int)
    consecutive = np.minimum(unexcused_count, np.random.poisson(unexcused_count / 2 + 0.2, n)).astype(int)
    last_attended = np.where(attendance > 0.9, np.random.randint(0, 4, n), np.random.randint(1, 21, n))
    assignment_late = np.random.poisson(missing * 3, n).astype(int)
    trend = np.random.normal(0, 0.5, n) - (missing * 1.5) - ((1 - attendance) * 1.2)
    days_overdue = np.where(overdue == 1, np.random.randint(1, 46, n), 0)
    progress = _clip(np.random.beta(3, 3, n), 0.05, 0.98)

    return {
        "Attendance_Rate": np.round(attendance, 4),
        "Unexcused_Absence_Count": unexcused_count,
        "Unexcused_Absence_Rate": np.round(unexcused_rate, 4),
        "Late_Count": late_count,
        "Consecutive_Unexcused_Absences": consecutive,
        "Days_Since_Last_Attended": last_attended,
        "Assignment_Missing_Rate": np.round(missing, 4),
        "Assignment_Late_Count": assignment_late,
        "Average_Score": np.round(avg, 2),
        "Score_Trend": np.round(trend, 2),
        "Has_Overdue_Invoice": overdue,
        "Days_Overdue": days_overdue,
        "Class_Progress_Ratio": np.round(progress, 4),
    }


def generate(total: int = TOTAL_RECORDS) -> pd.DataFrame:
    np.random.seed(RANDOM_SEED)
    frames = []
    for name, ratio, base_prob in PERSONAS:
        n = int(total * ratio)
        data = _persona_features(name, n)
        risk = np.full(n, base_prob)
        risk += np.where(data["Attendance_Rate"] < 0.65, 0.18, 0)
        risk += np.where(data["Unexcused_Absence_Count"] >= 3, 0.18, 0)
        risk += np.where(data["Consecutive_Unexcused_Absences"] >= 2, 0.20, 0)
        risk += np.where(data["Days_Since_Last_Attended"] >= 10, 0.12, 0)
        risk += np.where(data["Assignment_Missing_Rate"] >= 0.45, 0.16, 0)
        risk += np.where(data["Average_Score"] < 5.0, 0.10, 0)
        risk += np.where(data["Score_Trend"] <= -1.2, 0.08, 0)
        risk += np.where(data["Has_Overdue_Invoice"] == 1, 0.08, 0)
        risk += np.random.normal(0, 0.04, n)
        risk = _clip(risk, 0.01, 0.98)
        data["Is_Dropout"] = (np.random.rand(n) < risk).astype(int)
        data["_Persona"] = name
        frames.append(pd.DataFrame(data))

    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def main() -> None:
    df = generate()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.drop(columns=["_Persona"]).to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.groupby("_Persona")["Is_Dropout"].mean().round(3))


if __name__ == "__main__":
    main()
