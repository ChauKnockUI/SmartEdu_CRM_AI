"""Source-specific public dropout CSVs. Never synthesize attendance or tuition."""
from pathlib import Path

import numpy as np
import pandas as pd

from dropout_public_common import (
    SCHEMA_VERSION, TARGET_COL, FINANCE_FEATURES, OULAD_FEATURES,
)

KEYS = ["code_module", "code_presentation", "id_student"]
CUTOFF_DAY = 30


def observed_range(series, lo, hi, audit, name):
    """Invalid observed values become missing with an audit count, never clipped."""
    values = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & (values.isna() | ~values.between(lo, hi))
    audit[name] = int(bad.sum())
    return values.mask(bad)


def validate_dataset(df, source):
    features = FINANCE_FEATURES if source == "UCI" else OULAD_FEATURES
    required = features + ["record_id", "student_id", "Data_Source", "schema_version",
                           "observation_milestone", TARGET_COL]
    if set(required).difference(df.columns):
        raise ValueError(f"Missing columns for {source}: {sorted(set(required).difference(df.columns))}")
    if df.empty or df.record_id.isna().any() or df.student_id.isna().any() or df.record_id.duplicated().any():
        raise ValueError("Enrollment identifiers must be present and unique")
    if not df.Data_Source.eq(source).all() or not df.schema_version.eq(SCHEMA_VERSION).all():
        raise ValueError("Source/schema mismatch")
    if not df[TARGET_COL].isin([0, 1]).all():
        raise ValueError("Missing or nonbinary target")
    if np.isinf(df[features].to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values")
    limits = {"Average_Score": (0, 10), "Pass_Rate": (0, 1), "Not_Evaluated_Rate": (0, 1),
              "Assignment_Missing_Rate": (0, 1), "Score_Trend": (-10, 10),
              "Has_Debt": (0, 1), "Tuition_Up_To_Date": (0, 1)}
    for col in features:
        values = df[col].dropna()
        if col in limits and not values.between(*limits[col]).all():
            raise ValueError(f"Out-of-range feature: {col}")
        if col in ["Has_Debt", "Tuition_Up_To_Date"] and not values.isin([0, 1]).all():
            raise ValueError(f"Invalid binary flag: {col}")
        if col == "Assignment_Late_Count" and ((values < 0) | (values % 1 != 0)).any():
            raise ValueError("Invalid submission count")
    if source == "OULAD":
        if not set(KEYS + ["observation_day"]).issubset(df.columns):
            raise ValueError("OULAD enrollment keys/cutoff missing")
        if df[KEYS].isna().any().any() or df.duplicated(KEYS).any():
            raise ValueError("OULAD enrollment keys must be unique")
        if not df.observation_day.eq(CUTOFF_DAY).all():
            raise ValueError("Unexpected OULAD cutoff")


def process_uci(path):
    raw = pd.read_csv(path, sep=";")
    raw.columns = raw.columns.str.strip()
    audit = {"raw_rows": len(raw), "raw_identical_rows": int(raw.duplicated().sum())}
    raw["source_row_id"] = np.arange(len(raw))
    resolved = raw.Target.isin(["Dropout", "Graduate"])
    audit["excluded_unresolved_target"] = int((~resolved).sum())
    df = raw[resolved].copy()
    enrolled = pd.to_numeric(df["Curricular units 1st sem (enrolled)"], errors="coerce")
    keep = enrolled.notna() & (enrolled > 0) & (enrolled % 1 == 0)
    audit["excluded_invalid_or_zero_enrollment"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    enrolled = enrolled[keep].reset_index(drop=True)
    approved = pd.to_numeric(df["Curricular units 1st sem (approved)"], errors="coerce")
    without = pd.to_numeric(df["Curricular units 1st sem (without evaluations)"], errors="coerce")
    invalid_approved = approved.notna() & ((approved < 0) | (approved > enrolled) | (approved % 1 != 0))
    invalid_without = without.notna() & ((without < 0) | (without > enrolled) | (without % 1 != 0))
    audit["invalid_approved_counts"] = int(invalid_approved.sum())
    audit["invalid_without_evaluation_counts"] = int(invalid_without.sum())
    approved = approved.mask(invalid_approved)
    without = without.mask(invalid_without)
    grade = observed_range(df["Curricular units 1st sem (grade)"], 0, 20, audit, "invalid_grade")
    out = pd.DataFrame(index=df.index)
    # UCI aggregate grade is unavailable when there are no approved units.
    out["Average_Score"] = (grade / 2).where(approved > 0)
    out["Pass_Rate"] = approved / enrolled
    out["Not_Evaluated_Rate"] = without / enrolled
    for source_col, target_col in [("Debtor", "Has_Debt"), ("Tuition fees up to date", "Tuition_Up_To_Date")]:
        values = pd.to_numeric(df[source_col], errors="coerce")
        audit[f"invalid_{target_col}"] = int((df[source_col].notna() & ~values.isin([0, 1])).sum())
        out[target_col] = values.where(values.isin([0, 1]))
    out["record_id"] = "UCI_" + df.source_row_id.astype(str)
    out["student_id"] = out.record_id  # UCI has no original student identifier.
    out["Data_Source"] = "UCI"
    out["observation_milestone"] = "end_semester_1"
    out["schema_version"] = SCHEMA_VERSION
    out[TARGET_COL] = (df.Target == "Dropout").astype(int)
    audit["output_rows"] = len(out)
    validate_dataset(out, "UCI")
    return out, audit


def process_oulad(directory):
    directory = Path(directory)
    info = pd.read_csv(directory / "studentInfo.csv")
    reg = pd.read_csv(directory / "studentRegistration.csv")
    audit = {"raw_rows": len(info)}
    df = info.merge(reg[KEYS + ["date_registration", "date_unregistration"]],
                    on=KEYS, how="left", validate="one_to_one", indicator=True)
    audit["unmatched_registration"] = int(df._merge.ne("both").sum())
    registered = df.date_registration.notna() & (df.date_registration <= CUTOFF_DAY)
    active = df.date_unregistration.isna() | (df.date_unregistration > CUTOFF_DAY)
    audit["excluded_not_registered_by_cutoff"] = int((~registered).sum())
    audit["excluded_withdrawn_by_cutoff"] = int((registered & ~active).sum())
    df = df[registered & active].copy()
    if not df.final_result.isin(["Withdrawn", "Pass", "Distinction", "Fail"]).all():
        raise ValueError("Unknown OULAD final result")
    df[TARGET_COL] = df.final_result.eq("Withdrawn").astype(int)
    asm = pd.read_csv(directory / "assessments.csv")
    asm = asm[["code_module", "code_presentation", "id_assessment", "date"]].rename(columns={"date": "date_due"})
    due = asm[asm.date_due <= CUTOFF_DAY].groupby(KEYS[:2]).size().rename("due_count").reset_index()
    sa = pd.read_csv(directory / "studentAssessment.csv")
    if sa.duplicated(["id_assessment", "id_student"]).any():
        raise ValueError("Duplicate student-assessment records")
    sa["score"] = observed_range(sa.score, 0, 100, audit, "invalid_score")
    sa = sa.merge(asm, on="id_assessment", how="left", validate="many_to_one", indicator=True)
    if sa._merge.ne("both").any():
        raise ValueError("Assessment metadata missing")
    audit["excluded_submissions_after_cutoff"] = int((sa.date_submitted > CUTOFF_DAY).sum())
    sa = sa[sa.date_submitted <= CUTOFF_DAY].merge(df[KEYS], on=KEYS, how="inner", validate="many_to_one")
    sa = sa.sort_values(["date_submitted", "id_assessment"], kind="stable")
    sa["late"] = (sa.date_submitted > sa.date_due).astype(int)
    sa["passed"] = (sa.score >= 40).astype(float).where(sa.score.notna())
    g = sa.groupby(KEYS)
    agg = g.agg(Average_Score=("score", "mean"), Pass_Rate=("passed", "mean"),
                Assignment_Late_Count=("late", "sum"), first_score=("score", "first"),
                last_score=("score", "last"), scored_count=("score", "count"))
    agg["Average_Score"] /= 10
    agg["Score_Trend"] = ((agg.last_score - agg.first_score) / 10).where(agg.scored_count >= 2)
    agg["submitted_due_count"] = sa[sa.date_due <= CUTOFF_DAY].groupby(KEYS).size()
    df = df[KEYS + [TARGET_COL]].merge(agg.reset_index(), on=KEYS, how="left", validate="one_to_one")
    df = df.merge(due, on=KEYS[:2], how="left", validate="many_to_one")
    df["due_count"] = df.due_count.fillna(0)
    df["submitted_due_count"] = df.submitted_due_count.fillna(0)
    if (df.submitted_due_count > df.due_count).any():
        raise ValueError("Submitted due count exceeds due assessments")
    df["Assignment_Missing_Rate"] = 1 - df.submitted_due_count / df.due_count.replace(0, np.nan)
    df["Assignment_Late_Count"] = df.Assignment_Late_Count.fillna(0)
    df["student_id"] = df.id_student.astype(str)
    df["record_id"] = "OULAD_" + df[KEYS].astype(str).agg("_".join, axis=1)
    df["Data_Source"] = "OULAD"
    df["observation_day"] = CUTOFF_DAY
    df["observation_milestone"] = "day_30"
    df["schema_version"] = SCHEMA_VERSION
    out = df[OULAD_FEATURES + ["record_id", "student_id", "Data_Source", "observation_day",
                              "observation_milestone", "schema_version", TARGET_COL] + KEYS].reset_index(drop=True)
    audit["output_rows"] = len(out)
    validate_dataset(out, "OULAD")
    return out, audit
