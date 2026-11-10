"""Meaningful regression checks for public-data cleaning and evaluation isolation."""
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dropout_public_common import (
    ACADEMIC_FEATURES, FINANCE_FEATURES, OULAD_FEATURES, SCHEMA_VERSION, TARGET_COL,
    group_holdout, make_cv_splits, make_pipeline, model_zoo, pipeline_signature,
)
from dropout_public_data import process_uci, process_oulad, validate_dataset


class PublicDataTests(unittest.TestCase):
    def test_group_split_keeps_returning_students_together(self):
        df = pd.DataFrame([{"record_id": f"{s}_{t}", "student_id": str(s),
                            "Data_Source": "OULAD", TARGET_COL: s % 2}
                           for s in range(60) for t in [2013, 2014]])
        train, test = group_holdout(df)
        self.assertFalse(set(train.student_id) & set(test.student_id))
        folds = make_cv_splits(train, train[TARGET_COL], train.student_id)
        seen = []
        for fit, valid in folds:
            self.assertFalse(set(train.iloc[fit].student_id) & set(train.iloc[valid].student_id))
            seen.extend(valid)
        self.assertEqual(sorted(seen), list(range(len(train))))

    def test_configuration_guard_survives_pickle(self):
        pipe = make_pipeline(model_zoo()["HistGradientBoosting"], ACADEMIC_FEATURES)
        restored = pickle.loads(pickle.dumps(pipe))
        self.assertEqual(pipeline_signature(pipe), pipeline_signature(restored))
        restored.set_params(prep__num__imp__strategy="mean")
        self.assertNotEqual(pipeline_signature(pipe), pipeline_signature(restored))

    def test_uci_does_not_clip_invalid_values_or_merge_finance_flags(self):
        row = {"Target": "Graduate", "Curricular units 1st sem (enrolled)": 4,
               "Curricular units 1st sem (approved)": 2, "Curricular units 1st sem (grade)": 14,
               "Curricular units 1st sem (without evaluations)": 1,
               "Debtor": 1, "Tuition fees up to date": 1}
        rows = [row, {**row, "Target": "Enrolled"},
                {**row, "Curricular units 1st sem (enrolled)": 0},
                {**row, "Curricular units 1st sem (grade)": 25, "Debtor": 2,
                 "Curricular units 1st sem (without evaluations)": 8}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "uci.csv"
            pd.DataFrame(rows).to_csv(path, sep=";", index=False)
            df, audit = process_uci(path)
        self.assertEqual(df.record_id.tolist(), ["UCI_0", "UCI_3"])
        self.assertEqual(audit["excluded_unresolved_target"], 1)
        self.assertEqual(audit["excluded_invalid_or_zero_enrollment"], 1)
        self.assertEqual(df.iloc[0].Average_Score, 7)
        self.assertEqual(df.iloc[0].Not_Evaluated_Rate, 0.25)
        self.assertEqual(df.iloc[0].Has_Debt, 1)
        self.assertEqual(df.iloc[0].Tuition_Up_To_Date, 1)
        self.assertTrue(df.loc[1, ["Average_Score", "Has_Debt", "Not_Evaluated_Rate"]].isna().all())
        self.assertNotIn("Has_Overdue_Invoice", df.columns)
        self.assertNotIn("Assignment_Missing_Rate", df.columns)

    def test_academic_pipeline_cannot_consume_finance_or_identifiers(self):
        df = pd.DataFrame({"Average_Score": [2, 3, 8, 9], "Pass_Rate": [0, 0.2, 0.8, 1],
                           "Not_Evaluated_Rate": [0.8, 0.5, 0, 0], "Has_Debt": [1, 1, 0, 0],
                           "Tuition_Up_To_Date": [0, 0, 1, 1], "record_id": ["a", "b", "c", "d"]})
        pipe = make_pipeline(model_zoo()["Logistic Regression"], ACADEMIC_FEATURES).fit(df, [1, 1, 0, 0])
        changed = df.assign(Has_Debt=0, Tuition_Up_To_Date=0, record_id="unrelated")
        np.testing.assert_array_equal(pipe.predict_proba(df), pipe.predict_proba(changed))

    def test_oulad_uses_only_observed_assessments_and_needs_no_vle(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            keys = {"code_module": "AAA", "code_presentation": "2013J"}
            pd.DataFrame([{**keys, "id_student": i, "final_result": "Pass"} for i in [1, 2, 3]]).to_csv(p / "studentInfo.csv", index=False)
            pd.DataFrame([{**keys, "id_student": i, "date_registration": 31 if i == 2 else -5,
                           "date_unregistration": 30 if i == 3 else np.nan} for i in [1, 2, 3]]).to_csv(p / "studentRegistration.csv", index=False)
            pd.DataFrame([{**keys, "id_assessment": 1, "date": 20},
                          {**keys, "id_assessment": 2, "date": 40}]).to_csv(p / "assessments.csv", index=False)
            pd.DataFrame([{"id_assessment": 1, "id_student": 1, "date_submitted": 20, "score": 0},
                          {"id_assessment": 2, "id_student": 1, "date_submitted": 31, "score": 100}]).to_csv(p / "studentAssessment.csv", index=False)
            df, _ = process_oulad(p)
            self.assertEqual(df.student_id.tolist(), ["1"])
            self.assertEqual(df.iloc[0].Average_Score, 0)
            self.assertEqual(df.iloc[0].Pass_Rate, 0)
            self.assertEqual(df.iloc[0].Assignment_Missing_Rate, 0)
            self.assertTrue(np.isnan(df.iloc[0].Score_Trend))
            self.assertFalse(set(FINANCE_FEATURES[3:]).intersection(df.columns))
            self.assertFalse(any("VLE" in c or "Activity" in c for c in df))
            future = pd.read_csv(p / "studentAssessment.csv")
            future.loc[future.date_submitted > 30, "score"] = 17
            future.to_csv(p / "studentAssessment.csv", index=False)
            pd.testing.assert_frame_equal(df, process_oulad(p)[0])
            # No due assessments is unknown missing rate, not zero missed work.
            asm = pd.read_csv(p / "assessments.csv")
            asm["date"] = 40
            asm.to_csv(p / "assessments.csv", index=False)
            self.assertTrue(np.isnan(process_oulad(p)[0].iloc[0].Assignment_Missing_Rate))

    def test_reject_duplicate_enrollment(self):
        df = pd.DataFrame([{**dict.fromkeys(FINANCE_FEATURES, np.nan), "record_id": "a", "student_id": "a",
                            "Data_Source": "UCI", "schema_version": SCHEMA_VERSION,
                            "observation_milestone": "end_semester_1", TARGET_COL: 1}] * 2)
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_dataset(df, "UCI")


if __name__ == "__main__":
    unittest.main()
