"""Regression checks for outcome independence, serving parity and failure paths."""
import hashlib
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ai_models")]
import api_server
import explain_lead_scoring as explain
import train_real_lead_scoring_rf as training
from lead_contract import (CORE_FEATURES, FEATURES, MODEL_VERSION, engineer_lead_features,
                           engineer_source_features, prediction_fields)

CRM = {"Lead_Source": "google_ads", "Occupation": "student_y1_y2",
       "Study_Purpose": "career", "Course_Interested": "ielts",
       "Call_Attempt_Count": 3, "Last_Engagement_Status": "interested_need_time",
       "Days_Since_Created": 5}


def source_fixture():
    return pd.DataFrame({
        "Lead Source": ["Google", "Olark Chat", None, "Facebook"],
        "What is your current occupation": ["Student", None, "Unemployed", "Select"],
        "What matters most to you in choosing a course": ["Better Career Prospects", None, None, None],
        "Last Activity": ["Email Opened", "SMS Sent", None, "Unreachable"],
        "Last Notable Activity": ["Modified"] * 4,
        "TotalVisits": [3, 0, np.nan, 7], "Converted": [1, 0, 1, 0],
        "Tags": ["Closed by Sales", "Lost to competitor", "Other", "Ringing"],
    })


class DataSafetyTests(unittest.TestCase):
    def test_features_do_not_depend_on_labels_tags_or_unsupported_columns(self):
        raw = source_fixture()
        expected = engineer_source_features(raw)
        raw["Converted"] = 1 - raw["Converted"]
        raw["Tags"] = "Converted"
        raw["Specialization"] = "Finance Management"
        raw["Total Time Spent on Website"] = 100000
        raw["TotalVisits"] = 99999
        pd.testing.assert_frame_equal(expected, engineer_source_features(raw))
        pd.testing.assert_frame_equal(expected, engineer_source_features(raw.drop(columns=["Converted", "Tags"])))

    def test_public_and_crm_observations_reach_the_same_feature_space(self):
        public = engineer_source_features(source_fixture()).iloc[[0]].reset_index(drop=True)
        crm = engineer_lead_features(pd.DataFrame([CRM]))
        pd.testing.assert_frame_equal(public, crm)
        changed = {**CRM, "Course_Interested": "backend_nodejs", "Days_Since_Created": 1000}
        pd.testing.assert_frame_equal(crm, engineer_lead_features(pd.DataFrame([changed])))

    def test_unknown_is_not_invented_negative_evidence(self):
        data = engineer_source_features(source_fixture())
        self.assertEqual(data.loc[1, "Occupation_Group"], "unknown")
        self.assertEqual(data.loc[1, "Engagement_Group"], "unknown")
        self.assertEqual(data.loc[1, "Source_Group"], "website")
        self.assertNotIn("Interaction_Band", data.columns)
        self.assertEqual(data.loc[3, "Engagement_Group"], "unreachable")

    def test_mapping_is_deterministic_and_rejects_duplicate_lead_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            raw = source_fixture()
            raw.to_csv(path, index=False)
            first = training.map_and_clean_real_data(path)
            raw["Converted"] = 1 - raw["Converted"]
            raw.to_csv(path, index=False)
            second = training.map_and_clean_real_data(path)
            pd.testing.assert_frame_equal(first[FEATURES], second[FEATURES])
            raw["Prospect ID"] = ["same", "same", "third", "fourth"]
            raw.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                training.map_and_clean_real_data(path)

    def test_feature_selection_prefers_simpler_model_for_negligible_gain(self):
        result = {name: {"mean": {"roc_auc": score}} for name, score in
                  [("intake", .75), ("engagement", .78), ("interaction", .783)]}
        self.assertEqual(training.select_candidate(result), "engagement")

    def test_cross_validation_never_receives_final_test_rows(self):
        data = pd.concat([engineer_source_features(source_fixture())] * 10, ignore_index=True)
        data["Converted"] = [0, 1] * 20
        train, test = train_test_split(data, test_size=.2, random_state=42, stratify=data.Converted)
        fake = SimpleNamespace()
        fake.fit = Mock(return_value=fake)
        fake.predict_proba = lambda X: np.tile([.6, .4], (len(X), 1))
        observed = []

        def cv_stub(model, X, y, **kwargs):
            observed.append(set(X.index))
            return {f"test_{name}": np.full(5, .7) for name in kwargs["scoring"]}

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with patch.object(training, "map_and_clean_real_data", return_value=data), \
                 patch.object(training, "cross_validate", side_effect=cv_stub), \
                 patch.object(training, "build_rf_pipeline", return_value=fake), \
                 patch.object(training, "REAL_DATASET_PATH", target / "mapped.csv"), \
                 patch.object(training, "OUTPUT_MODEL_PATH", target / "model.pkl"), \
                 patch.object(training, "RESULT_JSON_PATH", target / "results.json"), \
                 patch.object(training, "REPORT_PATH", target / "report.md"), \
                 patch.object(training.joblib, "dump", side_effect=lambda model, path: Path(path).write_bytes(b"test artifact")):
                training.main()
        self.assertEqual(len(observed), 2)
        for indices in observed:
            self.assertEqual(indices, set(train.index))
            self.assertFalse(indices & set(test.index))


class ApiPolicyTests(unittest.TestCase):
    def test_threshold_boundaries(self):
        for probability, label, prediction in [(.3, "COLD_LEAD", "Not Converted"),
            (.499999, "COLD_LEAD", "Not Converted"), (.5, "WARM_LEAD", "Converted"),
            (.799999, "WARM_LEAD", "Converted"), (.8, "HOT_LEAD", "Converted")]:
            with self.subTest(probability=probability):
                result = prediction_fields(probability)
                self.assertEqual((result["recommendation"], result["prediction"]), (label, prediction))
        for value in [float("nan"), float("inf"), -1, 2]:
            with self.assertRaises(ValueError):
                prediction_fields(value)

    def test_predict_and_explain_agree_even_if_shap_is_missing(self):
        for probability in [.33, .5, .8]:
            fake = SimpleNamespace(lead_metadata_={"model_version": MODEL_VERSION, "active_features": FEATURES},
                predict_proba=lambda X: np.array([[1 - probability, probability]]))
            features = api_server.LeadFeaturesV2(**CRM)
            # Patch the function in the module actually imported by the API.
            with patch.dict(api_server._models, lead_scoring_v2=fake), \
                 patch("ai_models.explain_lead_scoring.forest_ensemble_impacts", side_effect=ImportError("shap")):
                predicted = api_server.predict_lead_v2(features)
                explained = api_server.explain_lead_v2(features)
            for name, value in predicted.items():
                self.assertEqual(value, explained[name])
            self.assertEqual(explained["explanation_status"], "unavailable")
            self.assertEqual(explained["explanation"], {"positive_factors": [], "negative_factors": []})

    def test_unavailable_model_is_a_503(self):
        with patch.dict(api_server._models, lead_scoring_v2=None):
            for endpoint in [api_server.predict_lead_v2, api_server.explain_lead_v2]:
                with self.assertRaises(api_server.HTTPException) as raised:
                    endpoint(api_server.LeadFeaturesV2(**CRM))
                self.assertEqual(raised.exception.status_code, 503)

    def test_binary_shap_uses_positive_class_without_flattening_classes(self):
        values = np.array([[[-.1, .1], [.2, -.2], [-.3, .3]]])
        np.testing.assert_allclose(explain.positive_class_shap(values, 3), [.1, -.2, .3])
        old_values = [values[:, :, 0], values[:, :, 1]]
        np.testing.assert_allclose(explain.positive_class_shap(old_values, 3), [.1, -.2, .3])


class SavedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = joblib.load(training.OUTPUT_MODEL_PATH)

    def test_artifact_contract_and_report_fingerprints_match(self):
        report = json.loads(training.RESULT_JSON_PATH.read_text(encoding="utf-8"))
        self.assertEqual(self.pipeline.lead_metadata_["model_version"], MODEL_VERSION)
        self.assertEqual(hashlib.sha256(training.OUTPUT_MODEL_PATH.read_bytes()).hexdigest(), report["model_sha256"])
        self.assertEqual(hashlib.sha256(training.REAL_DATASET_PATH.read_bytes()).hexdigest(), report["mapped_data_sha256"])
        for model in self.pipeline.stages_.values():
            for fold in model.named_steps["classifier"].calibrated_classifiers_:
                self.assertIn("preprocessor", fold.estimator.named_steps)

    def test_saved_pipeline_predicts_identically_for_raw_and_engineered_crm(self):
        raw = pd.DataFrame([CRM])
        engineered = engineer_lead_features(raw)
        np.testing.assert_allclose(self.pipeline.predict_proba(raw), self.pipeline.predict_proba(engineered))

    def test_real_shap_explains_all_forests_additively(self):
        result = explain.get_lead_explanation_json(self.pipeline, CRM)
        self.assertEqual(result["explanation_status"], "available")
        details = result["explanation_details"]
        self.assertAlmostEqual(details["base_probability"] + sum(details["contributions"].values()),
                               details["uncalibrated_probability"], places=5)

    def test_initial_score_and_explanation_use_profile_only(self):
        new = {**CRM, "Has_Contact_Activity": False, "Call_Attempt_Count": 0, "Last_Engagement_Status": "unknown"}
        result = explain.get_lead_explanation_json(self.pipeline, new)
        self.assertEqual(result["scoring_stage"], "initial")
        self.assertEqual(result["assessment_basis"], "profile_only")
        self.assertEqual(result["active_features"], CORE_FEATURES)
        self.assertEqual(set(result["explanation_details"]["contributions"]), set(CORE_FEATURES))
        changed = {**new, "Call_Attempt_Count": 100, "Last_Engagement_Status": "rejected"}
        np.testing.assert_allclose(self.pipeline.predict_proba(pd.DataFrame([new])), self.pipeline.predict_proba(pd.DataFrame([changed])))

    def test_contact_without_result_retains_intake_estimate(self):
        from lead_contract import get_lead_prediction_json
        new = {**CRM, "Has_Contact_Activity": False, "Call_Attempt_Count": 0, "Last_Engagement_Status": "unknown"}
        contacted = {**new, "Has_Contact_Activity": True, "Call_Attempt_Count": 1}
        initial = get_lead_prediction_json(self.pipeline, new)
        result = get_lead_prediction_json(self.pipeline, contacted)
        self.assertEqual(result["scoring_stage"], "post_contact")
        self.assertEqual(result["assessment_basis"], "response_pending")
        self.assertEqual(result["conversion_probability"], initial["conversion_probability"])

    def test_post_contact_does_not_treat_call_count_as_visits(self):
        from lead_contract import get_lead_prediction_json
        contacted = {**CRM, "Has_Contact_Activity": True, "Call_Attempt_Count": 0,
                     "Activity_Count_7d": 0, "Days_Since_Last_Activity": 60}
        result = get_lead_prediction_json(self.pipeline, contacted)
        self.assertEqual(result["scoring_stage"], "post_contact")
        self.assertIn("Engagement_Group", result["active_features"])
        self.assertNotIn("Interaction_Band", result["engineered_features"])
        many_calls = {**contacted, "Call_Attempt_Count": 50}
        np.testing.assert_allclose(self.pipeline.predict_proba(pd.DataFrame([contacted])), self.pipeline.predict_proba(pd.DataFrame([many_calls])))

    def test_future_crm_history_model_never_replaces_initial_profile_model(self):
        router = copy.copy(self.pipeline)
        history = object()
        router.crm_model_ = history
        selected, stage, basis = router.select_pipeline({**CRM, "Has_Contact_Activity": False})
        self.assertIs(selected, router.stages_["initial"])
        self.assertEqual((stage, basis), ("initial", "profile_only"))
        selected, stage, basis = router.select_pipeline({**CRM, "Has_Contact_Activity": True})
        self.assertIs(selected, history)
        self.assertEqual((stage, basis), ("post_contact", "crm_history"))


if __name__ == "__main__":
    unittest.main()
