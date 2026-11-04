"""Inference contract checks; no database or notifications are used."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from dropout_inference import ACADEMIC_FEATURES, FEATURES, load_dropout_bundle, score_dropout


class DropoutInferenceTests(unittest.TestCase):
    def bundle(self):
        frame = pd.DataFrame([[7, 0.5, 0.2, 1, 0], [8, 1, 0, 0, 1]], columns=FEATURES)
        classifier = Pipeline([('imputer', SimpleImputer()), ('classifier', DummyClassifier(strategy='prior'))]).fit(frame, [0, 1])
        return {'pipeline': classifier, 'features': FEATURES, 'variant': 'uci_academic_finance', 'threshold': 0.3}

    def test_uses_exported_threshold_instead_of_legacy_seventy_percent(self):
        result = score_dropout(self.bundle(), dict(zip(FEATURES, [None, 0, 0.5, 1, 0])))
        self.assertTrue(result['model_flagged'])
        self.assertEqual(result['risk_score'], 50)
        self.assertEqual(result['score_type'], 'source_model_score')
        self.assertTrue(result['includes_finance'])

    def test_rejects_legacy_columns_and_impossible_ratios(self):
        with self.assertRaises(ValueError):
            score_dropout(self.bundle(), {'Attendance_Rate': 0.5})
        with self.assertRaises(ValueError):
            score_dropout(self.bundle(), dict(zip(FEATURES, [7, 0.8, 0.8, 0, 1])))
        for value in [None, 0.5, -1, 2, float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                score_dropout(self.bundle(), dict(zip(FEATURES, [7, 0.5, 0, value, 1])))

    def test_serialized_contract_and_real_trained_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pkl'
            bundle = self.bundle()
            joblib.dump(bundle, path)
            self.assertEqual(load_dropout_bundle(path)['features'], FEATURES)
            bundle['variant'] = 'uci_academic'
            joblib.dump(bundle, path)
            with self.assertRaises(ValueError):
                load_dropout_bundle(path)
        path = Path(__file__).parent / 'ai_models/dropout_public/uci_academic_finance_bundle.pkl'
        actual = load_dropout_bundle(path)
        self.assertEqual(list(actual['pipeline'].feature_names_in_), FEATURES)

    def test_api_startup_and_prediction_with_real_trained_model(self):
        import api_server
        api_server._load_models()
        self.assertEqual(api_server._models['dropout_risk_version'], 'uci_academic_finance_public_1')
        result = api_server.predict_dropout(api_server.DropoutFeatures(
            Average_Score=7, Pass_Rate=0.5, Not_Evaluated_Rate=0.25, Has_Debt=1, Tuition_Up_To_Date=0))
        self.assertEqual(result['model_version'], 'uci_academic_finance_public_1')
        self.assertTrue(result['includes_finance'])
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 100)
        with self.assertRaises(ValueError):
            api_server.DropoutFeatures(Average_Score=7, Pass_Rate=1, Not_Evaluated_Rate=0, Attendance_Rate=1)

    def test_unknown_or_partial_tuition_uses_academic_model_without_imputation(self):
        import api_server
        api_server._load_models()
        for flags in [{}, {'Has_Debt': None, 'Tuition_Up_To_Date': None},
                      {'Has_Debt': 1, 'Tuition_Up_To_Date': None}]:
            values = dict(Average_Score=7, Pass_Rate=0.5, Not_Evaluated_Rate=0.25, **flags)
            result = api_server.predict_dropout(api_server.DropoutFeatures(**values))
            self.assertEqual(result['model_version'], 'uci_academic_public_1')
            self.assertFalse(result['includes_finance'])
            expected = score_dropout(api_server._models['dropout_academic_fallback'],
                                     {key: values[key] for key in ACADEMIC_FEATURES})
            self.assertEqual(result, expected)

    def test_real_finance_model_responds_to_tuition_at_fixed_academic_scores(self):
        bundle = load_dropout_bundle(Path(__file__).parent / 'ai_models/dropout_public/uci_academic_finance_bundle.pkl')
        current = dict(zip(FEATURES, [7, 0.5, 0.25, 0, 1]))
        overdue = {**current, 'Has_Debt': 1, 'Tuition_Up_To_Date': 0}
        self.assertGreater(score_dropout(bundle, overdue)['risk_score'], score_dropout(bundle, current)['risk_score'])

    def test_missing_finance_artifact_does_not_silently_discard_observed_tuition(self):
        import api_server
        from fastapi import HTTPException
        with patch.dict(api_server._models, dropout_risk=None):
            with self.assertRaises(HTTPException) as error:
                api_server.predict_dropout(api_server.DropoutFeatures(
                    Average_Score=7, Pass_Rate=1, Not_Evaluated_Rate=0, Has_Debt=0, Tuition_Up_To_Date=1))
            self.assertEqual(error.exception.status_code, 503)

    def test_api_rejects_nonbinary_tuition_flags(self):
        import api_server
        for value in [-1, 0.5, 2]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                api_server.DropoutFeatures(Average_Score=7, Pass_Rate=1, Not_Evaluated_Rate=0,
                                          Has_Debt=value, Tuition_Up_To_Date=1)


if __name__ == '__main__':
    unittest.main()
