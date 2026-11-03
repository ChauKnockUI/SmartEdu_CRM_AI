"""Inference contract checks; no database or notifications are used."""
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from dropout_inference import FEATURES, load_dropout_bundle, score_dropout


class DropoutInferenceTests(unittest.TestCase):
    def bundle(self):
        frame = pd.DataFrame([[7, 0.5, 0.2], [8, 1, 0]], columns=FEATURES)
        classifier = Pipeline([('imputer', SimpleImputer()), ('classifier', DummyClassifier(strategy='prior'))]).fit(frame, [0, 1])
        return {'pipeline': classifier, 'features': FEATURES, 'variant': 'uci_academic', 'threshold': 0.3}

    def test_uses_exported_threshold_instead_of_legacy_seventy_percent(self):
        result = score_dropout(self.bundle(), dict(zip(FEATURES, [None, 0, 0.5])))
        self.assertTrue(result['model_flagged'])
        self.assertEqual(result['risk_score'], 50)
        self.assertEqual(result['score_type'], 'source_model_score')

    def test_rejects_legacy_columns_and_impossible_ratios(self):
        with self.assertRaises(ValueError):
            score_dropout(self.bundle(), {'Attendance_Rate': 0.5})
        with self.assertRaises(ValueError):
            score_dropout(self.bundle(), dict(zip(FEATURES, [7, 0.8, 0.8])))

    def test_serialized_contract_and_real_trained_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pkl'
            bundle = self.bundle()
            joblib.dump(bundle, path)
            self.assertEqual(load_dropout_bundle(path)['features'], FEATURES)
            bundle['variant'] = 'uci_academic_finance'
            joblib.dump(bundle, path)
            with self.assertRaises(ValueError):
                load_dropout_bundle(path)
        path = Path(__file__).parent / 'ai_models/dropout_public/uci_academic_bundle.pkl'
        actual = load_dropout_bundle(path)
        self.assertEqual(list(actual['pipeline'].feature_names_in_), FEATURES)

    def test_api_startup_and_prediction_with_real_trained_model(self):
        import api_server
        api_server._load_models()
        self.assertEqual(api_server._models['dropout_risk_version'], 'uci_academic_public_1')
        result = api_server.predict_dropout(api_server.DropoutFeatures(
            Average_Score=7, Pass_Rate=0.5, Not_Evaluated_Rate=0.25))
        self.assertEqual(result['model_version'], 'uci_academic_public_1')
        self.assertGreaterEqual(result['risk_score'], 0)
        self.assertLessEqual(result['risk_score'], 100)
        with self.assertRaises(ValueError):
            api_server.DropoutFeatures(Average_Score=7, Pass_Rate=1, Not_Evaluated_Rate=0, Attendance_Rate=1)


if __name__ == '__main__':
    unittest.main()
