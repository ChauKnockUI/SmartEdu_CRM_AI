import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "ai_models")]
from lead_contract import CRM_FEATURES, CRM_MODEL_VERSION, engineer_crm_features, get_lead_prediction_json
from train_crm_lead_scoring import load_snapshots, temporal_partitions, build_model, metrics, main
from explain_lead_scoring import get_lead_explanation_json


def fixture():
    rows = []
    for lead in range(120):
        created = pd.Timestamp('2025-01-01', tz='UTC') + pd.Timedelta(days=lead * 3)
        for day in [0, 1, 2]:
            rows.append({ 'lead_key': f'42:{lead}', 'snapshot_at': created + pd.Timedelta(days=day),
                'lead_created_at': created, 'observed_until': pd.Timestamp('2027-01-01', tz='UTC'),
                'outcome_window_days': 30, 'converted_30d': lead % 2, 'eligible_for_training': True, 'outcome_status': 'observed',
                'Lead_Source': 'google_ads', 'Occupation': 'student_y1_y2' if lead % 2 else 'working_professional',
                'Study_Purpose': 'pass_exam', 'Course_Interested': 'IELTS 6.0', 'Last_Engagement_Status': 'unknown',
                'Call_Attempt_Count': day, 'Days_Since_Created': day, 'Answered_Call_Count': 0,
                'Activity_Count_7d': day, 'Days_Since_Last_Activity': None })
    return pd.DataFrame(rows)


class CrmTrainingTests(unittest.TestCase):
    def test_training_saves_candidate_but_refuses_premature_activation(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'export.json'
            path.write_text(json.dumps({'dataset_version': 'lead_snapshot_1', 'rows': json.loads(data.to_json(orient='records', date_format='iso'))}), encoding='utf-8')
            with patch.object(sys, 'argv', ['train', '--dataset', str(path), '--output-dir', str(root), '--activate']), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit, 'activation refused'):
                    main()
            self.assertTrue((root / 'lead_scoring_crm_candidate.pkl').exists())
            self.assertTrue((root / 'lead_scoring_crm_report.json').exists())
            self.assertFalse((root / 'lead_scoring_crm_pipeline.pkl').exists())

    def test_many_hot_snapshots_do_not_count_as_many_distinct_leads(self):
        result = metrics(np.array([1] * 25 + [0] * 5), np.array([.9] * 25 + [.1] * 5), ['same'] * 25 + ['other'] * 5)
        self.assertEqual(result['hot_count'], 25)
        self.assertEqual(result['hot_leads'], 1)

    def test_partitions_disjoint_by_identity_and_purge_future_outcome_windows(self):
        train, validation, test = temporal_partitions(fixture())
        groups = [set(part.lead_key) for part in [train, validation, test]]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertLessEqual((train.snapshot_at + pd.Timedelta(days=30)).max(), validation.snapshot_at.min())
        self.assertLessEqual((validation.snapshot_at + pd.Timedelta(days=30)).max(), test.snapshot_at.min())

    def test_loader_rejects_incomplete_windows_even_if_labeled_observed(self):
        data = fixture().iloc[:1].copy()
        data['observed_until'] = data.snapshot_at + pd.Timedelta(days=2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'export.json'
            path.write_text(json.dumps({'dataset_version': 'lead_snapshot_1', 'rows': json.loads(data.to_json(orient='records', date_format='iso'))}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                load_snapshots(path)

    def test_loader_excludes_pending_and_caps_same_day_repeats(self):
        data = fixture().iloc[:2].copy()
        duplicate = data.iloc[[0]].copy()
        duplicate['snapshot_at'] += pd.Timedelta(hours=1)
        pending = data.iloc[[0]].copy()
        pending['eligible_for_training'] = False
        pending['outcome_status'] = 'pending'
        pending['converted_30d'] = None
        data = pd.concat([data, duplicate, pending])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'export.json'
            path.write_text(json.dumps({'dataset_version': 'lead_snapshot_1', 'rows': json.loads(data.to_json(orient='records', date_format='iso'))}), encoding='utf-8')
            self.assertEqual(len(load_snapshots(path)), 1)

    def test_crm_training_excludes_initial_snapshots_even_when_labels_are_mature(self):
        data = fixture().iloc[:3].copy()
        data['Has_Contact_Activity'] = [False, True, True]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'export.json'
            path.write_text(json.dumps({'dataset_version': 'lead_snapshot_1', 'rows': json.loads(data.to_json(orient='records', date_format='iso'))}), encoding='utf-8')
            selected = load_snapshots(path)
            self.assertEqual(len(selected), 2)
            self.assertTrue(selected.Has_Contact_Activity.all())

    def test_calibration_never_splits_one_lead_between_training_and_calibration(self):
        train, _, _ = temporal_partitions(fixture())
        pipeline = build_model(train, 6)
        for tr, cal in pipeline.named_steps['classifier'].cv:
            self.assertFalse(set(train.iloc[tr].lead_key) & set(train.iloc[cal].lead_key))

    def test_real_crm_model_handles_missing_history_and_numeric_shap(self):
        train, _, test = temporal_partitions(fixture())
        model = build_model(train, 6).fit(train[CRM_FEATURES], train.converted_30d)
        engineered = engineer_crm_features(train)
        model.lead_metadata_ = {'model_version': CRM_MODEL_VERSION, 'active_features': CRM_FEATURES,
            'observed_categories': {name: engineered[name].dropna().unique().tolist() for name in CRM_FEATURES[:5]}}
        row = test.iloc[0][CRM_FEATURES].to_dict()
        predicted = get_lead_prediction_json(model, row)
        self.assertEqual(predicted['prediction_horizon_days'], 30)
        self.assertEqual(predicted['score_type'], 'crm_conversion_probability')
        self.assertIsNone(predicted['engineered_features']['Days_Since_Last_Activity'])
        explained = get_lead_explanation_json(model, row)
        self.assertEqual(explained['explanation_status'], 'available')
        details = explained['explanation_details']
        self.assertAlmostEqual(details['base_probability'] + sum(details['contributions'].values()), details['uncalibrated_probability'], places=5)
        self.assertTrue(np.isfinite(predicted['conversion_probability']))


if __name__ == '__main__':
    unittest.main()
