# Lead scoring and early care

The CRM displays a conversion score and keeps the existing overdue-contact filter.
Visible care suggestions come from the frontend's `leadGuidance.ts` rules,
combining contact evidence, customer responses and a current model score.
The backend retains `lead_care_1` for compatibility and audit history.
Neither rule set estimates how much a call would improve conversion.

## Current model and UI

- `lead_staged_rf_3` is the public-data reference model. Its score has no 30-day
  outcome definition and is shown as a reference estimate.
- Its artifact bundles two separately fitted pipelines. New leads use only
  source, occupation and learning goal. After an actual call/email/SMS/meeting,
  recorded response may be added. An attempt without a response result keeps
  the profile-only score and is displayed as awaiting contact results.
- Neither public pipeline uses call counts or `TotalVisits`. Notes and manual
  pipeline-stage changes do not count as contact. SHAP explains only the
  pipeline actually selected, so the initial explanation contains no interaction factors.
- Source-domain evaluation compares the two pipelines on the same held-out
  split. Public data lacks contact timestamps, so it cannot establish accuracy
  for genuinely new CRM leads. A positive response does not guarantee a score
  increase across two separately trained models.
- The backend computes care using actual lead state, latest recorded response,
  contact age and call attempts. Untouched leads, stale contacts and low scores
  can all require early consultation. Refusal and invalid numbers have separate actions.
- Enrolled/lost leads do not receive acquisition reminders. A low score alone
  never means the lead should be ignored.
- The list and board show the conversion score without care-priority tags.
  The board retains the existing 3-day contact filter, which also includes
  leads with no recorded contact. Cards distinguish these as not yet contacted.
  The detail page shows a next action, a suggested consultation question,
  paginated history and collapsible model explanations. Suggested time limits
  are not displayed as confirmed customer appointments. Care priority remains internal for compatibility;
  its broad rules are not used as a visible tag or a separate board filter.
- Missing AI does not become a zero score. Old scores remain visible as previous
  assessments while the care rules continue to work.

## Snapshots without a schema migration

The existing tenant-scoped `ai_predictions` history is append-only. Each
assessment appends `input_data.snapshot_version = lead_snapshot_1` containing:

- scoring timestamp, lead creation timestamp, source update timestamp and contact timestamp;
- lead state at scoring time (used for care/eligibility, never as a model feature);
- source, occupation, purpose, actual course family input and last explicit engagement;
- actual call attempts, recorded connected calls, contacts in the last 7 days,
  lead age and days since the last real contact;
- a fixed 30-day outcome window.

Phone numbers, email addresses, names and free-text notes are not included.
Internal notes/status-change events do not count as contacts. Snapshot capture
occurs before calling AI; successful score persistence and the history write
use one tenant transaction. Older concurrent responses cannot replace a newer score.
Snapshots are also collected when AI is unavailable, so the dataset is not
restricted to cases where inference happened to succeed.

## Export and label outcomes

An authenticated tenant admin can download:

```text
GET /api/leads/scoring-dataset
Authorization: Bearer <admin token>
```

The endpoint exports the current tenant only, up to 50,000 snapshots. It is
registered before `/:id` and is admin-only; no database connection is exposed
to the AI service. JSON includes no personal contact information.

`converted_30d = 1` means a Student record linked to the lead was created after
the snapshot and within 30 days. This matches the existing conversion transaction.
Even known positive cases wait until the full window has elapsed before becoming
eligible, avoiding an early-positive-only training cohort. Unmatured snapshots
have a null label. Existing students, deleted leads and enrollment without an
authoritative Student timestamp are excluded rather than guessed as negatives.
The source snapshot is never edited when labels are derived during export.

Do not relabel public X Education data as 30-day CRM snapshots.

## Train the real CRM model

Save the exported JSON locally. From `SmartEdu_CRM_AI`:

```powershell
..\.venv\Scripts\python.exe ai_models/train_crm_lead_scoring.py --dataset path/to/lead-scoring-snapshots.json
```

The trainer:

1. validates the schema, binary labels and completed 30-day windows, then
   selects actual post-contact snapshots to match the history model's serving cohort;
2. keeps at most one snapshot per lead/day without inspecting the outcome;
3. assigns entire leads to chronological train/validation/test cohorts;
4. removes training rows whose outcome window crosses validation start, and
   validation rows whose outcome window crosses test start;
5. groups calibration folds by lead and fits preprocessing within each fold;
6. chooses between two RF depths on validation PR-AUC and reports final test
   precision/recall, Brier loss, HOT precision, distinct HOT leads and top-20% precision.

Insufficient time coverage, missing classes or insufficient distinct leads stop
training with an explanation. At least several separated cohorts with complete
outcome windows are necessary; a large number of repeated snapshots from a few
leads does not replace having distinct leads.

Outputs are `lead_scoring_crm_candidate.pkl` and `lead_scoring_crm_report.json`.
The acceptance target is fixed before evaluation: precision at score >= 80%
must reach 85% with at least 20 distinct HOT leads on both validation and test.
It is an empirical acceptance check, not a guarantee of future performance.

After reviewing the report, add `--activate` to the command. Activation is
refused when the gate fails; the reference model remains available. Successful
activation writes `lead_scoring_crm_pipeline.pkl`. Restart the AI service to
load it. The API then uses that model after actual contact, identifying its
score as a 30-day CRM conversion estimate. Initial scoring still uses only the
three profile features from the public intake model.
Existing lead scores are refreshed on the next assessment; no old snapshot is rewritten.

## API compatibility and rollout

`/predict-lead` and its old artifact remain for compatibility only. Its legacy
trainer and generated enhanced dataset were removed because that experiment
invented student years and modified outcome labels. CRM uses only the staged
v2 endpoints below; do not train or serve the old experiment as the CRM model.

Existing `/predict-lead-v2` and `/explain-lead-v2` requests still work. The new
optional observed counts are `Answered_Call_Count`, `Activity_Count_7d` and
`Days_Since_Last_Activity`. The backend also sends `Has_Contact_Activity` from
actual activity records, covering old emails/meetings outside the last 7 days.
Missing values remain unknown to the CRM model.
The response adds `api_contract_version = lead_scoring_3`,
`conversion_probability`, `score_type` and `prediction_horizon_days` while
retaining the legacy percentage/recommendation fields. Responses include
`scoring_stage` (`initial`/`post_contact`), `assessment_basis` and `active_features`.
The backend rejects old models and a stage that disagrees with the snapshot.
The UI shows the stage in score tooltips and on the detail page; it does not add
another care-priority tag or change the Kanban funnel.

From the workspace root, rebuild the running AI image after code changes:

```powershell
docker compose up -d --build smartedu-ai
```

The model artifact is mounted, but `api_server.py` is copied into the image;
a container restart alone cannot update an older API implementation.
Datasets, tests, docs and report images are excluded from the Docker build context.
Train and export reports locally; the inference image only needs model artifacts
and their Python feature code.

Deploy the AI and backend changes together, then refresh the frontend. The
backend rejects old/inconsistent AI responses and continues care rules until a
matching model response becomes available. No database migration is required.

Care deadlines are re-evaluated on list/detail reads using the saved contact
timestamps; refreshing the screen does not keep moving an overdue deadline.
Priority is not a replacement for customer consent or sales judgment.
