# Lead conversion: aligned feature evaluation

Model: `lead_staged_rf_3`. Source: X Education, 9,240 records.

Public source and CRM records are processed into a shared, coarser feature vocabulary.
No target-derived fallback, random assignment, or full-dataset fitted statistic is used.

| Feature | Public source | CRM |
|---|---|---|
| Source_Group | Lead Source grouped into search/social/referral/website/other/unknown | Same grouping of lead_source |
| Occupation_Group | Student/working/unemployed/other/unknown | Both student year groups merge into student |
| Goal_Group | Explicit career motivation; other source answers unknown | career/company -> career; exam/study abroad -> academic; hobby -> personal |
| Engagement_Group | Recorded email/phone/chat response -> active; unreachable/bounced -> unreachable; otherwise unknown | Explicit contact result; used only after actual contact |

Initial model uses Source_Group, Occupation_Group and Goal_Group only.
After actual contact, the response model may add Engagement_Group.
A contact without a recorded result retains the intake model until response evidence exists.
TotalVisits and call counts are not features of either public-data model.

Course choice and lead age are excluded: specialization is not a course, and website seconds are not lead age.
Tags are excluded; SMS Sent and Converted to Lead do not establish customer response.
Missing occupation is unknown, not unemployed. No student year is inferred.

## Model selection

Split 80/20 with stratification and seed 42 before model selection.
All candidates use the same five CV folds within the 80% training partition.
Every calibration fold fits its own encoder and forest.
Prefer fewer proxies unless an added candidate improves mean CV ROC-AUC by more than 0.005.
The test set is not used to select features or tune thresholds.

| Candidate | Train CV AUC | Train CV F1 |
|---|---:|---:|
| intake | 0.7606 | 0.4827 |
| with_engagement | 0.7909 | 0.5678 |

Selected: `with_engagement`.
Stage feature sets: `{'initial': ['Source_Group', 'Occupation_Group', 'Goal_Group'], 'post_contact': ['Source_Group', 'Occupation_Group', 'Goal_Group', 'Engagement_Group']}`.

## Stage evaluation on the same public held-out partition

These are source-domain comparisons, not evidence of performance on new CRM leads.
Source data has no contact chronology; routed evaluation uses recorded response availability as a proxy.

| Model | Accuracy | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|
| initial | 0.7165 | 0.7427 | 0.6357 | 0.1875 |
| post_contact | 0.7235 | 0.7685 | 0.6804 | 0.1814 |
| routed_response_proxy | 0.7154 | 0.7649 | 0.6774 | 0.1861 |

## Held-out test

Train: 7,392; test: 1,848.
Prior baseline uses exactly the same train/test split.

| Metric | Selected RF | Prior baseline |
|---|---:|---:|
| accuracy | 0.7154 | 0.6147 |
| precision | 0.8811 | 0.0000 |
| recall | 0.3020 | 0.0000 |
| f1_score | 0.4498 | 0.0000 |
| roc_auc | 0.7649 | 0.5000 |
| pr_auc | 0.6774 | 0.3853 |
| brier_score | 0.1861 | 0.2368 |

Confusion matrix [TN, FP; FN, TP]: `[[1107, 29], [497, 215]]`.

HOT >= 0.80; WARM >= 0.50; binary conversion prediction >= 0.50.
Threshold analysis is descriptive; thresholds are not tuned on test outcomes.

## Limits

- Public-domain accuracy and probability calibration do not establish CRM performance.
- The initial model uses profile information only; call counts and TotalVisits are excluded.
- After contact, public response categories remain approximate; missing results retain the profile-only model.
- Source motivation contains career/unknown only; academic/personal goals lack source examples.
- Last activities lack timestamps; pre-outcome availability remains unverified despite excluding Tags.
- No conversion timestamps or outcome horizon exist, so temporal validation is unavailable.
- Coarse profiles repeat across splits; unique source lead IDs were checked before splitting.

## Next data collection

Export lead ID, scoring-time snapshots, real features, conversion timestamp and an explicit outcome horizon.
Keep snapshots of one lead in the same partition and use later dates for final temporal testing.
Incomplete outcome windows must not be labeled as negative.

The artifact bundles two trained pipelines; stage routing and SHAP use the same selected pipeline.
