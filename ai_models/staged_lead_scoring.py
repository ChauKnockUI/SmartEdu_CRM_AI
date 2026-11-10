"""Route scoring using actual CRM contact evidence, never website visit counts."""
import numpy as np
import pandas as pd
from lead_contract import engagement_group


def has_contact(row):
    explicit = row.get("Has_Contact_Activity")
    if explicit is not None and not pd.isna(explicit):
        return bool(explicit)
    # Compatibility for older callers. Backend sends the authoritative flag,
    # covering old emails/meetings even when there were no calls in the last week.
    if "Engagement_Group" in row:
        return row["Engagement_Group"] in {"active", "unreachable"}
    for name in ("Call_Attempt_Count", "Activity_Count_7d", "Answered_Call_Count"):
        value = row.get(name)
        if value is not None and not pd.isna(value) and float(value) > 0:
            return True
    last = row.get("Days_Since_Last_Activity")
    return (last is not None and not pd.isna(last)) or engagement_group(row.get("Last_Engagement_Status")) != "unknown"


class StagedLeadModel:
    def __init__(self, intake, engaged, metadata):
        self.stages_ = {"initial": intake, "post_contact": engaged}
        self.lead_metadata_ = metadata
        self.crm_model_ = None

    def select_pipeline(self, row):
        if not has_contact(row):
            return self.stages_["initial"], "initial", "profile_only"
        if self.crm_model_ is not None:
            return self.crm_model_, "post_contact", "crm_history"
        response = row.get("Engagement_Group", engagement_group(row.get("Last_Engagement_Status")))
        # Contact attempts alone do not establish a customer response.
        if response == "unknown":
            return self.stages_["initial"], "post_contact", "response_pending"
        selected = self.stages_["post_contact"]
        basis = "profile_and_response" if "Engagement_Group" in selected.lead_metadata_.get("active_features", []) else "profile_only"
        return selected, "post_contact", basis

    def predict_proba(self, frame):
        probabilities = np.empty((len(frame), 2))
        groups = {}
        for position, (_, row) in enumerate(frame.iterrows()):
            selected, _, _ = self.select_pipeline(row)
            key = id(selected)
            groups.setdefault(key, (selected, []))[1].append(position)
        for selected, positions in groups.values():
            probabilities[positions] = selected.predict_proba(frame.iloc[positions])
        return probabilities

    def predict(self, frame):
        return (self.predict_proba(frame)[:, 1] >= .5).astype(int)
