"""
generate_lead_scoring_v2_dataset.py
------------------------------------
SmartEdu CRM — Synthetic Dataset Generator for Lead Scoring v2

Generates a realistic CRM lead dataset with 7 business-aligned features:
  1. Lead_Source          (categorical)  — marketing channel
  2. Occupation           (categorical)  — customer segment
  3. Study_Purpose        (categorical)  — learning motivation
  4. Course_Interested    (categorical)  — target product
  5. Call_Attempt_Count   (numeric)      — sales effort intensity
  6. Last_Engagement_Status (categorical) — most recent interaction outcome
  7. Days_Since_Created   (numeric)      — lead freshness / staleness

Target: Converted (0 = not converted, 1 = converted)

Business Rules Encoded:
  - referral / organic leads convert higher than paid ads
  - student_y3_y4 with study_abroad / pass_exam have highest urgency
  - positive_interaction is the strongest single predictor
  - leads older than 14 days decay sharply
  - after 4-5 call attempts without conversion, probability drops
  - 15-20% noise added to prevent model from overfitting to rules

Target distribution: ~35-40% converted (realistic CRM ratio)

Usage:
    python generate_lead_scoring_v2_dataset.py
"""

import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
N_RECORDS   = 5000
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "archive")
OUTPUT_FILE = "lead_scoring_v2_dataset.csv"

# ---------------------------------------------------------------------------
# Value distributions (mimic real CRM data)
# ---------------------------------------------------------------------------
LEAD_SOURCES = {
    "facebook":         0.40,   # Social ads
    "google_ads":       0.25,   # Search intent
    "tiktok":           0.15,   # Video ads
    "website_organic":  0.10,   # SEO
    "referral":         0.08,   # Word-of-mouth
    "other":            0.02,   # Events, offline
}

OCCUPATIONS = {
    "student_y1_y2":        0.25,
    "student_y3_y4":        0.30,   # Biggest segment for edu centers
    "working_professional": 0.25,
    "unemployed":           0.10,
    "parent_enrolling":     0.10,   # Parents enrolling kids
}

STUDY_PURPOSES = {
    "pass_exam":            0.30,   # University requirement
    "study_abroad":         0.20,   # IELTS / TOEFL
    "career_advancement":   0.25,   # Career growth
    "hobby":                0.15,   # Casual learners
    "company_training":     0.10,   # Corporate training
}

COURSES = {
    "communication_english": 0.25,
    "ielts":                 0.25,
    "toeic":                 0.15,
    "frontend":              0.10,
    "backend_nodejs":        0.10,
    "data_analysis":         0.08,
    "kids_english":          0.07,
}

ENGAGEMENT_STATUSES = {
    "not_answering":        0.30,
    "busy_call_back":       0.20,
    "interested_need_time": 0.15,
    "positive_interaction": 0.20,
    "wrong_number":         0.05,
    "rejected":             0.10,
}


def weighted_choice(options: dict, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from a weighted categorical distribution."""
    values = list(options.keys())
    probs  = np.array(list(options.values()))
    probs  = probs / probs.sum()  # Normalize
    return rng.choice(values, size=size, p=probs)


def compute_conversion_probability(row: pd.Series) -> float:
    """
    Compute conversion probability based on business logic rules.
    
    This function encodes domain knowledge from experienced Sales teams:
    each feature contributes a score, and the final probability is the
    logistic sigmoid of the total score.
    """
    score = 0.0

    # === Lead Source effect ===
    source_scores = {
        "referral":         +1.2,   # Trusted recommendation
        "website_organic":  +0.8,   # Active seeker
        "google_ads":       +0.3,   # Intent-based search
        "facebook":         -0.1,   # Low-intent browsing
        "tiktok":           -0.3,   # Very low-intent
        "other":            +0.0,
    }
    score += source_scores.get(row["Lead_Source"], 0)

    # === Occupation / Urgency effect ===
    occupation_scores = {
        "student_y3_y4":        +0.9,   # Graduation pressure
        "working_professional": +0.4,   # Has budget, less time
        "student_y1_y2":        -0.2,   # No urgency yet
        "parent_enrolling":     +0.3,   # Spending on child
        "unemployed":           -0.5,   # Budget constraint
    }
    score += occupation_scores.get(row["Occupation"], 0)

    # === Study Purpose effect ===
    purpose_scores = {
        "study_abroad":       +1.0,   # Strong deadline-driven motivation
        "career_advancement": +0.7,   # Career pressure
        "pass_exam":          +0.5,   # Academic requirement
        "company_training":   +0.4,   # Company-funded
        "hobby":              -0.8,   # Low commitment
    }
    score += purpose_scores.get(row["Study_Purpose"], 0)

    # === Course effect (price sensitivity proxy) ===
    course_scores = {
        "communication_english": +0.3,   # Low price, easy entry
        "toeic":                 +0.2,   # Moderate price
        "kids_english":          +0.1,   # Parents often commit
        "ielts":                 -0.1,   # Premium, longer cycle
        "frontend":              +0.0,   # Tech courses
        "backend_nodejs":        -0.1,   # More niche
        "data_analysis":         -0.2,   # Newer, less trust
    }
    score += course_scores.get(row["Course_Interested"], 0)

    # === Call Attempt Count effect (diminishing returns) ===
    calls = row["Call_Attempt_Count"]
    if calls <= 1:
        score += 0.3    # First contact — fresh
    elif calls <= 3:
        score += 0.0    # Normal follow-up
    elif calls <= 5:
        score -= 0.8    # Losing interest
    else:
        score -= 1.5    # Almost certainly dead

    # === Last Engagement Status (STRONGEST SIGNAL) ===
    engagement_scores = {
        "positive_interaction":  +1.8,   # Gold — asking about price, schedule
        "interested_need_time":  +0.5,   # Warm but hesitant
        "busy_call_back":        -0.1,   # Neutral
        "not_answering":         -0.7,   # Can't reach
        "rejected":              -2.0,   # Explicit "no"
        "wrong_number":          -2.5,   # Invalid lead
    }
    score += engagement_scores.get(row["Last_Engagement_Status"], 0)

    # === Days Since Created (lead decay) ===
    days = row["Days_Since_Created"]
    if days <= 2:
        score += 1.0    # Hot window — 48 hours
    elif days <= 7:
        score += 0.2    # Still warm
    elif days <= 14:
        score -= 0.3    # Getting cold
    elif days <= 30:
        score -= 0.8    # Cold
    else:
        score -= 1.5    # Very cold / dead

    # === Interaction bonuses (feature combinations) ===
    # Combo: Y3-Y4 student + study_abroad = extremely motivated
    if row["Occupation"] == "student_y3_y4" and row["Study_Purpose"] == "study_abroad":
        score += 0.5

    # Combo: Fresh lead + positive interaction = almost guaranteed
    if days <= 2 and row["Last_Engagement_Status"] == "positive_interaction":
        score += 0.7

    # Combo: Many calls + not answering = dead lead
    if calls >= 4 and row["Last_Engagement_Status"] == "not_answering":
        score -= 0.5

    # Return score (sigmoid applied globally with noise later)
    return score


def generate_dataset(n: int = N_RECORDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate the full synthetic Lead Scoring v2 dataset.
    
    Steps:
      1. Sample features with softened conditional dependencies
      2. Compute raw conversion score using business logic
      3. Add base bias (to lower rate to ~35%) and numeric noise directly to score
      4. Bernoulli sampling to assign class labels
      5. Inject random missing values (NaNs) into features
    """
    rng = np.random.default_rng(seed)

    print(f"Generating {n} synthetic lead records...")

    # --- Step 1: Sample features with dependencies ---
    df = pd.DataFrame(index=range(n))
    df["Lead_Source"] = weighted_choice(LEAD_SOURCES, n, rng)
    df["Occupation"] = weighted_choice(OCCUPATIONS, n, rng)

    study_purposes = []
    courses = []

    for occ in df["Occupation"]:
        if occ in ["student_y1_y2", "student_y3_y4"]:
            sp = weighted_choice({"pass_exam": 0.4, "study_abroad": 0.3, "career_advancement": 0.15, "hobby": 0.15}, 1, rng)[0]
            if sp == "study_abroad":
                c = weighted_choice({"ielts": 0.6, "communication_english": 0.3, "toeic": 0.1}, 1, rng)[0]
            else:
                c = weighted_choice({"toeic": 0.5, "communication_english": 0.3, "ielts": 0.1, "frontend": 0.05, "backend_nodejs": 0.05}, 1, rng)[0]
        elif occ == "working_professional":
            sp = weighted_choice({"career_advancement": 0.6, "company_training": 0.2, "hobby": 0.1, "pass_exam": 0.1}, 1, rng)[0]
            c = weighted_choice({"communication_english": 0.3, "toeic": 0.2, "data_analysis": 0.3, "frontend": 0.1, "ielts": 0.1}, 1, rng)[0]
        elif occ == "parent_enrolling":
            sp = weighted_choice({"pass_exam": 0.5, "study_abroad": 0.2, "hobby": 0.3}, 1, rng)[0]
            c = weighted_choice({"kids_english": 0.8, "ielts": 0.1, "communication_english": 0.1}, 1, rng)[0]
        else: # unemployed
            sp = weighted_choice({"career_advancement": 0.5, "hobby": 0.3, "study_abroad": 0.2}, 1, rng)[0]
            c = weighted_choice({"communication_english": 0.4, "frontend": 0.2, "backend_nodejs": 0.2, "data_analysis": 0.1, "toeic": 0.1}, 1, rng)[0]

        study_purposes.append(sp)
        courses.append(c)

    df["Study_Purpose"] = study_purposes
    df["Course_Interested"] = courses
    df["Last_Engagement_Status"] = weighted_choice(ENGAGEMENT_STATUSES, n, rng)

    # Call Attempt Count: follows a right-skewed distribution (most leads 1-3 calls)
    df["Call_Attempt_Count"] = rng.choice(
        [1, 2, 3, 4, 5, 6, 7, 8],
        size=n,
        p=[0.20, 0.25, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
    )

    # Days Since Created: exponential-like (many fresh, few old)
    raw_days = rng.exponential(scale=10, size=n)
    df["Days_Since_Created"] = np.clip(raw_days.astype(int), 0, 90)

    # --- Step 2: Compute conversion probability score ---
    print("Computing conversion probability scores from business logic...")
    raw_scores = df.apply(compute_conversion_probability, axis=1)

    # --- Step 3: Global bias and numeric noise ---
    # Shift base score down to target ~35-40% conversion rate naturally
    base_bias = -1.3 
    # Add Gaussian noise directly to the score to fuzz the rules
    score_noise = rng.normal(0, 0.8, size=n)
    df["_prob"] = 1.0 / (1.0 + np.exp(-(raw_scores + base_bias + score_noise)))

    # --- Step 4: Generate binary label using probabilities ---
    df["Converted"] = rng.binomial(1, df["_prob"])

    # --- Step 5: Add Label Noise (~2%) ---
    noise_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    df.loc[noise_idx, "Converted"] = 1 - df.loc[noise_idx, "Converted"]
    print(f"  Added 2% label noise (flipped {len(noise_idx)} records).")

    # --- Step 6: Add Missing Data explicitly (~2-5%) ---
    missing_occ = rng.choice(n, size=int(n * 0.03), replace=False)
    df.loc[missing_occ, "Occupation"] = np.nan
    
    missing_days = rng.choice(n, size=int(n * 0.05), replace=False)
    # Using float type because integer pandas columns don't naturally support nan
    df["Days_Since_Created"] = df["Days_Since_Created"].astype(float)
    df.loc[missing_days, "Days_Since_Created"] = np.nan
    print(f"  Injected missing values (NaNs) into Occupation and Days_Since_Created.")

    print(f"  Final conversion rate: {df['Converted'].mean():.1%}")

    # Drop internal column
    df = df.drop(columns=["_prob"])

    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print comprehensive dataset statistics."""
    print("\n" + "=" * 65)
    print("  DATASET SUMMARY")
    print("=" * 65)
    print(f"  Total records     : {len(df)}")
    print(f"  Converted (1)     : {df['Converted'].sum()}  ({df['Converted'].mean():.1%})")
    print(f"  Not Converted (0) : {(df['Converted'] == 0).sum()}  ({(df['Converted'] == 0).mean():.1%})")

    print(f"\n  Features: {list(df.columns[:-1])}")

    print("\n--- Categorical Distributions ---")
    for col in ["Lead_Source", "Occupation", "Study_Purpose", "Course_Interested", "Last_Engagement_Status"]:
        print(f"\n  {col}:")
        vc = df[col].value_counts()
        conv_rate = df.groupby(col)["Converted"].mean()
        for val in vc.index:
            print(f"    {val:30s}  n={vc[val]:4d}  conv_rate={conv_rate[val]:.1%}")

    print("\n--- Numeric Distributions ---")
    for col in ["Call_Attempt_Count", "Days_Since_Created"]:
        print(f"\n  {col}:")
        print(f"    mean={df[col].mean():.1f}  median={df[col].median():.1f}  "
              f"min={df[col].min()}  max={df[col].max()}")
        # Conversion rate by bins
        if col == "Call_Attempt_Count":
            bins = [0, 1, 3, 5, 100]
            labels = ["1", "2-3", "4-5", "6+"]
        else:
            bins = [0, 2, 7, 14, 30, 100]
            labels = ["0-2d", "3-7d", "8-14d", "15-30d", "30+d"]
        
        df["_bin"] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        bin_conv = df.groupby("_bin", observed=False)["Converted"].agg(["count", "mean"])
        for label_val in labels:
            if label_val in bin_conv.index:
                row = bin_conv.loc[label_val]
                print(f"    {label_val:10s}  n={int(row['count']):4d}  conv_rate={row['mean']:.1%}")
        df = df.drop(columns=["_bin"])

    print("\n" + "=" * 65)


def main():
    print("=" * 65)
    print("  SmartEdu CRM — Lead Scoring v2 Dataset Generator")
    print("=" * 65)

    # Generate
    df = generate_dataset()

    # Summary
    print_dataset_summary(df)

    # Export
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    df.to_csv(output_path, index=False)
    print(f"\n  Dataset exported to: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
