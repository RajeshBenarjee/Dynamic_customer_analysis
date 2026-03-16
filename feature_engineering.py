# feature_engineering.py
"""
Creates all derived and proxy features from the preprocessed data.
Every feature is clearly documented as: direct | derived | proxy.
"""
import pandas as pd
import numpy as np
from utils import JOURNEY_STAGES

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    # ── Helper: safe normalise to 0-1 ──────────────────────────
    def norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    # ── 1. Recency proxy (derived from purchase_frequency) ──────
    # Higher purchase frequency → lower recency value (more recent)
    if "purchase_frequency" in df.columns:
        df["recency_proxy"] = 1 - norm(df["purchase_frequency"].fillna(0))
    else:
        df["recency_proxy"] = 0.5

    # ── 2. Monetary proxy (derived from avg_spend × purchase_count) ─
    if "avg_spend" in df.columns:
        spend = df["avg_spend"].fillna(df["avg_spend"].median()
                                       if "avg_spend" in df.columns else 50)
        count = df.get("purchase_count", pd.Series(1, index=df.index)).fillna(1)
        df["monetary_proxy"] = norm(spend * count)
    else:
        df["monetary_proxy"] = 0.5

    # ── 3. Product diversity (proxy from category_preference) ───
    if "category_preference" in df.columns:
        cat_counts = df["category_preference"].value_counts()
        df["product_diversity"] = df["category_preference"].map(
            lambda x: 1 / (cat_counts.get(x, 1)) if pd.notna(x) else 0.5
        )
        df["product_diversity"] = norm(df["product_diversity"])
    else:
        df["product_diversity"] = 0.5

    # ── 4. Browsing intensity proxy ─────────────────────────────
    if "browsing_frequency" in df.columns:
        df["browsing_intensity"] = norm(df["browsing_frequency"].fillna(0))
    else:
        df["browsing_intensity"] = 0.5

    # ── 5. Discount sensitivity (already binary / normalised) ───
    if "discount_sensitivity" not in df.columns:
        df["discount_sensitivity"] = 0.0

    # ── 6. Loyalty score (proxy: freq × satisfaction × ~churn) ──
    freq   = norm(df.get("purchase_frequency", pd.Series(0, index=df.index)).fillna(0))
    sat    = norm(df.get("satisfaction_score",  pd.Series(3, index=df.index)).fillna(3))
    rating = norm(df.get("rating",              pd.Series(3, index=df.index)).fillna(3))
    df["loyalty_score"] = (freq * 0.4 + sat * 0.35 + rating * 0.25).clip(0, 1)

    # ── 7. Churn proxy (inverse of loyalty) ─────────────────────
    df["churn_proxy"] = (1 - df["loyalty_score"]).clip(0, 1)

    # ── 8. Conversion proxy ─────────────────────────────────────
    # High browsing + high purchase freq → high conversion
    browse  = df["browsing_intensity"]
    pf      = norm(df.get("purchase_frequency", pd.Series(0, index=df.index)).fillna(0))
    df["conversion_proxy"] = ((browse + pf) / 2).clip(0, 1)

    # ── 9. Repeat purchase score ────────────────────────────────
    if "purchase_count" in df.columns:
        df["repeat_purchase_score"] = norm(df["purchase_count"].fillna(0))
    else:
        df["repeat_purchase_score"] = df["conversion_proxy"]

    # ── 10. Satisfaction score normalise ───────────────────────
    if "satisfaction_score" in df.columns:
        df["satisfaction_score"] = norm(df["satisfaction_score"].fillna(3))
    else:
        df["satisfaction_score"] = 0.5

    # ── 11. Rating normalise ────────────────────────────────────
    if "rating" not in df.columns:
        df["rating"] = 0.6
    else:
        df["rating"] = norm(df["rating"].fillna(3))

    # ── 12. Journey stage assignment (proxy / inferred) ─────────
    # NOTE: These are INFERRED stages — not true clickstream data.
    df["journey_stage"] = _assign_journey_stage(df)

    # ── 13. Abandonment flag (proxy) ────────────────────────────
    df["abandonment_proxy"] = (
        (df["browsing_intensity"] > 0.6) &
        (df["conversion_proxy"] < 0.4)
    ).astype(int)

    # ── 14. Risk label for prediction (binary) ──────────────────
    df["risk_label"] = (
        (df["churn_proxy"] > 0.6) |
        (df["satisfaction_score"] < 0.3) |
        (df["abandonment_proxy"] == 1)
    ).astype(int)

    return df


def _assign_journey_stage(df: pd.DataFrame) -> pd.Series:
    """
    INFERRED proxy journey stage — not true clickstream.
    Logic derived from available behavioral columns.
    """
    stages = pd.Series("Awareness", index=df.index)

    bi   = df.get("browsing_intensity", pd.Series(0, index=df.index))
    cp   = df.get("conversion_proxy",   pd.Series(0, index=df.index))
    sat  = df.get("satisfaction_score", pd.Series(0.5, index=df.index))
    loy  = df.get("loyalty_score",      pd.Series(0, index=df.index))
    disc = df.get("discount_sensitivity",pd.Series(0, index=df.index))

    # Awareness → Exploration (some browsing)
    stages[bi > 0.2] = "Exploration"

    # Exploration → Consideration (moderate browsing)
    stages[bi > 0.4] = "Consideration"

    # Consideration → Cart Intent (high browsing, some discount interest)
    stages[(bi > 0.5) & (disc > 0) ] = "Cart Intent"

    # Cart Intent → Checkout Intent
    stages[(bi > 0.6) & (cp > 0.4)] = "Checkout Intent"

    # Checkout → Purchase (high conversion proxy)
    stages[cp > 0.6] = "Purchase"

    # Purchase → Post-Purchase (high satisfaction/loyalty)
    stages[(cp > 0.6) & (sat > 0.6)] = "Post-Purchase"

    return stages


def get_journey_funnel_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["journey_stage"].value_counts().reindex(JOURNEY_STAGES, fill_value=0)
    return pd.DataFrame({
        "stage": JOURNEY_STAGES,
        "count": [counts.get(s, 0) for s in JOURNEY_STAGES]
    })