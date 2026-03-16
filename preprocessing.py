# preprocessing.py
"""
Cleans and standardises the merged DataFrame.
"""
import pandas as pd
import numpy as np
import re
from utils import NUMERIC_FEATURES, CATEGORICAL_FEATURES

FREQ_MAP = {
    "daily": 7, "weekly": 4, "bi-weekly": 2, "monthly": 1,
    "fortnightly": 2, "quarterly": 0.33, "annually": 0.08,
    "rarely": 0.25, "occasionally": 0.5, "often": 3, "always": 6,
    "sometimes": 1, "never": 0
}

SATISFACTION_MAP = {
    "very satisfied": 5, "satisfied": 4, "neutral": 3,
    "dissatisfied": 2, "very dissatisfied": 1,
    "positive": 4, "negative": 2, "mixed": 3
}

def clean_text_column(series: pd.Series) -> pd.Series:
    """Lowercase, strip, normalise whitespace."""
    return (series.astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .replace("nan", np.nan))

def encode_frequency(series: pd.Series) -> pd.Series:
    """Convert text frequency to numeric scale 0-7."""
    s = clean_text_column(series)
    return s.map(FREQ_MAP).fillna(s.apply(
        lambda x: float(x) if str(x).replace('.', '').isnumeric() else np.nan
    ))

def encode_satisfaction(series: pd.Series) -> pd.Series:
    s = clean_text_column(series)
    return s.map(SATISFACTION_MAP).fillna(
        pd.to_numeric(series, errors="coerce")
    )

def encode_binary(series: pd.Series) -> pd.Series:
    s = clean_text_column(series)
    return s.map({"yes": 1, "no": 0, "true": 1, "false": 0,
                  "1": 1, "0": 0}).fillna(pd.to_numeric(series, errors="coerce"))

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Frequency columns ──
    for col in ["purchase_frequency", "browsing_frequency"]:
        if col in df.columns:
            df[col] = encode_frequency(df[col])

    # ── Satisfaction / rating ──
    for col in ["satisfaction_score"]:
        if col in df.columns:
            df[col] = encode_satisfaction(df[col])

    for col in ["rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].clip(1, 5)

    # ── Spend / monetary ──
    for col in ["avg_spend"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r"[^\d.]", "", regex=True),
                errors="coerce"
            )

    # ── Binary ──
    for col in ["discount_sensitivity", "subscription_status"]:
        if col in df.columns:
            df[col] = encode_binary(df[col])

    # ── Categorical cleanup ──
    for col in ["gender", "payment_method", "category_preference",
                "age_group", "location"]:
        if col in df.columns:
            df[col] = clean_text_column(df[col]).str.title()

    # ── Age ──
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce").clip(10, 100)
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 18, 25, 35, 50, 65, 100],
            labels=["<18", "18-25", "26-35", "36-50", "51-65", "65+"]
        ).astype(str)

    # ── Text ──
    if "review_text" in df.columns:
        df["review_text"] = clean_text_column(df["review_text"])
        df["review_text"] = df["review_text"].replace("nan", "")

    # ── Fill missing numerics with median ──
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median if not np.isnan(median) else 0)

    # ── Fill missing categoricals ──
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    # ── Drop full-duplicate rows ──
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    return df