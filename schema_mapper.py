# schema_mapper.py
"""
Maps each raw dataset's columns to a unified analytical schema.
Documents what is: direct | derived | inferred (proxy).
"""
import pandas as pd
import numpy as np
from utils import NUMERIC_FEATURES

# ── Column alias maps for each dataset ───────────────────────────
# Key = unified name, Value = list of possible raw column names
COLUMN_ALIASES = {
    "customer_id":          ["customer_id", "id", "customerid", "user_id"],
    "age":                  ["age"],
    "gender":               ["gender"],
    "category_preference":  ["product_category","category","shopping_mall","item_purchased"],
    "purchase_frequency":   ["purchase_frequency","purchase_freq","frequency_of_purchases",
                             "frequency","shopping_frequency"],
    "avg_spend":            ["purchase_amount","amount","price","average_spend",
                             "purchase_amount_(usd)"],
    "discount_sensitivity": ["discount_availed","coupon_used","promo_code_used",
                             "discount_applied","use_of_promo_codes"],
    "satisfaction_score":   ["customer_satisfaction","satisfaction","rating",
                             "review_rating","service_rating"],
    "rating":               ["product_rating","ratings","rating","review_rating",
                             "star_rating"],
    "review_text":          ["review","review_text","feedback","comments","product_review",
                             "customer_review"],
    "payment_method":       ["payment_method","preferred_payment_method",
                             "preferred_order_pay","payment_option"],
    "subscription_status":  ["subscription_status","membership_status","loyalty_member",
                             "prime_member","loyalty_status"],
    "purchase_count":       ["number_of_items_purchased","quantity","order_quantity",
                             "total_orders","num_purchases"],
    "browsing_frequency":   ["browsing_frequency","browsing_habit","visit_frequency",
                             "sessions_per_week"],
    "cart_behavior":        ["add_to_cart_browsing","cart_completion_rate","add_to_cart"],
    "return_rate":          ["return_rate","product_return","return_customer"],
    "age_group":            ["age_group","age_bracket"],
    "location":             ["location","city","state","region","country"],
    "gender":               ["gender","sex"],
}

def _find_column(df: pd.DataFrame, unified_name: str):
    """Return matched column value Series or None."""
    aliases = COLUMN_ALIASES.get(unified_name, [unified_name])
    for alias in aliases:
        if alias in df.columns:
            return df[alias].copy()
    return None

def map_dataset(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Produce a harmonised DataFrame with unified column names."""
    if df.empty:
        return df

    out = pd.DataFrame()
    out["_source"] = source_name

    for unified_name in COLUMN_ALIASES:
        series = _find_column(df, unified_name)
        if series is not None:
            out[unified_name] = series.values

    # Ensure customer_id exists
    if "customer_id" not in out.columns or out["customer_id"].isnull().all():
        out["customer_id"] = [f"{source_name}_{i}" for i in range(len(df))]

    return out


def merge_all(dfs: dict) -> pd.DataFrame:
    """Merge all mapped datasets into one unified DataFrame."""
    mapped = []
    for name, df in dfs.items():
        if not df.empty:
            m = map_dataset(df, name)
            mapped.append(m)

    if not mapped:
        return pd.DataFrame()

    merged = pd.concat(mapped, ignore_index=True, sort=False)
    merged.reset_index(drop=True, inplace=True)
    return merged