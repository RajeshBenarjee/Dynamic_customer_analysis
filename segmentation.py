# segmentation.py
"""
Clusters customers into behavioural segments using K-Means (primary),
DBSCAN, and Hierarchical clustering (comparison).

All segment labels are INFERRED from proxy features — not direct
clickstream identities.
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from utils import (
    save_model, load_model, save_df,
    NUMERIC_FEATURES, SEGMENT_LABELS, SEGMENT_COLORS, MODEL_DIR
)

# ── Feature set used for clustering ──────────────────────────────
# Only include columns that are actually present after feature engineering
SEG_FEATURES = [
    "purchase_frequency",
    "avg_spend",
    "discount_sensitivity",
    "satisfaction_score",
    "rating",
    "browsing_intensity",
    "loyalty_score",
    "churn_proxy",
    "conversion_proxy",
    "repeat_purchase_score",
    "product_diversity",
    "recency_proxy",
    "monetary_proxy",
]


def _available_features(df: pd.DataFrame) -> list:
    """Return only those SEG_FEATURES that exist in df."""
    return [f for f in SEG_FEATURES if f in df.columns]


def _prep_matrix(df: pd.DataFrame, features: list) -> np.ndarray:
    """Extract, impute, and scale feature matrix."""
    X = df[features].copy()
    # Fill any remaining NaN with column median
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


# ── Optimal k search ─────────────────────────────────────────────
def find_optimal_k(df: pd.DataFrame, k_range: range = range(3, 10)) -> dict:
    """
    Compute inertia and silhouette scores for a range of k values.
    Returns dict with lists for plotting the elbow / silhouette curve.
    """
    features = _available_features(df)
    if len(features) < 2:
        return {"k": list(k_range), "inertia": [], "silhouette": []}

    X, _ = _prep_matrix(df, features)

    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if k > 1:
            silhouettes.append(silhouette_score(X, labels, sample_size=min(3000, len(X))))
        else:
            silhouettes.append(0)

    return {
        "k": list(k_range),
        "inertia": inertias,
        "silhouette": silhouettes,
    }


# ── Main segmentation function ───────────────────────────────────
def run_segmentation(
    df: pd.DataFrame,
    n_clusters: int = 5,
    method: str = "kmeans",          # "kmeans" | "dbscan" | "hierarchical"
    force_retrain: bool = False,
) -> pd.DataFrame:
    """
    Adds three new columns to df:
      - segment_id      : integer cluster label
      - segment_label   : human-readable segment name
      - pca_x / pca_y   : 2-D PCA coords for scatter plot
    Returns the enriched DataFrame.
    """
    features = _available_features(df)
    if len(features) < 2:
        df["segment_id"]    = 0
        df["segment_label"] = "Unclassified"
        df["pca_x"] = 0.0
        df["pca_y"] = 0.0
        return df

    X, scaler = _prep_matrix(df, features)

    # ── Try loading cached model ──────────────────────────────────
    model_key = f"segmentation_{method}_{n_clusters}"
    cached = None if force_retrain else load_model(model_key)

    if cached is not None and isinstance(cached, dict):
        model = cached["model"]
    elif cached is not None:
        model = cached
    else:
        model = _train_model(X, n_clusters, method)
        save_model({"model": model, "scaler": scaler, "features": features}, model_key)

    # ── Assign labels ─────────────────────────────────────────────
    if method == "dbscan":
        labels = model.labels_
    elif hasattr(model, "predict"):
        labels = model.predict(X)
    elif hasattr(model, "labels_"):
        labels = model.labels_
    else:
        labels = np.zeros(len(X), dtype=int)

    # ── Map to human-readable names ───────────────────────────────
    # Use centroid-based heuristics to pick the best SEGMENT_LABELS key
    df["segment_label"] = _assign_human_labels(df, features, labels, n_clusters)

    # ── PCA for 2-D scatter ───────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]

    return df


def _train_model(X: np.ndarray, n_clusters: int, method: str):
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        model.fit(X)
    elif method == "dbscan":
        model = DBSCAN(eps=0.8, min_samples=5)
        model.fit(X)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        model.fit(X)
        # AgglomerativeClustering has no .predict(); store labels in model
        model.labels_ = model.labels_
    else:
        raise ValueError(f"Unknown method: {method}")
    return model


def _assign_human_labels(
    df: pd.DataFrame,
    features: list,
    labels: np.ndarray,
    n_clusters: int,
) -> pd.Series:
    """
    Map numeric cluster IDs to business-friendly names using
    per-cluster feature averages and simple threshold rules.

    Rule priority (highest wins):
      1. churn_proxy > 0.65                → Satisfaction-Risk Users
      2. loyalty_score > 0.65              → Loyal Repeat Buyers
      3. avg_spend high + freq high        → High-Value Customers
      4. browsing_intensity high + conv low→ Abandoned-Intent Users
      5. discount_sensitivity high         → Discount-Driven Shoppers
      6. browsing high + purchase low      → Explorers / Hesitant Buyers
      7. everything low                    → Low-Engagement Users
      8. fallback                          → generic label
    """
    result = pd.Series("Low-Engagement Users", index=df.index)
    unique_labels = sorted(set(labels))

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        if mask.sum() == 0:
            continue

        sub = df.loc[mask]

        def avg(col):
            return sub[col].mean() if col in sub.columns else 0.5

        churn   = avg("churn_proxy")
        loyal   = avg("loyalty_score")
        spend   = avg("monetary_proxy")
        freq    = avg("purchase_frequency") / 7.0   # normalise 0-1
        browse  = avg("browsing_intensity")
        conv    = avg("conversion_proxy")
        disc    = avg("discount_sensitivity")

        if churn > 0.65:
            label = "Satisfaction-Risk Users"
        elif loyal > 0.65:
            if spend > 0.6:
                label = "High-Value Customers"
            else:
                label = "Loyal Repeat Buyers"
        elif browse > 0.6 and conv < 0.35:
            label = "Abandoned-Intent Users"
        elif disc > 0.55:
            label = "Discount-Driven Shoppers"
        elif browse > 0.45 and freq < 0.35:
            label = "Explorers"
        elif browse > 0.3 and conv < 0.45:
            label = "Hesitant Buyers"
        else:
            label = "Low-Engagement Users"

        result[mask] = label

    return result


# ── Segment profile summary ───────────────────────────────────────
def get_segment_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per segment showing:
    - count, % share
    - mean of each SEG_FEATURE present
    - dominant category_preference (if available)
    - dominant payment_method (if available)
    """
    features = _available_features(df)
    if "segment_label" not in df.columns:
        return pd.DataFrame()

    agg = {f: "mean" for f in features if f in df.columns}
    agg["customer_id"] = "count" if "customer_id" in df.columns else None
    agg = {k: v for k, v in agg.items() if v is not None}

    profiles = df.groupby("segment_label").agg(agg).reset_index()
    profiles.rename(columns={"customer_id": "count"}, inplace=True)

    total = len(df)
    if "count" in profiles.columns:
        profiles["share_pct"] = (profiles["count"] / total * 100).round(1)

    # Dominant category
    if "category_preference" in df.columns:
        dom_cat = (
            df.groupby("segment_label")["category_preference"]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown")
            .reset_index()
            .rename(columns={"category_preference": "top_category"})
        )
        profiles = profiles.merge(dom_cat, on="segment_label", how="left")

    # Dominant payment
    if "payment_method" in df.columns:
        dom_pay = (
            df.groupby("segment_label")["payment_method"]
            .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown")
            .reset_index()
            .rename(columns={"payment_method": "top_payment"})
        )
        profiles = profiles.merge(dom_pay, on="segment_label", how="left")

    # Round numeric columns
    num_cols = profiles.select_dtypes(include="number").columns
    profiles[num_cols] = profiles[num_cols].round(3)

    return profiles


# ── Silhouette evaluation ─────────────────────────────────────────
def evaluate_clustering(df: pd.DataFrame) -> dict:
    """Returns silhouette score and cluster size breakdown."""
    features = _available_features(df)
    if "segment_id" not in df.columns or len(features) < 2:
        return {"silhouette": None, "cluster_sizes": {}}

    X, _ = _prep_matrix(df, features)
    labels = df["segment_id"].values

    try:
        score = silhouette_score(X, labels, sample_size=min(3000, len(X)))
    except Exception:
        score = None

    sizes = df["segment_label"].value_counts().to_dict()
    return {"silhouette": round(score, 4) if score else None, "cluster_sizes": sizes}


# ── Explainable segment rules ─────────────────────────────────────
SEGMENT_RULES = {
    "Loyal Repeat Buyers": {
        "rule": "High purchase frequency + high loyalty score + high satisfaction",
        "business_insight": "Best candidates for VIP programs and early-access offers.",
        "risk": "Low churn risk",
    },
    "Explorers": {
        "rule": "High browsing intensity + low purchase frequency",
        "business_insight": "Need richer product content and comparison tools to convert.",
        "risk": "Medium conversion risk",
    },
    "Discount-Driven Shoppers": {
        "rule": "High discount sensitivity + moderate conversion proxy",
        "business_insight": "Respond strongly to promotions; risk of margin erosion.",
        "risk": "Low churn, high discount dependency",
    },
    "Hesitant Buyers": {
        "rule": "Moderate browsing + low conversion + low purchase frequency",
        "business_insight": "Need trust signals, reviews, and simpler checkout flow.",
        "risk": "Medium-high abandonment risk",
    },
    "High-Value Customers": {
        "rule": "High avg_spend + high loyalty + high repeat purchase score",
        "business_insight": "Priority for retention; personalised service pays off here.",
        "risk": "Low churn risk",
    },
    "Low-Engagement Users": {
        "rule": "Low browsing + low purchase + low satisfaction",
        "business_insight": "Re-engagement campaigns or accept natural churn.",
        "risk": "High churn risk",
    },
    "Satisfaction-Risk Users": {
        "rule": "High churn proxy + low satisfaction score + low rating",
        "business_insight": "Immediate intervention — personalised apology / offer.",
        "risk": "High churn risk",
    },
    "Abandoned-Intent Users": {
        "rule": "High browsing intensity + low conversion proxy + discount interest",
        "business_insight": "Cart-abandonment nudges and limited-time offers can recover these.",
        "risk": "High abandonment proxy",
    },
}


def get_segment_rule(label: str) -> dict:
    return SEGMENT_RULES.get(label, {
        "rule": "No rule defined",
        "business_insight": "—",
        "risk": "Unknown",
    })