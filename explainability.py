# explainability.py
"""
Explainability layer for the Customer Journey Analysis system.

Provides four types of explanations consumed by the dashboard
and the recommendation engine:

  A. SHAP-based model explanations
       - global feature importance bar chart data
       - per-customer waterfall / force plot data

  B. Cluster / segment profile explanations
       - plain-English rule summary per segment
       - radar chart data per segment

  C. NLP evidence linking
       - connects TF-IDF keywords + sentiment signals
         to segment and risk labels

  D. Customer explanation cards
       - single-customer summary combining all of the above
       - used in the Session / User Drill-down page
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from utils import load_model, SEGMENT_COLORS
from segmentation import get_segment_rule, SEG_FEATURES


# ╔══════════════════════════════════════════════════════════════╗
# ║  A. SHAP explanations                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def get_global_feature_importance(
    model_name: str = "xgboost",
    top_n: int = 12,
) -> pd.DataFrame:
    """
    Returns a DataFrame sorted by mean |SHAP| value (or
    model.feature_importances_ if SHAP is unavailable).

    Columns: feature | importance | normalised_importance
    """
    pipeline = load_model(f"prediction_{model_name}")
    if pipeline is None:
        # Try fallback models
        for fallback in ["random_forest", "logistic_regression"]:
            pipeline = load_model(f"prediction_{fallback}")
            if pipeline is not None:
                model_name = fallback
                break

    if pipeline is None:
        return pd.DataFrame({"feature": [], "importance": []})

    inner = pipeline.named_steps["model"]

    if hasattr(inner, "feature_importances_"):
        importances = inner.feature_importances_
    elif hasattr(inner, "coef_"):
        importances = np.abs(inner.coef_[0])
    else:
        return pd.DataFrame({"feature": [], "importance": []})

    # Recover feature names from imputer if available
    if hasattr(pipeline.named_steps.get("imputer", {}), "feature_names_in_"):
        features = list(pipeline.named_steps["imputer"].feature_names_in_)
    else:
        features = [f"feature_{i}" for i in range(len(importances))]

    df_imp = pd.DataFrame({
        "feature":    features[:len(importances)],
        "importance": importances,
    }).sort_values("importance", ascending=False).head(top_n)

    total = df_imp["importance"].sum()
    df_imp["normalised_importance"] = (
        df_imp["importance"] / total * 100
    ).round(2)

    return df_imp.reset_index(drop=True)


def get_shap_waterfall_data(
    customer_row: pd.Series,
    model_name: str = "xgboost",
) -> dict:
    """
    Returns SHAP waterfall data for a single customer row.

    Output dict:
      {
        "base_value":   float,
        "final_value":  float,
        "features":     [str, ...],
        "shap_values":  [float, ...],
        "feature_vals": [float, ...],
        "error":        str | None,
      }

    If SHAP is unavailable, falls back to feature importance
    scaled by feature deviation from the population mean.
    """
    try:
        import shap
        SHAP_OK = True
    except ImportError:
        SHAP_OK = False

    pipeline = load_model(f"prediction_{model_name}")
    if pipeline is None:
        for fb in ["random_forest", "logistic_regression"]:
            pipeline = load_model(f"prediction_{fb}")
            if pipeline is not None:
                model_name = fb
                break

    if pipeline is None:
        return {"error": "No trained model found. Run train_models() first."}

    # Get feature list
    if hasattr(pipeline.named_steps.get("imputer", {}), "feature_names_in_"):
        features = list(pipeline.named_steps["imputer"].feature_names_in_)
    else:
        features = [c for c in SEG_FEATURES if c in customer_row.index]

    # Build single-row DataFrame
    row_df = pd.DataFrame([customer_row.reindex(features, fill_value=0)])

    # Run through pipeline up to (but not including) the final model
    try:
        X_transformed = pipeline[:-1].transform(row_df)
    except Exception as e:
        return {"error": f"Transform failed: {e}"}

    inner = pipeline.named_steps["model"]

    if SHAP_OK:
        try:
            explainer = shap.TreeExplainer(inner)
            sv = explainer.shap_values(X_transformed)
            if isinstance(sv, list):
                sv = sv[1]           # binary: take positive class
            base = (explainer.expected_value
                    if not isinstance(explainer.expected_value, list)
                    else explainer.expected_value[1])
            shap_vals = sv[0].tolist()
            final_val = float(base) + sum(shap_vals)

            return {
                "base_value":   round(float(base), 4),
                "final_value":  round(final_val, 4),
                "features":     features,
                "shap_values":  [round(v, 4) for v in shap_vals],
                "feature_vals": [round(float(customer_row.get(f, 0)), 4)
                                 for f in features],
                "error":        None,
            }
        except Exception as e:
            pass   # fall through to heuristic fallback

    # ── Heuristic fallback ────────────────────────────────────────
    # Approximate contribution = importance × signed deviation from mean
    imp_df  = get_global_feature_importance(model_name, top_n=len(features))
    imp_map = dict(zip(imp_df["feature"], imp_df["importance"]))

    # Use normalised risk_score as the final value proxy
    risk_score = float(customer_row.get("risk_score", 0.5))
    base_value = 0.5

    heuristic_shap = []
    for f in features:
        imp = imp_map.get(f, 0.0)
        val = float(customer_row.get(f, 0.5))
        # signed deviation: above 0.5 = pushes toward risk if importance is high
        signed = imp * (val - 0.5) * 2
        heuristic_shap.append(round(signed, 4))

    return {
        "base_value":   round(base_value, 4),
        "final_value":  round(risk_score, 4),
        "features":     features,
        "shap_values":  heuristic_shap,
        "feature_vals": [round(float(customer_row.get(f, 0)), 4)
                         for f in features],
        "error":        "heuristic_fallback (SHAP not available)",
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. Segment / cluster explanations                           ║
# ╚══════════════════════════════════════════════════════════════╝

def get_segment_radar_data(
    df: pd.DataFrame,
    segment_label: str,
) -> dict:
    """
    Returns data for a radar / spider chart for one segment vs
    the population average.

    Output:
      {
        "axes":        [feature_name, ...],
        "segment_avg": [float, ...],
        "global_avg":  [float, ...],
        "label":       segment_label,
        "color":       hex_color,
      }
    """
    radar_features = [
        "purchase_frequency", "avg_spend", "browsing_intensity",
        "loyalty_score", "satisfaction_score", "discount_sensitivity",
        "conversion_proxy", "churn_proxy",
    ]
    available = [f for f in radar_features if f in df.columns]

    if "segment_label" not in df.columns or not available:
        return {}

    seg_df   = df[df["segment_label"] == segment_label]
    glob_avg = df[available].mean().tolist()
    seg_avg  = seg_df[available].mean().tolist() if not seg_df.empty else glob_avg

    # Normalise to 0-1 for radar display
    def norm_list(lst):
        mn, mx = min(lst + glob_avg), max(lst + glob_avg)
        if mx == mn:
            return [0.5] * len(lst)
        return [round((v - mn) / (mx - mn), 3) for v in lst]

    return {
        "axes":         available,
        "segment_avg":  norm_list(seg_avg),
        "global_avg":   norm_list(glob_avg),
        "label":        segment_label,
        "color":        SEGMENT_COLORS.get(segment_label, "#6C5CE7"),
        "count":        len(seg_df),
        "rule":         get_segment_rule(segment_label),
    }


def get_all_segment_profiles_explained(df: pd.DataFrame) -> list:
    """
    Returns a list of radar dicts for every segment present in df.
    """
    if "segment_label" not in df.columns:
        return []
    return [
        get_segment_radar_data(df, label)
        for label in df["segment_label"].unique()
    ]


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. NLP evidence linking                                     ║
# ╚══════════════════════════════════════════════════════════════╝

def get_nlp_evidence(
    df: pd.DataFrame,
    segment_label: str = None,
    risk_level: str = None,
    top_n: int = 10,
) -> dict:
    """
    Surfaces NLP signals that characterise a given segment
    or risk group.

    Returns:
      {
        "top_intents":       [(intent, count), ...],
        "top_keywords":      [(keyword, score), ...],
        "sentiment_mix":     {"Positive": %, "Neutral": %, "Negative": %},
        "complaint_rate":    float,
        "frustration_rate":  float,
        "pricing_rate":      float,
        "trust_rate":        float,
        "example_texts":     [str, ...],   ← up to 3 example snippets
      }
    """
    sub = df.copy()
    if segment_label and "segment_label" in sub.columns:
        sub = sub[sub["segment_label"] == segment_label]
    if risk_level and "risk_level" in sub.columns:
        sub = sub[sub["risk_level"] == risk_level]

    if sub.empty:
        return {}

    n = len(sub)

    # Intent distribution
    top_intents = []
    if "intent_label" in sub.columns:
        top_intents = (
            sub["intent_label"].value_counts().head(5)
              .reset_index()
              .rename(columns={"intent_label": "intent", "count": "count"})
              .values.tolist()
        )

    # Sentiment mix
    sentiment_mix = {}
    if "sentiment_label" in sub.columns:
        vc = sub["sentiment_label"].value_counts()
        sentiment_mix = {
            lbl: round(vc.get(lbl, 0) / n * 100, 1)
            for lbl in ["Positive", "Neutral", "Negative"]
        }

    # Signal rates
    def rate(col):
        if col in sub.columns:
            return round(sub[col].mean() * 100, 1) if sub[col].dtype != object else 0.0
        return 0.0

    complaint_rate   = round((sub["complaint_count"] > 0).mean() * 100, 1) \
                       if "complaint_count" in sub.columns else 0.0

    # Top keywords from TF-IDF
    top_keywords = []
    if "clean_text" in sub.columns:
        from nlp_engine import get_top_keywords
        kw_df = get_top_keywords(sub, n=top_n)
        top_keywords = list(zip(kw_df["keyword"], kw_df["tfidf_score"].round(4)))

    # Example review snippets (up to 3 with actual text)
    example_texts = []
    text_col = next(
        (c for c in ["clean_text", "review_text"] if c in sub.columns), None
    )
    if text_col:
        sample = sub[sub[text_col].str.len() > 20][text_col].head(3).tolist()
        example_texts = [t[:150] + ("…" if len(t) > 150 else "") for t in sample]

    return {
        "top_intents":      top_intents,
        "top_keywords":     top_keywords,
        "sentiment_mix":    sentiment_mix,
        "complaint_rate":   complaint_rate,
        "frustration_rate": rate("frustration_flag"),
        "pricing_rate":     rate("pricing_concern"),
        "trust_rate":       rate("trust_issue"),
        "example_texts":    example_texts,
        "n_customers":      n,
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║  D. Customer explanation card                                ║
# ╚══════════════════════════════════════════════════════════════╝

def build_customer_card(
    customer_row: pd.Series,
    df_full: pd.DataFrame,
    model_name: str = "xgboost",
) -> dict:
    """
    Produces a complete explanation card for one customer.
    Used in the Session / User Drill-down page.

    Returns a dict with sections:
      profile        : key feature values
      journey        : inferred stage + stage description
      risk           : score, level, SHAP waterfall data
      segment        : label, rule, radar data
      nlp            : sentiment, top intents, example text
      similar_cases  : placeholder (filled by faiss_engine)
    """
    card = {}

    # ── Profile ───────────────────────────────────────────────────
    profile_fields = [
        "customer_id", "age", "gender", "age_group",
        "category_preference", "payment_method",
        "avg_spend", "purchase_frequency",
        "subscription_status", "location",
    ]
    card["profile"] = {
        f: _fmt(customer_row.get(f))
        for f in profile_fields
        if f in customer_row.index
    }

    # ── Journey stage ─────────────────────────────────────────────
    stage = customer_row.get("journey_stage", "Unknown")
    card["journey"] = {
        "stage":       stage,
        "stage_index": {s: i for i, s in enumerate(
            ["Awareness","Exploration","Consideration",
             "Cart Intent","Checkout Intent","Purchase","Post-Purchase"]
        )}.get(stage, 0),
        "description": _stage_description(stage),
        "abandonment_flag": bool(customer_row.get("abandonment_proxy", 0)),
    }

    # ── Risk ──────────────────────────────────────────────────────
    card["risk"] = {
        "score":       round(float(customer_row.get("risk_score", 0.5)), 4),
        "level":       customer_row.get("risk_level", "Unknown"),
        "model_used":  customer_row.get("risk_model_used", "unknown"),
        "churn_proxy": round(float(customer_row.get("churn_proxy", 0.5)), 3),
        "waterfall":   get_shap_waterfall_data(customer_row, model_name),
    }

    # ── Segment ───────────────────────────────────────────────────
    seg_label = customer_row.get("segment_label", "Unknown")
    card["segment"] = {
        "label":  seg_label,
        "color":  SEGMENT_COLORS.get(seg_label, "#888"),
        "rule":   get_segment_rule(seg_label),
        "radar":  get_segment_radar_data(df_full, seg_label),
    }

    # ── NLP evidence ──────────────────────────────────────────────
    card["nlp"] = {
        "sentiment_score":  round(float(customer_row.get("sentiment_score", 0)), 3),
        "sentiment_label":  customer_row.get("sentiment_label", "Neutral"),
        "intent_label":     customer_row.get("intent_label",    "neutral"),
        "complaint_count":  int(customer_row.get("complaint_count", 0)),
        "urgency_score":    round(float(customer_row.get("urgency_score", 0)), 3),
        "frustration":      bool(customer_row.get("frustration_flag", 0)),
        "pricing_concern":  bool(customer_row.get("pricing_concern", 0)),
        "trust_issue":      bool(customer_row.get("trust_issue", 0)),
        "review_snippet":   _get_snippet(customer_row),
        "topic_label":      customer_row.get("topic_label", "—"),
    }

    # ── Similar cases placeholder ─────────────────────────────────
    # Filled at dashboard render time via faiss_engine
    card["similar_cases"] = []

    return card


# ── Internal helpers ──────────────────────────────────────────────

def _fmt(val) -> str:
    """Format a field value for display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if isinstance(val, float):
        return str(round(val, 3))
    return str(val)


def _get_snippet(row: pd.Series, max_len: int = 200) -> str:
    for col in ["clean_text", "review_text", "feedback"]:
        val = row.get(col, "")
        if isinstance(val, str) and len(val.strip()) > 10:
            return val[:max_len] + ("…" if len(val) > max_len else "")
    return "No review text available."


def _stage_description(stage: str) -> str:
    return {
        "Awareness":       "Customer has low engagement — barely browsing. Needs discovery nudges.",
        "Exploration":     "Browsing actively but no strong purchase signals yet.",
        "Consideration":   "Showing interest in specific categories. Comparison / review phase.",
        "Cart Intent":     "High browsing with discount sensitivity — likely evaluating price.",
        "Checkout Intent": "Near-conversion. Payment friction or hesitation is the main blocker.",
        "Purchase":        "Has completed a purchase. Repeat purchase likelihood is the key metric.",
        "Post-Purchase":   "Satisfaction and loyalty signals determine retention vs churn.",
    }.get(stage, "Stage unknown — insufficient signal from available features.")


# ── Convenience: global explanation summary for dashboard ─────────
def get_global_explanation_summary(df: pd.DataFrame) -> dict:
    """
    Single call that returns everything the Explainability page needs:
      - global feature importance
      - all segment radar data
      - NLP evidence for each segment
      - top risk drivers (features with highest positive SHAP contrib)
    """
    summary = {}

    # Feature importance
    summary["feature_importance"] = get_global_feature_importance(top_n=12)

    # Segment radars
    summary["segment_radars"] = get_all_segment_profiles_explained(df)

    # NLP evidence per segment
    if "segment_label" in df.columns:
        summary["segment_nlp"] = {
            label: get_nlp_evidence(df, segment_label=label)
            for label in df["segment_label"].unique()
        }
    else:
        summary["segment_nlp"] = {}

    # Top risk drivers (features with highest importance)
    imp = summary["feature_importance"]
    if not imp.empty:
        summary["top_risk_drivers"] = imp.head(5)["feature"].tolist()
    else:
        summary["top_risk_drivers"] = []

    return summary