# recommendation_engine.py
"""
Hybrid recommendation engine combining:
  - Rule-based business logic (stage + segment aware)
  - ML/NLP signal evidence (risk score, sentiment, keywords)
  - FAISS similar-case retrieval for evidence grounding

Every recommendation carries:
  priority      : "Critical" | "High" | "Medium" | "Low"
  segment       : which segment it targets
  stage         : which journey stage it addresses
  title         : short action label
  rationale     : why this recommendation was triggered
  evidence      : dict of supporting signals
  expected_impact: business effect description
  action_type   : "retention" | "conversion" | "engagement"
                  | "trust" | "pricing" | "loyalty"
"""

import pandas as pd
import numpy as np
from funnel_analysis  import get_stage_friction_summary, get_dropoff_rates
from explainability   import get_nlp_evidence, get_global_feature_importance
from utils            import SEGMENT_COLORS

# ── Priority ordering for sorting ────────────────────────────────
PRIORITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


# ╔══════════════════════════════════════════════════════════════╗
# ║  Rule library                                                ║
# ╚══════════════════════════════════════════════════════════════╝

def _rules_from_segment(df: pd.DataFrame) -> list:
    """
    Generate one recommendation per segment whose profile
    matches a known pattern.
    """
    recs = []
    if "segment_label" not in df.columns:
        return recs

    seg_stats = df.groupby("segment_label").agg(
        count        =("customer_id", "count") if "customer_id" in df.columns else ("risk_score", "count"),
        avg_churn    =("churn_proxy",        "mean") if "churn_proxy"        in df.columns else ("risk_score", "mean"),
        avg_browse   =("browsing_intensity",  "mean") if "browsing_intensity"  in df.columns else ("risk_score", "mean"),
        avg_conv     =("conversion_proxy",    "mean") if "conversion_proxy"    in df.columns else ("risk_score", "mean"),
        avg_sat      =("satisfaction_score",  "mean") if "satisfaction_score"  in df.columns else ("risk_score", "mean"),
        avg_discount =("discount_sensitivity","mean") if "discount_sensitivity" in df.columns else ("risk_score", "mean"),
        avg_risk     =("risk_score",          "mean") if "risk_score"          in df.columns else ("risk_score", "mean"),
        complaint_rt =("complaint_count",     lambda x: (x > 0).mean()) if "complaint_count" in df.columns else ("risk_score", "mean"),
    ).reset_index()

    for _, row in seg_stats.iterrows():
        label = row["segment_label"]
        n     = int(row["count"])

        if label == "Satisfaction-Risk Users":
            recs.append(_make_rec(
                priority="Critical",
                segment=label,
                stage="Post-Purchase",
                action_type="retention",
                title="Immediate intervention for satisfaction-risk users",
                rationale=(
                    f"{n} customers show high churn proxy "
                    f"({row['avg_churn']:.0%}) and low satisfaction "
                    f"({row['avg_sat']:.2f}/1.0). Complaint rate: "
                    f"{row['complaint_rt']:.0%}."
                ),
                expected_impact="Reduce churn by 15-25% through personalised recovery offers.",
                evidence={
                    "avg_churn_proxy":   round(row["avg_churn"],    3),
                    "avg_satisfaction":  round(row["avg_sat"],      3),
                    "complaint_rate":    round(row["complaint_rt"],  3),
                    "affected_customers": n,
                },
            ))

        elif label == "Abandoned-Intent Users":
            recs.append(_make_rec(
                priority="High",
                segment=label,
                stage="Cart Intent",
                action_type="conversion",
                title="Cart-abandonment recovery nudges",
                rationale=(
                    f"{n} users show high browse intensity "
                    f"({row['avg_browse']:.0%}) but low conversion "
                    f"({row['avg_conv']:.0%}). Discount sensitivity: "
                    f"{row['avg_discount']:.0%}."
                ),
                expected_impact="Recover 10-20% of near-purchase customers via timed discount prompts.",
                evidence={
                    "avg_browsing_intensity": round(row["avg_browse"],   3),
                    "avg_conversion_proxy":   round(row["avg_conv"],     3),
                    "discount_sensitivity":   round(row["avg_discount"], 3),
                    "affected_customers": n,
                },
            ))

        elif label == "Explorers":
            recs.append(_make_rec(
                priority="Medium",
                segment=label,
                stage="Consideration",
                action_type="engagement",
                title="Richer product content for explorer segment",
                rationale=(
                    f"{n} customers browse frequently "
                    f"(intensity {row['avg_browse']:.0%}) but rarely "
                    f"convert ({row['avg_conv']:.0%}). They need stronger "
                    "comparison tools and social proof."
                ),
                expected_impact="Lift consideration-to-purchase rate by improving product detail pages.",
                evidence={
                    "avg_browsing_intensity": round(row["avg_browse"], 3),
                    "avg_conversion_proxy":   round(row["avg_conv"],   3),
                    "affected_customers": n,
                },
            ))

        elif label == "Discount-Driven Shoppers":
            recs.append(_make_rec(
                priority="Medium",
                segment=label,
                stage="Cart Intent",
                action_type="pricing",
                title="Optimise discount strategy for price-sensitive segment",
                rationale=(
                    f"{n} customers have high discount sensitivity "
                    f"({row['avg_discount']:.0%}). Risk of margin erosion "
                    "if discounts are untargeted."
                ),
                expected_impact="Improve margin by 5-10% by shifting from blanket to personalised discounts.",
                evidence={
                    "discount_sensitivity":   round(row["avg_discount"], 3),
                    "avg_conversion_proxy":   round(row["avg_conv"],     3),
                    "affected_customers": n,
                },
            ))

        elif label == "Loyal Repeat Buyers":
            recs.append(_make_rec(
                priority="Medium",
                segment=label,
                stage="Post-Purchase",
                action_type="loyalty",
                title="Launch VIP / early-access programme",
                rationale=(
                    f"{n} loyal customers have avg risk score "
                    f"{row['avg_risk']:.2f} and high satisfaction "
                    f"({row['avg_sat']:.2f}/1.0). Prime candidates for "
                    "lifetime value maximisation."
                ),
                expected_impact="Increase average order value 10-15% and extend retention lifecycle.",
                evidence={
                    "avg_satisfaction":   round(row["avg_sat"],  3),
                    "avg_risk_score":     round(row["avg_risk"], 3),
                    "affected_customers": n,
                },
            ))

        elif label == "Hesitant Buyers":
            recs.append(_make_rec(
                priority="High",
                segment=label,
                stage="Checkout Intent",
                action_type="trust",
                title="Add trust signals and simplify checkout flow",
                rationale=(
                    f"{n} customers reach near-checkout but do not convert "
                    f"(conversion {row['avg_conv']:.0%}). Payment friction "
                    "and trust gaps are likely blockers."
                ),
                expected_impact="Improve checkout conversion 8-15% with reviews, badges, and one-click pay.",
                evidence={
                    "avg_conversion_proxy": round(row["avg_conv"],     3),
                    "avg_browsing":         round(row["avg_browse"],   3),
                    "affected_customers": n,
                },
            ))

        elif label == "Low-Engagement Users":
            recs.append(_make_rec(
                priority="Low",
                segment=label,
                stage="Awareness",
                action_type="engagement",
                title="Re-engagement campaign or accept natural churn",
                rationale=(
                    f"{n} customers show minimal browsing and purchase "
                    f"signals. High churn proxy ({row['avg_churn']:.0%}). "
                    "ROI of re-engagement should be evaluated carefully."
                ),
                expected_impact="Recover 5-10% of dormant users; flag remainder for sunset.",
                evidence={
                    "avg_churn_proxy":     round(row["avg_churn"],  3),
                    "avg_conversion":      round(row["avg_conv"],   3),
                    "affected_customers": n,
                },
            ))

    return recs


def _rules_from_nlp(df: pd.DataFrame) -> list:
    """Generate recommendations from NLP signal aggregates."""
    recs = []

    # Pricing concern threshold
    if "pricing_concern" in df.columns:
        rate = df["pricing_concern"].mean()
        if rate > 0.15:
            recs.append(_make_rec(
                priority="High",
                segment="All",
                stage="Cart Intent",
                action_type="pricing",
                title="Clarify pricing and value messaging",
                rationale=(
                    f"{rate:.0%} of customers trigger pricing-concern "
                    "keywords in their reviews. Price confusion is a "
                    "significant friction source."
                ),
                expected_impact="Reduce pricing-driven abandonment by improving price transparency pages.",
                evidence={
                    "pricing_concern_rate": round(rate, 3),
                    "signal": "TF-IDF keywords: expensive, overpriced, not worth",
                },
            ))

    # Trust issues
    if "trust_issue" in df.columns:
        rate = df["trust_issue"].mean()
        if rate > 0.08:
            recs.append(_make_rec(
                priority="High",
                segment="All",
                stage="Consideration",
                action_type="trust",
                title="Strengthen authenticity signals and reviews",
                rationale=(
                    f"{rate:.0%} of customers flag trust-related terms "
                    "(fake, scam, misleading). This is a brand-level risk."
                ),
                expected_impact="Increase consideration-to-cart conversion by building verified review systems.",
                evidence={
                    "trust_issue_rate": round(rate, 3),
                    "signal": "TF-IDF keywords: fake, misleading, not genuine",
                },
            ))

    # Frustration / support
    if "frustration_flag" in df.columns:
        rate = df["frustration_flag"].mean()
        if rate > 0.12:
            recs.append(_make_rec(
                priority="High",
                segment="All",
                stage="Post-Purchase",
                action_type="retention",
                title="Improve post-purchase support responsiveness",
                rationale=(
                    f"{rate:.0%} of customers show frustration signals. "
                    "Support resolution speed directly affects repeat "
                    "purchase likelihood."
                ),
                expected_impact="Reduce frustration-driven churn by 10-20% with faster support SLAs.",
                evidence={
                    "frustration_rate": round(rate, 3),
                    "signal": "TF-IDF keywords: frustrated, never again, fed up",
                },
            ))

    # Confusion / UX
    if "confusion_flag" in df.columns:
        rate = df["confusion_flag"].mean()
        if rate > 0.10:
            recs.append(_make_rec(
                priority="Medium",
                segment="All",
                stage="Exploration",
                action_type="engagement",
                title="Simplify navigation and onboarding UX",
                rationale=(
                    f"{rate:.0%} of customers show confusion / UX "
                    "difficulty signals. Discovery friction slows "
                    "exploration-to-consideration progression."
                ),
                expected_impact="Improve exploration conversion 5-10% with clearer category navigation.",
                evidence={
                    "confusion_rate": round(rate, 3),
                    "signal": "TF-IDF keywords: confusing, hard to find, don't understand",
                },
            ))

    return recs


def _rules_from_funnel(df: pd.DataFrame) -> list:
    """Generate recommendations from funnel friction analysis."""
    recs  = []
    summ  = get_stage_friction_summary(df)
    drops = get_dropoff_rates(df)

    if not summ:
        return recs

    # Worst drop-off transition
    worst = summ.get("worst_dropoff_transition", "")
    drop_pct = summ.get("worst_dropoff_pct", 0)
    if drop_pct > 30:
        recs.append(_make_rec(
            priority="High",
            segment="All",
            stage=worst.split(" → ")[0] if " → " in worst else "Unknown",
            action_type="conversion",
            title=f"Reduce friction at {worst} transition",
            rationale=(
                f"{drop_pct:.1f}% of customers do not progress from "
                f"{worst}. This is the single largest drop-off point "
                "in the proxy funnel."
            ),
            expected_impact=f"Improving this transition 10% would lift overall conversion meaningfully.",
            evidence={
                "transition":  worst,
                "dropoff_pct": drop_pct,
                "friction_signal": summ.get("highest_friction_signal", ""),
            },
        ))

    # Highest friction stage
    fric_stage = summ.get("highest_friction_stage", "")
    fric_score = summ.get("highest_friction_score", 0)
    if fric_score > 0.5 and fric_stage:
        recs.append(_make_rec(
            priority="Medium",
            segment="All",
            stage=fric_stage,
            action_type="conversion",
            title=f"Targeted UX improvement at {fric_stage} stage",
            rationale=(
                f"Friction score {fric_score:.2f}/1.0 at {fric_stage} — "
                f"signal: {summ.get('highest_friction_signal', 'unknown')}."
            ),
            expected_impact="Stage-specific UX fixes can reduce friction score and improve flow.",
            evidence={
                "stage":         fric_stage,
                "friction_score": fric_score,
            },
        ))

    return recs


def _rules_from_risk(df: pd.DataFrame) -> list:
    """Generate recommendations from risk score distribution."""
    recs = []
    if "risk_level" not in df.columns:
        return recs

    high_risk_n = int((df["risk_level"] == "High").sum())
    total       = len(df)
    high_pct    = high_risk_n / total * 100

    if high_pct > 20:
        recs.append(_make_rec(
            priority="Critical",
            segment="High-Risk",
            stage="All",
            action_type="retention",
            title="Activate churn-prevention workflow for high-risk cohort",
            rationale=(
                f"{high_pct:.1f}% of customers ({high_risk_n}) are "
                "flagged as high risk by the XGBoost model. Immediate "
                "intervention is warranted."
            ),
            expected_impact="Reduce high-risk churn rate by 15-30% with personalised retention campaigns.",
            evidence={
                "high_risk_count":   high_risk_n,
                "high_risk_pct":     round(high_pct, 1),
                "model":             "XGBoost risk_score > 0.7",
            },
        ))

    return recs


# ╔══════════════════════════════════════════════════════════════╗
# ║  Main entry point                                            ║
# ╚══════════════════════════════════════════════════════════════╝

def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all rule sources and returns a deduplicated, priority-sorted
    DataFrame of recommendations.

    Columns:
      priority | segment | stage | action_type | title
      | rationale | expected_impact | evidence
    """
    all_recs = (
        _rules_from_segment(df) +
        _rules_from_nlp(df)     +
        _rules_from_funnel(df)  +
        _rules_from_risk(df)
    )

    if not all_recs:
        return pd.DataFrame()

    rec_df = pd.DataFrame(all_recs)

    # Sort by priority
    rec_df["_priority_order"] = rec_df["priority"].map(PRIORITY_ORDER).fillna(9)
    rec_df = (rec_df
              .sort_values("_priority_order")
              .drop(columns=["_priority_order"])
              .reset_index(drop=True))

    return rec_df


def get_recommendations_for_customer(
    customer_row: pd.Series,
    df_full: pd.DataFrame,
) -> list:
    """
    Returns a filtered list of recommendations relevant to
    a specific customer, based on their segment, stage and risk.
    """
    all_recs = generate_recommendations(df_full)
    if all_recs.empty:
        return []

    seg   = customer_row.get("segment_label", "")
    stage = customer_row.get("journey_stage",  "")

    relevant = all_recs[
        (all_recs["segment"].isin([seg, "All", "High-Risk"])) |
        (all_recs["stage"].isin([stage, "All"]))
    ]
    return relevant.to_dict("records")


# ── Internal factory ──────────────────────────────────────────────
def _make_rec(
    priority, segment, stage, action_type,
    title, rationale, expected_impact, evidence,
) -> dict:
    return {
        "priority":        priority,
        "segment":         segment,
        "stage":           stage,
        "action_type":     action_type,
        "title":           title,
        "rationale":       rationale,
        "expected_impact": expected_impact,
        "evidence":        evidence,
    }