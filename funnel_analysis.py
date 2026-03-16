# funnel_analysis.py
"""
Proxy-based Customer Journey Funnel Analysis.

IMPORTANT: All funnel stages are INFERRED from behavioural proxy
features (browsing_intensity, conversion_proxy, discount_sensitivity,
satisfaction_score, loyalty_score). This system does NOT use real
clickstream or page-level navigation data — those columns do not
exist in the source datasets. Every stage and drop-off metric
produced here is clearly labelled as a proxy / approximation.

Outputs used by the dashboard:
  - get_funnel_counts()         → stage bar / funnel chart
  - get_dropoff_rates()         → drop-off % between stages
  - get_friction_indicators()   → per-stage friction score table
  - get_category_funnel()       → funnel breakdown by product category
  - get_sankey_data()           → Plotly Sankey source/target/value lists
  - get_cohort_retention()      → retention-proxy heatmap data
  - get_conversion_summary()    → single-number KPIs
  - get_abandonment_heatmap()   → segment × stage abandonment matrix
"""

import pandas as pd
import numpy as np
from utils import JOURNEY_STAGES, STAGE_COLORS


# ── Stage ordering map ────────────────────────────────────────────
STAGE_ORDER = {s: i for i, s in enumerate(JOURNEY_STAGES)}


def _stage_position(stage: str) -> int:
    return STAGE_ORDER.get(stage, -1)


# ── 1. Funnel stage counts ────────────────────────────────────────
def get_funnel_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns ordered DataFrame:
      stage | count | cumulative_pct | stage_index

    NOTE: stage column is populated by feature_engineering.py
    using proxy rules — not true clickstream.
    """
    if "journey_stage" not in df.columns:
        return pd.DataFrame({
            "stage": JOURNEY_STAGES,
            "count": [0] * len(JOURNEY_STAGES),
        })

    vc = df["journey_stage"].value_counts()
    rows = []
    for stage in JOURNEY_STAGES:
        rows.append({
            "stage":       stage,
            "count":       int(vc.get(stage, 0)),
            "stage_index": STAGE_ORDER[stage],
            "color":       STAGE_COLORS[STAGE_ORDER[stage]],
        })

    result = pd.DataFrame(rows)
    total  = result["count"].sum()
    result["pct_of_total"] = (result["count"] / total * 100).round(1) if total > 0 else 0.0
    result["cumulative_pct"] = result["count"].cumsum() / total * 100 if total > 0 else 0.0
    return result


# ── 2. Drop-off rates between stages ─────────────────────────────
def get_dropoff_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns DataFrame:
      from_stage | to_stage | from_count | to_count
      | dropoff_count | dropoff_pct | conversion_rate

    Drop-off is defined as: customers at stage N who do NOT
    appear at stage N+1. Because stages are inferred (not tracked),
    this approximates funnel efficiency, not exact step-loss.
    """
    counts = get_funnel_counts(df)
    rows   = []

    for i in range(len(JOURNEY_STAGES) - 1):
        from_stage = JOURNEY_STAGES[i]
        to_stage   = JOURNEY_STAGES[i + 1]
        from_count = int(counts.loc[counts["stage"] == from_stage, "count"].values[0])
        to_count   = int(counts.loc[counts["stage"] == to_stage,   "count"].values[0])
        dropoff    = max(from_count - to_count, 0)
        conv_rate  = round(to_count / from_count * 100, 1) if from_count > 0 else 0.0
        drop_pct   = round(dropoff  / from_count * 100, 1) if from_count > 0 else 0.0

        rows.append({
            "from_stage":       from_stage,
            "to_stage":         to_stage,
            "from_count":       from_count,
            "to_count":         to_count,
            "dropoff_count":    dropoff,
            "dropoff_pct":      drop_pct,
            "conversion_rate":  conv_rate,
        })

    return pd.DataFrame(rows)


# ── 3. Friction indicators per stage ─────────────────────────────
def get_friction_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a friction_score (0-1) for each inferred stage
    based on available proxy features. Higher score = more friction.

    Friction signals by stage:
      Awareness        : low browsing_intensity
      Exploration      : high browsing + low product_diversity
      Consideration    : high browsing + low conversion_proxy
      Cart Intent      : high discount_sensitivity + low conversion
      Checkout Intent  : low conversion_proxy + high churn_proxy
      Purchase         : low repeat_purchase_score
      Post-Purchase    : low satisfaction_score + low rating
    """
    if "journey_stage" not in df.columns:
        return pd.DataFrame({
            "stage":         JOURNEY_STAGES,
            "friction_score": [0.5] * len(JOURNEY_STAGES),
            "signal":        ["no data"] * len(JOURNEY_STAGES),
        })

    rows = []
    for stage in JOURNEY_STAGES:
        sub = df[df["journey_stage"] == stage]
        if sub.empty:
            rows.append({"stage": stage, "friction_score": 0.0,
                         "customer_count": 0, "signal": "no customers"})
            continue

        def avg(col, default=0.5):
            return sub[col].mean() if col in sub.columns else default

        if stage == "Awareness":
            score  = 1 - avg("browsing_intensity")
            signal = "low browse engagement"
        elif stage == "Exploration":
            score  = avg("browsing_intensity") * (1 - avg("product_diversity"))
            signal = "wide browse, narrow discovery"
        elif stage == "Consideration":
            score  = avg("browsing_intensity") * (1 - avg("conversion_proxy"))
            signal = "intent without purchase momentum"
        elif stage == "Cart Intent":
            score  = avg("discount_sensitivity") * (1 - avg("conversion_proxy"))
            signal = "price sensitivity blocking checkout"
        elif stage == "Checkout Intent":
            score  = (1 - avg("conversion_proxy") + avg("churn_proxy")) / 2
            signal = "near-purchase friction / payment hesitation"
        elif stage == "Purchase":
            score  = 1 - avg("repeat_purchase_score")
            signal = "single-purchase risk, low repeat tendency"
        elif stage == "Post-Purchase":
            score  = (1 - avg("satisfaction_score") + (1 - avg("rating"))) / 2
            signal = "satisfaction / rating risk"
        else:
            score  = 0.5
            signal = "unknown"

        rows.append({
            "stage":          stage,
            "friction_score": round(float(np.clip(score, 0, 1)), 3),
            "customer_count": len(sub),
            "signal":         signal,
            "avg_satisfaction": round(avg("satisfaction_score"), 3),
            "avg_conversion":  round(avg("conversion_proxy"),   3),
            "avg_churn":       round(avg("churn_proxy"),        3),
        })

    return pd.DataFrame(rows)


# ── 4. Category-level funnel breakdown ───────────────────────────
def get_category_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each product category, shows how many customers sit
    at each inferred journey stage.

    Returns a pivoted DataFrame:
      category | Awareness | Exploration | … | Post-Purchase | top_stage
    """
    if "journey_stage" not in df.columns or "category_preference" not in df.columns:
        return pd.DataFrame()

    pivot = (
        df.groupby(["category_preference", "journey_stage"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )

    # Ensure all stage columns exist
    for s in JOURNEY_STAGES:
        if s not in pivot.columns:
            pivot[s] = 0

    pivot = pivot[["category_preference"] + JOURNEY_STAGES]

    # Add conversion proxy per category
    pivot["total"]          = pivot[JOURNEY_STAGES].sum(axis=1)
    pivot["purchase_rate"]  = (
        pivot["Purchase"] / pivot["total"] * 100
    ).round(1).fillna(0)
    pivot["top_stage"]      = pivot[JOURNEY_STAGES].idxmax(axis=1)

    return pivot.sort_values("purchase_rate", ascending=False).reset_index(drop=True)


# ── 5. Sankey diagram data ────────────────────────────────────────
def get_sankey_data(df: pd.DataFrame) -> dict:
    """
    Produces source/target/value lists for a Plotly Sankey chart
    showing inferred flow between consecutive journey stages.

    Because we have no actual transition records, the flow values
    are the *minimum* of adjacent stage counts — a conservative
    estimate of how many customers moved through.

    Returns:
      {
        "labels":  [...stage names...],
        "source":  [int, ...],
        "target":  [int, ...],
        "value":   [int, ...],
        "colors":  [...hex colors...],
      }
    """
    counts = get_funnel_counts(df)
    count_map = {
        row["stage"]: row["count"]
        for _, row in counts.iterrows()
    }

    labels  = JOURNEY_STAGES
    source  = []
    target  = []
    value   = []
    link_colors = []

    for i in range(len(JOURNEY_STAGES) - 1):
        s_stage = JOURNEY_STAGES[i]
        t_stage = JOURNEY_STAGES[i + 1]
        s_count = count_map.get(s_stage, 0)
        t_count = count_map.get(t_stage, 0)

        # Flow = min(from, to) — conservative proxy
        flow = min(s_count, t_count)
        if flow > 0:
            source.append(i)
            target.append(i + 1)
            value.append(flow)
            link_colors.append(STAGE_COLORS[i] + "88")   # semi-transparent

    return {
        "labels": labels,
        "source": source,
        "target": target,
        "value":  value,
        "node_colors": STAGE_COLORS[:len(labels)],
        "link_colors": link_colors,
    }


# ── 6. Retention / cohort proxy ───────────────────────────────────
def get_cohort_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a pseudo-cohort retention table by binning customers
    into cohorts based on available temporal proxy:
      - age_group (if available)
      - purchase_frequency bin
      - category_preference

    Returns a pivot table where:
      rows    = cohort label
      columns = journey stage
      values  = % of cohort reaching that stage

    Clearly labelled as a PROXY — not true cohort tracking.
    """
    # Choose cohort dimension
    if "age_group" in df.columns and df["age_group"].nunique() >= 2:
        cohort_col = "age_group"
    elif "category_preference" in df.columns and df["category_preference"].nunique() >= 2:
        cohort_col = "category_preference"
    elif "purchase_frequency" in df.columns:
        df = df.copy()
        df["_freq_bin"] = pd.qcut(
            df["purchase_frequency"].fillna(0),
            q=4,
            labels=["Low freq", "Mid-low freq", "Mid-high freq", "High freq"],
            duplicates="drop",
        ).astype(str)
        cohort_col = "_freq_bin"
    else:
        return pd.DataFrame()

    if "journey_stage" not in df.columns:
        return pd.DataFrame()

    # Count per cohort × stage
    cross = (
        df.groupby([cohort_col, "journey_stage"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    for s in JOURNEY_STAGES:
        if s not in cross.columns:
            cross[s] = 0

    cross = cross[[cohort_col] + JOURNEY_STAGES]
    cross.columns = ["cohort"] + JOURNEY_STAGES

    # Convert to % of cohort reaching each stage
    row_totals = cross[JOURNEY_STAGES].sum(axis=1)
    for s in JOURNEY_STAGES:
        cross[s] = (cross[s] / row_totals * 100).round(1).fillna(0)

    return cross.set_index("cohort")


# ── 7. Conversion summary KPIs ────────────────────────────────────
def get_conversion_summary(df: pd.DataFrame) -> dict:
    """
    Returns dict of high-level funnel KPIs for the dashboard
    Executive Summary and Funnel page.
    """
    total = len(df)
    if total == 0:
        return {}

    def stage_count(s):
        if "journey_stage" not in df.columns:
            return 0
        return int((df["journey_stage"] == s).sum())

    awareness_n  = stage_count("Awareness")  + stage_count("Exploration")
    purchase_n   = stage_count("Purchase")   + stage_count("Post-Purchase")
    abandon_n    = int(df["abandonment_proxy"].sum()) if "abandonment_proxy" in df.columns else 0
    high_risk_n  = int((df["risk_level"] == "High").sum()) if "risk_level" in df.columns else 0
    avg_conv     = round(df["conversion_proxy"].mean() * 100, 1) if "conversion_proxy" in df.columns else 0
    avg_sat      = round(df["satisfaction_score"].mean(), 3)     if "satisfaction_score" in df.columns else 0

    proxy_conv_rate = round(purchase_n / total * 100, 1) if total > 0 else 0
    abandon_rate    = round(abandon_n  / total * 100, 1) if total > 0 else 0

    return {
        "total_customers":       total,
        "awareness_count":       awareness_n,
        "purchase_count":        purchase_n,
        "abandonment_count":     abandon_n,
        "high_risk_count":       high_risk_n,
        "proxy_conversion_rate": proxy_conv_rate,
        "abandonment_rate":      abandon_rate,
        "avg_conversion_proxy":  avg_conv,
        "avg_satisfaction":      avg_sat,
    }


# ── 8. Abandonment heatmap (segment × stage) ─────────────────────
def get_abandonment_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pivot table of abandonment_proxy rate (%) for
    each segment × journey stage combination.

    Used for the Plotly heatmap in the Funnel Analysis page.
    """
    if "abandonment_proxy" not in df.columns:
        return pd.DataFrame()
    if "segment_label" not in df.columns or "journey_stage" not in df.columns:
        return pd.DataFrame()

    heatmap = (
        df.groupby(["segment_label", "journey_stage"])["abandonment_proxy"]
          .mean()
          .mul(100)
          .round(1)
          .unstack(fill_value=0)
          .reset_index()
    )

    # Ensure all stages present
    for s in JOURNEY_STAGES:
        if s not in heatmap.columns:
            heatmap[s] = 0.0

    return heatmap[["segment_label"] + JOURNEY_STAGES]


# ── 9. Stage friction summary (for recommendation engine) ────────
def get_stage_friction_summary(df: pd.DataFrame) -> dict:
    """
    Returns a compact dict summarising the top-friction stage
    and the worst-converting transition.
    Used by recommendation_engine.py to generate stage-aware tips.
    """
    friction = get_friction_indicators(df)
    dropoffs = get_dropoff_rates(df)

    if friction.empty or dropoffs.empty:
        return {}

    top_friction_row = friction.loc[friction["friction_score"].idxmax()]
    worst_drop_row   = dropoffs.loc[dropoffs["dropoff_pct"].idxmax()]

    return {
        "highest_friction_stage":   top_friction_row["stage"],
        "highest_friction_score":   top_friction_row["friction_score"],
        "highest_friction_signal":  top_friction_row["signal"],
        "worst_dropoff_transition": (
            f"{worst_drop_row['from_stage']} → {worst_drop_row['to_stage']}"
        ),
        "worst_dropoff_pct":        worst_drop_row["dropoff_pct"],
        "overall_conversion_rate":  get_conversion_summary(df).get("proxy_conversion_rate", 0),
    }