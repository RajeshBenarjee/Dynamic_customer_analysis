# dashboard_app.py
"""
Main Streamlit dashboard for the Customer Journey Analysis System.
Run with:  streamlit run dashboard_app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
st.set_page_config(
    page_title="Customer Journey Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from data_ingestion      import load_all_datasets, get_dataset_summary
from schema_mapper       import merge_all
from preprocessing       import preprocess
from feature_engineering import engineer_features, get_journey_funnel_counts
from segmentation        import (run_segmentation, get_segment_profiles,
                                  evaluate_clustering, get_segment_rule)
from prediction          import train_models, score_dataframe, get_risk_summary, get_high_risk_customers
from nlp_engine          import (run_nlp_pipeline, get_embeddings,
                                  get_top_keywords, get_complaint_themes,
                                  get_sentiment_summary)
from faiss_engine        import (build_index, find_similar_complaints,
                                  find_similar_customers, index_status)
from funnel_analysis     import (get_funnel_counts, get_dropoff_rates,
                                  get_friction_indicators, get_category_funnel,
                                  get_sankey_data, get_cohort_retention,
                                  get_conversion_summary, get_abandonment_heatmap)
from explainability      import (build_customer_card,
                                  get_global_explanation_summary,
                                  get_global_feature_importance)
from recommendation_engine import generate_recommendations, get_recommendations_for_customer
from utils               import kpi_card, section_header, badge, risk_badge, SEGMENT_COLORS, STAGE_COLORS


# ╔══════════════════════════════════════════════════════════════╗
# ║  Custom CSS                                                  ║
# ╚══════════════════════════════════════════════════════════════╝
st.markdown("""
<style>
  [data-testid="stSidebar"]          { background: #0d0d1a; }
  [data-testid="stSidebar"] *        { color: #e0e0e0 !important; }
  .main .block-container              { padding-top: 1.5rem; }
  div[data-testid="metric-container"] { background:#1a1a2e; border-radius:12px;
                                         padding:12px; border:1px solid #2d2d50; }
  .stTabs [data-baseweb="tab"]        { color:#aaa; font-size:14px; }
  .stTabs [aria-selected="true"]      { color:#6C5CE7 !important;
                                         border-bottom:2px solid #6C5CE7; }
  .rec-card { background:#12122a; border:1px solid #2d2d50; border-radius:12px;
               padding:16px 20px; margin-bottom:12px; }
  .rec-critical { border-left:4px solid #e17055 !important; }
  .rec-high     { border-left:4px solid #fdcb6e !important; }
  .rec-medium   { border-left:4px solid #6C5CE7 !important; }
  .rec-low      { border-left:4px solid #636e72 !important; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Pipeline (cached so it runs once per session)               ║
# ╚══════════════════════════════════════════════════════════════╝
@st.cache_data(show_spinner=False)
def run_full_pipeline():
    with st.spinner("Loading and processing datasets…"):
        dfs     = load_all_datasets()
        merged  = merge_all(dfs)
        if merged.empty:
            return None, None, None

        cleaned  = preprocess(merged)
        featured = engineer_features(cleaned)

    with st.spinner("Running segmentation…"):
        segmented = run_segmentation(featured, n_clusters=5, method="kmeans")

    with st.spinner("Training prediction models…"):
        train_results = train_models(segmented)
        scored        = score_dataframe(segmented)

    with st.spinner("Running NLP pipeline…"):
        nlp_df = run_nlp_pipeline(scored)

    with st.spinner("Building FAISS index…"):
        texts      = nlp_df["clean_text"].fillna("").tolist()
        embeddings = get_embeddings(texts, cache_key="sbert_embeddings")
        build_index(embeddings, nlp_df, force_rebuild=False)

    return nlp_df, train_results, embeddings


@st.cache_data(show_spinner=False)
def get_recommendations(_df):
    return generate_recommendations(_df)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Sidebar                                                     ║
# ╚══════════════════════════════════════════════════════════════╝
def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("## Filters")

    filters = {}

    if "age_group" in df.columns:
        opts = ["All"] + sorted(df["age_group"].dropna().unique().tolist())
        filters["age_group"] = st.sidebar.selectbox("Age group", opts)

    if "gender" in df.columns:
        opts = ["All"] + sorted(df["gender"].dropna().unique().tolist())
        filters["gender"] = st.sidebar.selectbox("Gender", opts)

    if "category_preference" in df.columns:
        opts = ["All"] + sorted(df["category_preference"].dropna().unique().tolist())
        filters["category"] = st.sidebar.selectbox("Category", opts)

    if "segment_label" in df.columns:
        opts = ["All"] + sorted(df["segment_label"].dropna().unique().tolist())
        filters["segment"] = st.sidebar.selectbox("Segment", opts)

    if "risk_level" in df.columns:
        opts = ["All", "High", "Medium", "Low"]
        filters["risk_level"] = st.sidebar.selectbox("Risk level", opts)

    if "sentiment_label" in df.columns:
        opts = ["All", "Positive", "Neutral", "Negative"]
        filters["sentiment"] = st.sidebar.selectbox("Sentiment", opts)

    if "payment_method" in df.columns:
        opts = ["All"] + sorted(df["payment_method"].dropna().unique().tolist())
        filters["payment"] = st.sidebar.selectbox("Payment method", opts)

    st.sidebar.markdown("---")
    st.sidebar.caption("⚠️ All funnel stages are **proxy / inferred** from behavioural features — not true clickstream data.")

    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    col_map = {
        "age_group": "age_group",
        "gender":    "gender",
        "category":  "category_preference",
        "segment":   "segment_label",
        "risk_level":"risk_level",
        "sentiment": "sentiment_label",
        "payment":   "payment_method",
    }
    for fkey, col in col_map.items():
        val = filters.get(fkey, "All")
        if val != "All" and col in df.columns:
            mask &= df[col] == val
    return df[mask].reset_index(drop=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  Page 1 — Executive Summary                                  ║
# ╚══════════════════════════════════════════════════════════════╝
def page_executive_summary(df: pd.DataFrame, train_results: dict):
    section_header("Executive Summary", "High-level KPIs across the entire customer base")

    kpis = get_conversion_summary(df)
    risk = get_risk_summary(df)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total customers",      f"{kpis.get('total_customers', 0):,}", color="#6C5CE7")
    with c2:
        kpi_card("Proxy conversion rate", f"{kpis.get('proxy_conversion_rate', 0)}%", color="#00b894")
    with c3:
        kpi_card("Abandonment rate",      f"{kpis.get('abandonment_rate', 0)}%", color="#e17055")
    with c4:
        kpi_card("High-risk customers",   f"{risk.get('High', {}).get('count', 0):,}", color="#d63031")

    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi_card("Avg satisfaction",  f"{kpis.get('avg_satisfaction', 0):.2f}/1.0", color="#0984e3")
    with c6:
        kpi_card("Purchase count",    f"{kpis.get('purchase_count', 0):,}",          color="#00b894")
    with c7:
        kpi_card("Avg conversion proxy", f"{kpis.get('avg_conversion_proxy', 0)}%", color="#6C5CE7")
    with c8:
        kpi_card("Medium-risk count", f"{risk.get('Medium', {}).get('count', 0):,}", color="#fdcb6e")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1, 1])

    # Segment pie
    with col_left:
        st.markdown("#### Segment distribution")
        if "segment_label" in df.columns:
            seg_counts = df["segment_label"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            fig = px.pie(
                seg_counts, names="Segment", values="Count",
                color="Segment",
                color_discrete_map=SEGMENT_COLORS,
                hole=0.45,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", showlegend=True,
                legend=dict(font=dict(size=11)),
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Risk donut
    with col_right:
        st.markdown("#### Risk level distribution")
        risk_labels = ["High", "Medium", "Low"]
        risk_vals   = [risk.get(l, {}).get("count", 0) for l in risk_labels]
        risk_colors = ["#e17055", "#fdcb6e", "#00b894"]
        fig2 = go.Figure(go.Pie(
            labels=risk_labels, values=risk_vals,
            marker_colors=risk_colors, hole=0.45,
            textinfo="label+percent",
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
            margin=dict(t=20, b=20), showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Dataset summary
    st.markdown("#### Dataset loading status")
    dfs_raw = load_all_datasets()
    st.dataframe(get_dataset_summary(dfs_raw), use_container_width=True)

    # Model metrics
    if train_results and "metrics" in train_results:
        st.markdown("#### Model performance")
        metrics_df = pd.DataFrame(train_results["metrics"]).T.reset_index()
        metrics_df.columns = ["Model"] + list(metrics_df.columns[1:])
        st.dataframe(
            metrics_df.style.highlight_max(
                subset=["roc_auc", "f1"], color="#6C5CE7"
            ),
            use_container_width=True,
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║  Page 2 — Journey / Funnel Analysis                          ║
# ╚══════════════════════════════════════════════════════════════╝
def page_funnel(df: pd.DataFrame):
    section_header(
        "Journey & Funnel Analysis",
        "PROXY stages — inferred from browsing_intensity, conversion_proxy, "
        "discount_sensitivity, satisfaction_score. Not true clickstream.",
    )

    col_left, col_right = st.columns([1, 1])

    # Funnel bar chart
    with col_left:
        st.markdown("#### Inferred funnel stage counts")
        counts = get_funnel_counts(df)
        fig = go.Figure(go.Bar(
            x=counts["count"],
            y=counts["stage"],
            orientation="h",
            marker_color=STAGE_COLORS[:len(counts)],
            text=counts["count"],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", margin=dict(t=10, b=10, l=150),
            xaxis_title="Customer count", yaxis_title="",
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Drop-off table
    with col_right:
        st.markdown("#### Stage-to-stage drop-off (proxy)")
        drops = get_dropoff_rates(df)
        st.dataframe(
            drops[["from_stage", "to_stage", "dropoff_pct", "conversion_rate"]]
              .style.background_gradient(subset=["dropoff_pct"], cmap="Reds"),
            use_container_width=True, height=340,
        )

    # Sankey
    st.markdown("#### Customer flow Sankey (proxy transitions)")
    sankey = get_sankey_data(df)
    if sankey["value"]:
        fig_s = go.Figure(go.Sankey(
            node=dict(
                pad=15, thickness=20,
                label=sankey["labels"],
                color=sankey["node_colors"],
            ),
            link=dict(
                source=sankey["source"],
                target=sankey["target"],
                value=sankey["value"],
                color=sankey["link_colors"],
            ),
        ))
        fig_s.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
            margin=dict(t=10, b=10), height=300,
        )
        st.plotly_chart(fig_s, use_container_width=True)

    # Friction heatmap
    st.markdown("#### Friction indicators per stage")
    friction = get_friction_indicators(df)
    fig_f = px.bar(
        friction, x="stage", y="friction_score",
        color="friction_score",
        color_continuous_scale="Reds",
        text="signal",
        title="",
    )
    fig_f.update_traces(textposition="outside")
    fig_f.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", showlegend=False,
        margin=dict(t=10, b=80), height=320,
        xaxis_title="", yaxis_title="Friction score (0–1)",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_f, use_container_width=True)

    # Abandonment heatmap
    st.markdown("#### Abandonment rate heatmap (segment × stage, proxy %)")
    heatmap_df = get_abandonment_heatmap(df)
    if not heatmap_df.empty:
        z    = heatmap_df.drop(columns=["segment_label"]).values
        segs = heatmap_df["segment_label"].tolist()
        from utils import JOURNEY_STAGES
        fig_h = go.Figure(go.Heatmap(
            z=z, x=JOURNEY_STAGES, y=segs,
            colorscale="Purples", text=z,
            texttemplate="%{text:.1f}%",
        ))
        fig_h.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e0e0",
            margin=dict(t=10, b=10), height=340,
        )
        st.plotly_chart(fig_h, use_container_width=True)

    # Category funnel
    st.markdown("#### Category-level funnel breakdown")
    cat_funnel = get_category_funnel(df)
    if not cat_funnel.empty:
        st.dataframe(
            cat_funnel.head(15)
              .style.background_gradient(subset=["purchase_rate"], cmap="Greens"),
            use_container_width=True,
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║  Page 3 — Segmentation                                       ║
# ╚══════════════════════════════════════════════════════════════╝
def page_segmentation(df: pd.DataFrame):
    section_header("Customer Segmentation", "K-Means clustering on proxy behavioural features")

    eval_res = evaluate_clustering(df)
    sil = eval_res.get("silhouette")
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Segments found", df["segment_label"].nunique() if "segment_label" in df.columns else 0, color="#6C5CE7")
    with c2: kpi_card("Silhouette score", f"{sil:.3f}" if sil else "N/A", color="#00b894")
    with c3: kpi_card("Total customers", f"{len(df):,}", color="#0984e3")

    col_left, col_right = st.columns([1.2, 0.8])

    # PCA scatter
    with col_left:
        st.markdown("#### Segment scatter (PCA 2-D)")
        if "pca_x" in df.columns and "segment_label" in df.columns:
            sample = df.sample(min(2000, len(df)), random_state=42)
            fig = px.scatter(
                sample, x="pca_x", y="pca_y",
                color="segment_label",
                color_discrete_map=SEGMENT_COLORS,
                opacity=0.7, size_max=6,
                hover_data=[c for c in ["customer_id","risk_level","satisfaction_score"]
                            if c in sample.columns],
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", legend_title="Segment",
                margin=dict(t=10, b=10), height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Segment size bar
    with col_right:
        st.markdown("#### Segment sizes")
        if "segment_label" in df.columns:
            vc = df["segment_label"].value_counts().reset_index()
            vc.columns = ["Segment", "Count"]
            fig2 = px.bar(
                vc, x="Count", y="Segment", orientation="h",
                color="Segment", color_discrete_map=SEGMENT_COLORS,
                text="Count",
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", showlegend=False,
                margin=dict(t=10, b=10, l=180), height=420,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Segment profiles table
    st.markdown("#### Segment feature profiles")
    profiles = get_segment_profiles(df)
    if not profiles.empty:
        num_cols = profiles.select_dtypes(include="number").columns.tolist()
        st.dataframe(
            profiles.style.background_gradient(subset=num_cols, cmap="Purples"),
            use_container_width=True,
        )

    # Segment rules
    st.markdown("#### Explainable segment rules")
    if "segment_label" in df.columns:
        for label in df["segment_label"].unique():
            rule = get_segment_rule(label)
            color = SEGMENT_COLORS.get(label, "#6C5CE7")
            st.markdown(
                f"<div style='border-left:3px solid {color};"
                f"padding:10px 16px;margin:6px 0;background:#12122a;"
                f"border-radius:0 8px 8px 0'>"
                f"<b style='color:{color}'>{label}</b><br>"
                f"<span style='color:#aaa;font-size:13px'>{rule.get('rule','')}</span><br>"
                f"<span style='color:#888;font-size:12px'>{rule.get('business_insight','')}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ╔══════════════════════════════════════════════════════════════╗
# ║  Page 4 — Drop-off Prediction                                ║
# ╚══════════════════════════════════════════════════════════════╝
def page_prediction(df: pd.DataFrame, train_results: dict):
    section_header("Drop-off & Risk Prediction", "XGBoost · Random Forest · Logistic Regression")

    risk = get_risk_summary(df)
    c1, c2, c3 = st.columns(3)
    for col, level, color in zip(
        [c1, c2, c3],
        ["High", "Medium", "Low"],
        ["#e17055", "#fdcb6e", "#00b894"],
    ):
        with col:
            kpi_card(f"{level} risk", f"{risk.get(level,{}).get('count',0):,} "
                     f"({risk.get(level,{}).get('pct',0)}%)", color=color)

    # Risk score histogram
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("#### Risk score distribution")
        if "risk_score" in df.columns:
            fig = px.histogram(
                df, x="risk_score", nbins=40,
                color_discrete_sequence=["#6C5CE7"],
            )
            fig.add_vline(x=0.4, line_dash="dash", line_color="#fdcb6e", annotation_text="Med threshold")
            fig.add_vline(x=0.7, line_dash="dash", line_color="#e17055", annotation_text="High threshold")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", margin=dict(t=10, b=10), height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    with col_right:
        st.markdown("#### Global feature importance")
        imp = get_global_feature_importance(top_n=10)
        if not imp.empty:
            fig2 = px.bar(
                imp, x="normalised_importance", y="feature",
                orientation="h", color_discrete_sequence=["#6C5CE7"],
                text="normalised_importance",
            )
            fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", margin=dict(t=10, b=10, l=160), height=300,
                xaxis_title="Importance (%)", yaxis_title="",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ROC curves
    if train_results and "roc_data" in train_results:
        st.markdown("#### ROC curves")
        fig_roc = go.Figure()
        colors = {"xgboost": "#6C5CE7", "random_forest": "#00b894",
                  "logistic_regression": "#fdcb6e"}
        for name, roc in train_results["roc_data"].items():
            fig_roc.add_trace(go.Scatter(
                x=roc["fpr"], y=roc["tpr"],
                name=f"{name} (AUC={roc['auc']})",
                line=dict(color=colors.get(name, "#aaa"), width=2),
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], name="Random",
            line=dict(color="#555", dash="dash"), showlegend=False,
        ))
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", xaxis_title="FPR", yaxis_title="TPR",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10), height=340,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # High-risk table
    st.markdown("#### Top high-risk customers")
    hr = get_high_risk_customers(df, top_n=20)
    if not hr.empty:
        st.dataframe(
            hr.style.background_gradient(subset=["risk_score"], cmap="Reds"),
            use_container_width=True,
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║  Page 5 — NLP Insights                                       ║
# ╚══════════════════════════════════════════════════════════════╝
def page_nlp(df: pd.DataFrame):
    section_header("NLP Insights", "Sentiment · Keywords · Complaint themes · Semantic search")

    sent = get_sentiment_summary(df)
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Positive sentiment", f"{sent.get('Positive',{}).get('pct',0)}%", color="#00b894")
    with c2: kpi_card("Neutral sentiment",  f"{sent.get('Neutral', {}).get('pct',0)}%", color="#0984e3")
    with c3: kpi_card("Negative sentiment", f"{sent.get('Negative',{}).get('pct',0)}%", color="#e17055")

    col_left, col_right = st.columns(2)

    # Complaint themes
    with col_left:
        st.markdown("#### Complaint signal themes")
        themes = get_complaint_themes(df)
        if not themes.empty:
            fig = px.bar(
                themes, x="count", y="theme", orientation="h",
                color="pct", color_continuous_scale="Reds",
                text="pct",
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=140), height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Top keywords
    with col_right:
        st.markdown("#### Top TF-IDF keywords")
        kw_df = get_top_keywords(df, n=15)
        if not kw_df.empty:
            fig2 = px.bar(
                kw_df, x="tfidf_score", y="keyword", orientation="h",
                color_discrete_sequence=["#6C5CE7"], text="tfidf_score",
            )
            fig2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", margin=dict(t=10, b=10, l=140), height=300,
                xaxis_title="TF-IDF score", yaxis_title="",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Sentiment by segment
    if "segment_label" in df.columns and "sentiment_label" in df.columns:
        st.markdown("#### Sentiment distribution by segment")
        cross = (df.groupby(["segment_label", "sentiment_label"])
                   .size()
                   .reset_index(name="count"))
        fig3 = px.bar(
            cross, x="segment_label", y="count",
            color="sentiment_label",
            color_discrete_map={"Positive":"#00b894","Neutral":"#0984e3","Negative":"#e17055"},
            barmode="stack",
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", margin=dict(t=10, b=80),
            xaxis_tickangle=-30, height=340,
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Semantic search panel
    st.markdown("#### Semantic similarity search (FAISS)")
    idx_status = index_status()
    if idx_status["ready"]:
        query = st.text_input(
            "Search for similar complaints / reviews",
            placeholder="e.g. product quality was terrible, never arrived",
        )
        filt_sent = st.selectbox("Filter by sentiment", ["Any", "Positive", "Neutral", "Negative"])
        if query:
            results = find_similar_complaints(
                query, k=6,
                filter_sentiment=None if filt_sent == "Any" else filt_sent,
            )
            if not results.empty and "error" not in results.columns:
                st.dataframe(results, use_container_width=True)
            else:
                st.info("No results found. Try a different query.")
    else:
        st.warning("FAISS index not built yet. Run the pipeline to enable semantic search.")


# ╔══════════════════════════════════════════════════════════════╗
# ║  Page 6 — Recommendations                                    ║
# ╚══════════════════════════════════════════════════════════════╝
def page_recommendations(df: pd.DataFrame):
    section_header("Recommendation Engine", "Evidence-based, segment-aware, stage-linked actions")

    recs = generate_recommendations(df)
    if recs.empty:
        st.info("No recommendations generated — ensure the full pipeline has run.")
        return

    priority_colors = {
        "Critical": "#e17055", "High": "#fdcb6e",
        "Medium":   "#6C5CE7", "Low":  "#636e72",
    }
    action_icons = {
        "retention":  "shield",
        "conversion": "trending_up",
        "engagement": "explore",
        "pricing":    "sell",
        "trust":      "verified",
        "loyalty":    "stars",
    }

    # Summary KPIs
    c1, c2, c3, c4 = st.columns(4)
    for col, pri, clr in zip(
        [c1, c2, c3, c4],
        ["Critical", "High", "Medium", "Low"],
        ["#e17055", "#fdcb6e", "#6C5CE7", "#636e72"],
    ):
        n = len(recs[recs["priority"] == pri])
        with col: kpi_card(f"{pri} priority", str(n), color=clr)

    st.markdown("<br>", unsafe_allow_html=True)

    # Filter bar
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        pri_filter = st.selectbox("Filter by priority",
            ["All"] + ["Critical", "High", "Medium", "Low"])
    with col_f2:
        act_filter = st.selectbox("Filter by action type",
            ["All"] + sorted(recs["action_type"].unique().tolist()))

    filtered = recs.copy()
    if pri_filter != "All":
        filtered = filtered[filtered["priority"] == pri_filter]
    if act_filter != "All":
        filtered = filtered[filtered["action_type"] == act_filter]

    # Render cards
    for _, rec in filtered.iterrows():
        pri   = rec["priority"]
        color = priority_colors.get(pri, "#888")
        ev    = rec.get("evidence", {})

        ev_html = " &nbsp;·&nbsp; ".join(
            f"<span style='color:#aaa'>{k.replace('_',' ')}: "
            f"<b style='color:#e0e0e0'>{v}</b></span>"
            for k, v in ev.items() if not isinstance(v, dict)
        )

        st.markdown(
            f"<div class='rec-card rec-{pri.lower()}'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='color:{color};font-weight:700;font-size:15px'>{rec['title']}</span>"
            f"<span style='background:{color}22;color:{color};padding:3px 10px;"
            f"border-radius:20px;font-size:12px;font-weight:600'>{pri}</span></div>"
            f"<div style='margin:6px 0 4px;color:#888;font-size:12px'>"
            f"Segment: <b style='color:#aaa'>{rec['segment']}</b> &nbsp;·&nbsp; "
            f"Stage: <b style='color:#aaa'>{rec['stage']}</b> &nbsp;·&nbsp; "
            f"Type: <b style='color:#aaa'>{rec['action_type']}</b></div>"
            f"<p style='color:#ccc;font-size:13px;margin:6px 0'>{rec['rationale']}</p>"
            f"<p style='color:#6C5CE7;font-size:12px;margin:4px 0'>"
            f"Expected impact: {rec['expected_impact']}</p>"
            f"<div style='margin-top:6px;font-size:12px'>{ev_html}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║  Page 7 — Customer Drill-down                                ║
# ╚══════════════════════════════════════════════════════════════╝
def page_drilldown(df: pd.DataFrame, embeddings: np.ndarray):
    section_header("Customer Drill-down", "Individual customer profile · risk · NLP · similar cases")

    if "customer_id" not in df.columns:
        st.warning("No customer_id column found in the dataset.")
        return

    cid_list = df["customer_id"].astype(str).tolist()
    selected = st.selectbox("Select a customer", cid_list[:500])

    row = df[df["customer_id"].astype(str) == selected]
    if row.empty:
        st.warning("Customer not found.")
        return

    row = row.iloc[0]
    card = build_customer_card(row, df)

    col_l, col_r = st.columns([1, 1])

    # Profile
    with col_l:
        st.markdown("#### Profile")
        profile = card.get("profile", {})
        profile_df = pd.DataFrame(
            [{"Field": k, "Value": v} for k, v in profile.items()]
        )
        st.dataframe(profile_df, use_container_width=True, hide_index=True)

    # Journey + risk
    with col_r:
        j = card.get("journey", {})
        r = card.get("risk",    {})
        st.markdown("#### Journey & risk")
        st.markdown(
            f"**Journey stage:** {j.get('stage','—')}  \n"
            f"_{j.get('description','')}_  \n\n"
            f"**Abandonment flag:** {'Yes' if j.get('abandonment_flag') else 'No'}  \n"
            f"**Risk score:** `{r.get('score', 0):.3f}` — "
            f"{risk_badge(r.get('score', 0))}  \n"
            f"**Churn proxy:** `{r.get('churn_proxy', 0):.3f}`",
            unsafe_allow_html=True,
        )

        nlp = card.get("nlp", {})
        st.markdown("#### NLP signals")
        st.markdown(
            f"**Sentiment:** `{nlp.get('sentiment_label','—')}` "
            f"(score `{nlp.get('sentiment_score',0):.3f}`)  \n"
            f"**Intent:** `{nlp.get('intent_label','—')}`  \n"
            f"**Complaint signals:** `{nlp.get('complaint_count',0)}`  \n"
            f"**Frustration:** {'Yes' if nlp.get('frustration') else 'No'} &nbsp; "
            f"**Pricing concern:** {'Yes' if nlp.get('pricing_concern') else 'No'} &nbsp; "
            f"**Trust issue:** {'Yes' if nlp.get('trust_issue') else 'No'}",
            unsafe_allow_html=True,
        )
        snippet = nlp.get("review_snippet", "")
        if snippet and snippet != "No review text available.":
            st.markdown(f"> _{snippet}_")

    # SHAP waterfall
    st.markdown("#### Feature contribution (SHAP / heuristic)")
    wf = card.get("risk", {}).get("waterfall", {})
    if wf and not (wf.get("error") or "").startswith("No trained"):
        features   = wf.get("features", [])
        shap_vals  = wf.get("shap_values", [])
        feat_vals  = wf.get("feature_vals", [])
        if features and shap_vals:
            wf_df = pd.DataFrame({
                "Feature":       features,
                "SHAP value":    shap_vals,
                "Feature value": feat_vals,
            }).sort_values("SHAP value", key=abs, ascending=False).head(10)

            fig_wf = go.Figure(go.Bar(
                x=wf_df["SHAP value"],
                y=wf_df["Feature"],
                orientation="h",
                marker_color=["#e17055" if v > 0 else "#00b894"
                              for v in wf_df["SHAP value"]],
                text=wf_df["Feature value"].round(3),
                textposition="outside",
            ))
            fig_wf.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0", xaxis_title="SHAP contribution",
                margin=dict(t=10, b=10, l=160), height=320,
            )
            st.plotly_chart(fig_wf, use_container_width=True)

    # Similar customers via FAISS
    if index_status()["ready"] and embeddings is not None:
        st.markdown("#### Similar customers (FAISS semantic search)")
        similar = find_similar_customers(
            selected, df, embeddings, k=5
        )
        if not similar.empty and "error" not in similar.columns:
            st.dataframe(similar, use_container_width=True)

    # Customer-specific recommendations
    st.markdown("#### Recommended actions for this customer")
    cust_recs = get_recommendations_for_customer(row, df)
    for rec in cust_recs[:3]:
        color = {"Critical":"#e17055","High":"#fdcb6e","Medium":"#6C5CE7","Low":"#636e72"}.get(rec["priority"],"#888")
        st.markdown(
            f"<div style='border-left:3px solid {color};padding:8px 14px;"
            f"background:#12122a;border-radius:0 8px 8px 0;margin:4px 0'>"
            f"<b style='color:{color}'>{rec['priority']}</b> — {rec['title']}<br>"
            f"<span style='color:#aaa;font-size:12px'>{rec['rationale'][:120]}…</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║  App entry point                                             ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    st.title("Customer Journey Behavior Analysis")
    st.caption("AI-powered · Proxy-based funnel · Explainable ML · Semantic NLP")

    # Run pipeline
    df, train_results, embeddings = run_full_pipeline()

    if df is None or df.empty:
        st.error(
            "No datasets found in the `data/` folder.  \n"
            "Please download the datasets from Kaggle and place the CSV files in `data/`.  \n"
            "Expected filenames:  \n"
            "- `amazon_customer_behavior.csv`  \n"
            "- `amazon_consumer_behaviour.csv`  \n"
            "- `customer_shopping_trends.csv`  \n"
            "- `ecommerce_consumer_behavior.csv`"
        )
        return

    # Sidebar filters
    filters = render_sidebar(df)
    filtered_df = apply_filters(df, filters)

    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} customers after filters.")

    # Navigation tabs
    tabs = st.tabs([
        "Executive Summary",
        "Journey & Funnel",
        "Segmentation",
        "Drop-off Prediction",
        "NLP Insights",
        "Recommendations",
        "Customer Drill-down",
    ])

    with tabs[0]: page_executive_summary(filtered_df, train_results)
    with tabs[1]: page_funnel(filtered_df)
    with tabs[2]: page_segmentation(filtered_df)
    with tabs[3]: page_prediction(filtered_df, train_results)
    with tabs[4]: page_nlp(filtered_df)
    with tabs[5]: page_recommendations(filtered_df)
    with tabs[6]: page_drilldown(filtered_df, embeddings)


if __name__ == "__main__":
    main()