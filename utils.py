# utils.py
import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from pathlib import Path

# ── Path helpers ──────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

for d in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    d.mkdir(exist_ok=True)

# ── Save / load helpers ───────────────────────────────────────────
def save_model(obj, name: str):
    path = MODEL_DIR / f"{name}.pkl"
    joblib.dump(obj, path)
    return path

def load_model(name: str):
    path = MODEL_DIR / f"{name}.pkl"
    if path.exists():
        return joblib.load(path)
    return None

def save_df(df: pd.DataFrame, name: str):
    path = OUTPUT_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path

def load_df(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()

# ── Streamlit UI helpers ──────────────────────────────────────────
def kpi_card(label: str, value, delta=None, color="#6C5CE7"):
    delta_html = f"<p style='color:#00b894;font-size:13px;margin:0'>{delta}</p>" if delta else ""
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{color}22,{color}11);
                border:1px solid {color}44;border-radius:12px;
                padding:16px 20px;margin:4px 0;'>
      <p style='color:#aaa;font-size:12px;margin:0 0 4px'>{label}</p>
      <p style='color:#fff;font-size:26px;font-weight:700;margin:0'>{value}</p>
      {delta_html}
    </div>""", unsafe_allow_html=True)

def section_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div style='margin:24px 0 12px'>
      <h2 style='color:#6C5CE7;font-size:22px;margin:0'>{title}</h2>
      <p style='color:#888;font-size:13px;margin:4px 0 0'>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def badge(text: str, color: str = "#6C5CE7"):
    return f"<span style='background:{color}33;color:{color};padding:2px 10px;border-radius:20px;font-size:12px;font-weight:600'>{text}</span>"

def risk_badge(score: float) -> str:
    if score >= 0.7:
        return badge("HIGH RISK", "#e17055")
    elif score >= 0.4:
        return badge("MEDIUM RISK", "#fdcb6e")
    else:
        return badge("LOW RISK", "#00b894")

# ── Feature constants (update if your dataset columns differ) ─────
NUMERIC_FEATURES = [
    "purchase_frequency", "avg_spend", "discount_sensitivity",
    "satisfaction_score", "rating", "product_diversity",
    "browsing_intensity", "repeat_purchase_score", "churn_proxy",
    "conversion_proxy", "sentiment_score", "complaint_count",
    "loyalty_score"
]

CATEGORICAL_FEATURES = [
    "gender", "payment_method", "category_preference",
    "subscription_status", "age_group"
]

JOURNEY_STAGES = [
    "Awareness", "Exploration", "Consideration",
    "Cart Intent", "Checkout Intent", "Purchase", "Post-Purchase"
]

SEGMENT_LABELS = {
    0: "Loyal Repeat Buyers",
    1: "Explorers",
    2: "Discount-Driven Shoppers",
    3: "Hesitant Buyers",
    4: "High-Value Customers",
    5: "Low-Engagement Users",
    6: "Satisfaction-Risk Users",
    7: "Abandoned-Intent Users"
}

# ── Color maps ────────────────────────────────────────────────────
SEGMENT_COLORS = {
    "Loyal Repeat Buyers":     "#6C5CE7",
    "Explorers":               "#00b894",
    "Discount-Driven Shoppers":"#fdcb6e",
    "Hesitant Buyers":         "#e17055",
    "High-Value Customers":    "#0984e3",
    "Low-Engagement Users":    "#636e72",
    "Satisfaction-Risk Users": "#d63031",
    "Abandoned-Intent Users":  "#fd79a8",
}

STAGE_COLORS = [
    "#6C5CE7","#00b894","#fdcb6e",
    "#e17055","#0984e3","#fd79a8","#a29bfe"
]