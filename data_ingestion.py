# data_ingestion.py
"""
Loads all four datasets from the /data folder.
Handles missing files gracefully.
Returns a dict of raw DataFrames.
"""
import pandas as pd
from pathlib import Path
from utils import DATA_DIR
import streamlit as st

DATASET_FILES = {
    "amazon_behavior":     "amazon_customer_behavior.csv",
    "amazon_consumer":     "amazon_consumer_behaviour.csv",
    "shopping_trends":     "customer_shopping_trends.csv",
    "ecommerce_behavior":  "ecommerce_consumer_behavior.csv",
}

@st.cache_data(show_spinner="Loading datasets…")
def load_all_datasets() -> dict:
    dfs = {}
    for key, filename in DATASET_FILES.items():
        path = DATA_DIR / filename
        if path.exists():
            try:
                df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
                df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
                dfs[key] = df
                print(f"[✓] Loaded {key}: {df.shape}")
            except Exception as e:
                print(f"[!] Failed to load {filename}: {e}")
                dfs[key] = pd.DataFrame()
        else:
            print(f"[!] Not found: {filename}")
            dfs[key] = pd.DataFrame()
    return dfs

def get_dataset_summary(dfs: dict) -> pd.DataFrame:
    rows = []
    for name, df in dfs.items():
        rows.append({
            "Dataset": name,
            "Rows": len(df),
            "Columns": df.shape[1] if not df.empty else 0,
            "Missing %": round(df.isnull().mean().mean() * 100, 1) if not df.empty else 0,
            "Status": "Loaded" if not df.empty else "Missing"
        })
    return pd.DataFrame(rows)