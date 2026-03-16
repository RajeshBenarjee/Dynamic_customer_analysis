# prediction.py
"""
Trains a drop-off / churn-risk prediction model.

Target label: risk_label (binary 0/1)
  - Derived in feature_engineering.py from:
    churn_proxy > 0.6 OR satisfaction < 0.3 OR abandonment_proxy == 1

Models trained (in order of complexity):
  1. Logistic Regression   — baseline
  2. Random Forest         — baseline ensemble
  3. XGBoost               — primary model (best expected performance)

Outputs per row:
  - risk_score      : float 0–1 (XGBoost probability)
  - risk_level      : "Low" | "Medium" | "High"
  - risk_model_used : name of model that produced the score
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report,
)
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from utils import save_model, load_model, save_df, MODEL_DIR

# ── Feature columns used for prediction ──────────────────────────
PRED_FEATURES = [
    "purchase_frequency",
    "avg_spend",
    "discount_sensitivity",
    "satisfaction_score",
    "rating",
    "browsing_intensity",
    "loyalty_score",
    "conversion_proxy",
    "repeat_purchase_score",
    "product_diversity",
    "recency_proxy",
    "monetary_proxy",
    "sentiment_score",        # added by nlp_engine (may be absent early on)
    "complaint_count",        # added by nlp_engine (may be absent early on)
]

TARGET_COL = "risk_label"


def _available_pred_features(df: pd.DataFrame) -> list:
    return [f for f in PRED_FEATURES if f in df.columns]


def _build_X(df: pd.DataFrame, features: list) -> pd.DataFrame:
    X = df[features].copy()
    # Encode any remaining categoricals
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    return X


# ── Build all three model pipelines ──────────────────────────────
def _build_pipelines() -> dict:
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ]

    pipelines = {
        "logistic_regression": Pipeline(base_steps + [
            ("model", LogisticRegression(
                max_iter=1000, C=1.0, class_weight="balanced", random_state=42
            ))
        ]),
        "random_forest": Pipeline(base_steps + [
            ("model", RandomForestClassifier(
                n_estimators=200, max_depth=8,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ]),
    }

    if XGB_AVAILABLE:
        pipelines["xgboost"] = Pipeline(base_steps + [
            ("model", XGBClassifier(
                n_estimators=200, max_depth=5,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, use_label_encoder=False,
                eval_metric="logloss", random_state=42, n_jobs=-1
            ))
        ])

    return pipelines


# ── Main training function ────────────────────────────────────────
def train_models(
    df: pd.DataFrame,
    test_size: float = 0.2,
    force_retrain: bool = False,
) -> dict:
    """
    Trains all pipelines and returns a results dict containing:
      {
        "metrics":        {model_name: {accuracy, precision, recall, f1, roc_auc}},
        "best_model":     name of best model by roc_auc,
        "feature_importance": {feature: importance_score},
        "roc_data":       {model_name: {"fpr": [...], "tpr": [...], "auc": ...}},
        "confusion_matrices": {model_name: np.ndarray},
        "classification_reports": {model_name: str},
        "X_test": pd.DataFrame,
        "y_test": pd.Series,
      }
    """
    features = _available_pred_features(df)

    if TARGET_COL not in df.columns or len(features) < 3:
        return {"error": "Insufficient features or missing target column."}

    X = _build_X(df, features)
    y = df[TARGET_COL].fillna(0).astype(int)

    # Balance check
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        return {"error": "Target column has only one class — cannot train."}

    # ── Train / test split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    results = {
        "metrics": {},
        "best_model": None,
        "feature_importance": {},
        "roc_data": {},
        "confusion_matrices": {},
        "classification_reports": {},
        "X_test": X_test,
        "y_test": y_test,
        "features": features,
    }

    pipelines = _build_pipelines()
    best_auc = -1.0

    for name, pipeline in pipelines.items():
        model_key = f"prediction_{name}"

        # Load cached or train fresh
        cached = None if force_retrain else load_model(model_key)
        if cached is not None:
            pipeline = cached
        else:
            pipeline.fit(X_train, y_train)
            save_model(pipeline, model_key)

        # ── Predictions ───────────────────────────────────────────
        y_pred  = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # ── Metrics ───────────────────────────────────────────────
        auc = roc_auc_score(y_test, y_proba)
        results["metrics"][name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred),  4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
            "roc_auc":   round(auc, 4),
        }
        results["confusion_matrices"][name] = confusion_matrix(y_test, y_pred)
        results["classification_reports"][name] = classification_report(
            y_test, y_pred, zero_division=0
        )

        # ── ROC curve data ────────────────────────────────────────
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        results["roc_data"][name] = {
            "fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": round(auc, 4)
        }

        # ── Feature importance ────────────────────────────────────
        inner_model = pipeline.named_steps["model"]
        if hasattr(inner_model, "feature_importances_"):
            results["feature_importance"][name] = dict(
                zip(features, inner_model.feature_importances_.tolist())
            )
        elif hasattr(inner_model, "coef_"):
            results["feature_importance"][name] = dict(
                zip(features, np.abs(inner_model.coef_[0]).tolist())
            )

        if auc > best_auc:
            best_auc = auc
            results["best_model"] = name

    return results


# ── Score new / all rows ──────────────────────────────────────────
def score_dataframe(
    df: pd.DataFrame,
    model_name: str = "xgboost",
) -> pd.DataFrame:
    """
    Adds risk_score (0-1) and risk_level ("Low"/"Medium"/"High")
    to every row in df using the saved prediction pipeline.

    Falls back to random_forest then logistic_regression if XGBoost
    model not found.
    """
    df = df.copy()

    for try_name in [model_name, "xgboost", "random_forest", "logistic_regression"]:
        pipeline = load_model(f"prediction_{try_name}")
        if pipeline is not None:
            model_name = try_name
            break
    else:
        # No model found — return neutral scores
        df["risk_score"]      = 0.5
        df["risk_level"]      = "Medium"
        df["risk_model_used"] = "none"
        return df

    features = _available_pred_features(df)
    X = _build_X(df, features)

    # Align columns to what the model was trained on
    trained_features = pipeline.named_steps["imputer"].feature_names_in_ \
        if hasattr(pipeline.named_steps["imputer"], "feature_names_in_") \
        else features

    for col in trained_features:
        if col not in X.columns:
            X[col] = 0.0
    X = X[trained_features]

    proba = pipeline.predict_proba(X)[:, 1]
    df["risk_score"]      = proba.round(4)
    df["risk_level"]      = pd.cut(
        df["risk_score"],
        bins=[0, 0.4, 0.7, 1.0],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)
    df["risk_model_used"] = model_name

    return df


# ── SHAP explanation (lazy import to avoid slow startup) ──────────
def get_shap_values(
    df: pd.DataFrame,
    model_name: str = "xgboost",
    max_rows: int = 500,
) -> dict:
    """
    Returns SHAP values and base value for up to max_rows rows.
    Only works with tree-based models (XGBoost / RandomForest).
    Returns None if SHAP unavailable or model incompatible.
    """
    try:
        import shap
    except ImportError:
        return {"error": "SHAP not installed. Run: pip install shap"}

    pipeline = load_model(f"prediction_{model_name}")
    if pipeline is None:
        return {"error": f"Model '{model_name}' not found. Train first."}

    features = _available_pred_features(df)
    X = _build_X(df, features).head(max_rows)

    # Run data through imputer + scaler steps
    X_transformed = pipeline[:-1].transform(X)   # all steps except final model

    inner_model = pipeline.named_steps["model"]

    try:
        explainer    = shap.TreeExplainer(inner_model)
        shap_values  = explainer.shap_values(X_transformed)

        # For binary classifiers, shap_values is a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return {
            "shap_values":  shap_values,          # shape (n, n_features)
            "base_value":   explainer.expected_value
                            if not isinstance(explainer.expected_value, list)
                            else explainer.expected_value[1],
            "feature_names": features,
            "X_sample":     X,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Quick summary stats for dashboard ────────────────────────────
def get_risk_summary(df: pd.DataFrame) -> dict:
    """Returns counts and percentages for each risk level."""
    if "risk_level" not in df.columns:
        return {}
    vc = df["risk_level"].value_counts()
    total = len(df)
    return {
        level: {
            "count": int(vc.get(level, 0)),
            "pct":   round(vc.get(level, 0) / total * 100, 1),
        }
        for level in ["High", "Medium", "Low"]
    }


def get_high_risk_customers(df: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Returns top_n highest-risk customers with key profile columns."""
    if "risk_score" not in df.columns:
        return pd.DataFrame()

    display_cols = [c for c in [
        "customer_id", "risk_score", "risk_level",
        "segment_label", "satisfaction_score",
        "churn_proxy", "conversion_proxy",
        "category_preference", "avg_spend",
        "payment_method", "age_group",
    ] if c in df.columns]

    return (
        df[display_cols]
        .sort_values("risk_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )