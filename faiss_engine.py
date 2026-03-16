# faiss_engine.py
"""
FAISS-based semantic similarity search engine.

Builds a vector index from BERT embeddings produced by nlp_engine.py.
Enables three search modes used in the dashboard:

  1. find_similar_complaints(query_text, k)
       → returns k rows most semantically similar to a free-text query

  2. find_similar_customers(customer_id, k)
       → returns k customers most similar to a given customer's review text

  3. find_abandonment_risk_peers(customer_id, k)
       → returns k customers with similar behaviour + high abandonment proxy

All results include the matched row's full profile from the master dataframe
so the dashboard can display them directly.

NOTE: FAISS operates on float32 L2-normalised vectors.
      Cosine similarity == inner product for normalised vectors.
      We use IndexFlatIP (inner product) for cosine similarity search.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from utils import save_model, load_model, MODEL_DIR

# ── Graceful FAISS import ─────────────────────────────────────────
try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    print("[FAISS] faiss-cpu not installed. Falling back to numpy cosine search.")


# ── Index management ──────────────────────────────────────────────
INDEX_PATH = MODEL_DIR / "faiss_index.bin"
META_KEY   = "faiss_meta"       # stores row IDs + short text snippets


def build_index(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    force_rebuild: bool = False,
) -> bool:
    """
    Builds and saves the FAISS index from embeddings.

    Args:
        embeddings : float32 array of shape (n, d) — L2 normalised
        df         : master dataframe (same row order as embeddings)
        force_rebuild: if True, ignores cached index

    Returns True on success.
    """
    if not force_rebuild and INDEX_PATH.exists():
        print("[FAISS] Index already exists. Use force_rebuild=True to overwrite.")
        return True

    n, d = embeddings.shape
    emb  = embeddings.astype(np.float32)

    if FAISS_OK:
        # IndexFlatIP = exact inner product (= cosine for normalised vectors)
        index = faiss.IndexFlatIP(d)
        index.add(emb)
        faiss.write_index(index, str(INDEX_PATH))
        print(f"[FAISS] Index built: {n} vectors, dim={d}")
    else:
        # Fallback: store the raw matrix (numpy cosine search at query time)
        save_model(emb, "faiss_numpy_fallback")
        print(f"[FAISS] Numpy fallback stored: {n} vectors, dim={d}")

    # Save row metadata for result lookup
    meta_cols = [c for c in [
        "customer_id", "review_text", "clean_text",
        "sentiment_label", "segment_label", "risk_level",
        "intent_label", "topic_label",
        "satisfaction_score", "churn_proxy",
        "category_preference", "abandonment_proxy",
    ] if c in df.columns]

    meta = df[meta_cols].reset_index(drop=True)
    save_model(meta, META_KEY)
    save_model(list(range(n)), "faiss_row_ids")
    return True


def _load_index():
    """Returns (index_or_matrix, meta_df) or (None, None) if not built."""
    meta = load_model(META_KEY)
    if meta is None:
        return None, None

    if FAISS_OK and INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        return index, meta
    else:
        fallback = load_model("faiss_numpy_fallback")
        if fallback is not None:
            return fallback, meta
    return None, None


def _numpy_search(matrix: np.ndarray, query_vec: np.ndarray, k: int):
    """Cosine similarity search using numpy dot product."""
    scores = matrix @ query_vec.reshape(-1)     # (n,) inner products
    top_k  = np.argsort(scores)[::-1][:k]
    return scores[top_k], top_k


def _encode_query(query_text: str) -> np.ndarray:
    """
    Encode a free-text query into a single embedding vector.
    Uses the same SBERT model as nlp_engine, or TF-IDF fallback.
    """
    from nlp_engine import get_embeddings, clean_text
    text = clean_text(query_text) or "unknown"
    vec  = get_embeddings([text], cache_key="__query_tmp__")
    # Override cache — single query should not persist
    return vec[0].astype(np.float32)


# ── Search function 1: free-text query ───────────────────────────
def find_similar_complaints(
    query_text: str,
    k: int = 5,
    filter_segment: str = None,
    filter_sentiment: str = None,
) -> pd.DataFrame:
    """
    Find the k most semantically similar complaints/reviews
    to the given query_text.

    Returns a DataFrame with columns:
      rank, similarity_score, review_snippet, sentiment_label,
      segment_label, risk_level, intent_label, topic_label
    """
    index, meta = _load_index()
    if index is None:
        return pd.DataFrame({"error": ["FAISS index not built. Run build_index() first."]})

    query_vec = _encode_query(query_text)

    if FAISS_OK and hasattr(index, "search"):
        scores, indices = index.search(query_vec.reshape(1, -1), k * 3)
        scores  = scores[0]
        indices = indices[0]
    else:
        scores, indices = _numpy_search(index, query_vec, k * 3)

    # Apply filters on meta
    results = []
    for score, idx in zip(scores, indices):
        if idx < 0 or idx >= len(meta):
            continue
        row = meta.iloc[idx].to_dict()
        if filter_segment and row.get("segment_label") != filter_segment:
            continue
        if filter_sentiment and row.get("sentiment_label") != filter_sentiment:
            continue
        row["similarity_score"] = round(float(score), 4)
        results.append(row)
        if len(results) >= k:
            break

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    out.insert(0, "rank", range(1, len(out) + 1))

    # Trim review text for display
    if "clean_text" in out.columns:
        out["review_snippet"] = out["clean_text"].str[:120] + "…"
    elif "review_text" in out.columns:
        out["review_snippet"] = out["review_text"].str[:120] + "…"

    return out[[c for c in [
        "rank", "similarity_score", "review_snippet",
        "sentiment_label", "segment_label", "risk_level",
        "intent_label", "topic_label", "satisfaction_score",
        "category_preference",
    ] if c in out.columns]].reset_index(drop=True)


# ── Search function 2: customer-to-customer similarity ───────────
def find_similar_customers(
    customer_id: str,
    df_full: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 5,
) -> pd.DataFrame:
    """
    Given a customer_id, find k most similar customers by
    review-text embedding distance.

    Args:
        customer_id : value from df_full["customer_id"]
        df_full     : master dataframe (same order as embeddings)
        embeddings  : float32 array (n, d) for all rows

    Returns DataFrame of k similar customers with profile columns.
    """
    if "customer_id" not in df_full.columns:
        return pd.DataFrame({"error": ["customer_id column not found"]})

    match = df_full[df_full["customer_id"] == customer_id]
    if match.empty:
        return pd.DataFrame({"error": [f"Customer '{customer_id}' not found"]})

    row_idx = match.index[0]
    # Remap to positional index
    pos_idx = df_full.index.get_loc(row_idx)
    query_vec = embeddings[pos_idx].astype(np.float32)

    scores = (embeddings @ query_vec)             # cosine sim for all rows
    top_indices = np.argsort(scores)[::-1]

    # Exclude the query customer itself
    top_indices = [i for i in top_indices if i != pos_idx][:k]
    top_scores  = scores[top_indices]

    display_cols = [c for c in [
        "customer_id", "segment_label", "risk_level",
        "sentiment_label", "satisfaction_score",
        "category_preference", "avg_spend",
        "intent_label", "churn_proxy",
    ] if c in df_full.columns]

    result = df_full.iloc[top_indices][display_cols].copy()
    result.insert(0, "similarity_score", top_scores.round(4))
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


# ── Search function 3: abandonment-risk peers ─────────────────────
def find_abandonment_risk_peers(
    customer_id: str,
    df_full: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 5,
) -> pd.DataFrame:
    """
    Find customers who are:
      (a) semantically similar to the query customer (text similarity)
      (b) also have high abandonment_proxy (== 1) or high churn_proxy

    Useful for: "show me other customers like this one who also abandoned."
    """
    if "customer_id" not in df_full.columns:
        return pd.DataFrame()

    match = df_full[df_full["customer_id"] == customer_id]
    if match.empty:
        return pd.DataFrame()

    pos_idx   = df_full.index.get_loc(match.index[0])
    query_vec = embeddings[pos_idx].astype(np.float32)

    # Score all rows by cosine similarity
    sim_scores = embeddings @ query_vec

    # Build candidate mask: high-risk rows only
    risk_mask = pd.Series([False] * len(df_full), index=df_full.index)
    if "abandonment_proxy" in df_full.columns:
        risk_mask |= (df_full["abandonment_proxy"] == 1)
    if "churn_proxy" in df_full.columns:
        risk_mask |= (df_full["churn_proxy"] > 0.6)
    if "risk_level" in df_full.columns:
        risk_mask |= (df_full["risk_level"] == "High")

    candidate_positions = [
        i for i, (_, flag) in enumerate(zip(df_full.index, risk_mask))
        if flag and i != pos_idx
    ]

    if not candidate_positions:
        # Fallback: just return most similar regardless of risk
        return find_similar_customers(customer_id, df_full, embeddings, k)

    candidate_scores  = sim_scores[candidate_positions]
    top_local_indices = np.argsort(candidate_scores)[::-1][:k]
    top_positions     = [candidate_positions[i] for i in top_local_indices]
    top_scores        = candidate_scores[top_local_indices]

    display_cols = [c for c in [
        "customer_id", "segment_label", "risk_level",
        "abandonment_proxy", "churn_proxy",
        "sentiment_label", "intent_label",
        "satisfaction_score", "category_preference",
    ] if c in df_full.columns]

    result = df_full.iloc[top_positions][display_cols].copy()
    result.insert(0, "similarity_score", top_scores.round(4))
    result.insert(0, "rank", range(1, len(result) + 1))
    return result.reset_index(drop=True)


# ── Index status ──────────────────────────────────────────────────
def index_status() -> dict:
    """Returns a dict describing whether the index is ready."""
    index, meta = _load_index()
    if index is None:
        return {"ready": False, "n_vectors": 0, "backend": "none"}

    if FAISS_OK and hasattr(index, "ntotal"):
        n = index.ntotal
        backend = "faiss"
    else:
        n = len(index) if hasattr(index, "__len__") else 0
        backend = "numpy"

    return {
        "ready":    True,
        "n_vectors": n,
        "backend":  backend,
        "meta_rows": len(meta) if meta is not None else 0,
    }