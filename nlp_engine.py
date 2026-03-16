# nlp_engine.py
"""
NLP layer for the Customer Journey Analysis system.

Processes all available text columns (review_text, feedback, comments, etc.)
and adds the following columns to the dataframe — one per row:

  sentiment_score    : float  -1.0 to +1.0  (TextBlob polarity)
  sentiment_label    : str    "Positive" | "Neutral" | "Negative"
  complaint_count    : int    count of complaint-signal keywords found
  urgency_score      : float  0.0 to 1.0
  frustration_flag   : int    0 / 1
  pricing_concern    : int    0 / 1
  trust_issue        : int    0 / 1
  confusion_flag     : int    0 / 1
  intent_label       : str    dominant detected intent
  topic_label        : str    dominant TF-IDF topic cluster (if text available)

All scores are DERIVED from available text columns.
If no text column exists, neutral/zero defaults are used and flagged.
"""

import re
import os
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF           # topic modelling
from sklearn.preprocessing import normalize

from utils import save_model, load_model, MODEL_DIR

# ── Optional heavy imports (graceful fallback) ────────────────────
try:
    from textblob import TextBlob
    TEXTBLOB_OK = True
except ImportError:
    TEXTBLOB_OK = False

try:
    from sentence_transformers import SentenceTransformer
    SBERT_OK = True
except ImportError:
    SBERT_OK = False

try:
    import nltk
    for pkg in ["punkt", "stopwords", "wordnet"]:
        nltk.download(pkg, quiet=True)
    NLTK_OK = True
except Exception:
    NLTK_OK = False


# ── Keyword dictionaries ──────────────────────────────────────────
COMPLAINT_KEYWORDS = [
    "terrible", "awful", "worst", "horrible", "disappointing",
    "broken", "defective", "damaged", "unusable", "waste",
    "refund", "return", "complaint", "issue", "problem",
    "bad quality", "poor quality", "not working", "didn't work",
    "never arrived", "late delivery", "wrong item", "missing",
    "fake", "scam", "fraud", "misleading",
]

URGENCY_KEYWORDS = [
    "urgent", "immediately", "asap", "right now", "emergency",
    "critical", "must", "need now", "time sensitive", "deadline",
    "still waiting", "weeks ago", "months ago", "no response",
    "follow up", "escalate",
]

FRUSTRATION_KEYWORDS = [
    "frustrated", "angry", "annoyed", "fed up", "ridiculous",
    "unacceptable", "outrageous", "disgusted", "furious",
    "never again", "last time", "done with", "switching",
]

PRICING_KEYWORDS = [
    "expensive", "overpriced", "too costly", "not worth",
    "cheaper elsewhere", "price hike", "cost too much",
    "better price", "discount", "coupon", "promo",
    "price match", "refund", "value for money",
]

TRUST_KEYWORDS = [
    "fake", "scam", "fraud", "counterfeit", "not genuine",
    "misleading", "false", "lied", "deceptive", "untrustworthy",
    "suspicious", "not as described", "bait and switch",
]

CONFUSION_KEYWORDS = [
    "confusing", "unclear", "hard to find", "complicated",
    "don't understand", "not sure how", "couldn't figure",
    "difficult to navigate", "where is", "how do i",
    "no instructions", "no guide", "setup problem",
]

INTENT_RULES = {
    "complaint":    COMPLAINT_KEYWORDS,
    "frustration":  FRUSTRATION_KEYWORDS,
    "pricing":      PRICING_KEYWORDS,
    "trust_issue":  TRUST_KEYWORDS,
    "confusion":    CONFUSION_KEYWORDS,
    "urgency":      URGENCY_KEYWORDS,
}


# ── Text preprocessing ────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags, normalise whitespace."""
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)           # strip HTML
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # strip URLs
    text = re.sub(r"[^a-z0-9\s'.,!?-]", " ", text)  # keep punctuation signals
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_count(text: str, keywords: list) -> int:
    """Count how many keywords from the list appear in text."""
    if not text:
        return 0
    return sum(1 for kw in keywords if kw in text)


def _keyword_flag(text: str, keywords: list, threshold: int = 1) -> int:
    return int(_keyword_count(text, keywords) >= threshold)


# ── Row-level scoring ─────────────────────────────────────────────
def _score_row(text: str) -> dict:
    """Return all NLP signal scores for a single text string."""
    text_clean = clean_text(text)

    # ── Sentiment via TextBlob ────────────────────────────────────
    if TEXTBLOB_OK and text_clean:
        blob = TextBlob(text_clean)
        polarity    = round(blob.sentiment.polarity, 4)
        subjectivity = round(blob.sentiment.subjectivity, 4)
    else:
        # Rule-based fallback
        pos_words = ["good", "great", "excellent", "love", "perfect",
                     "amazing", "happy", "satisfied", "recommend", "best"]
        neg_words = COMPLAINT_KEYWORDS + FRUSTRATION_KEYWORDS
        pos_count = _keyword_count(text_clean, pos_words)
        neg_count = _keyword_count(text_clean, neg_words)
        if pos_count + neg_count == 0:
            polarity = 0.0
        else:
            polarity = (pos_count - neg_count) / (pos_count + neg_count)
        subjectivity = 0.5

    # ── Sentiment label ───────────────────────────────────────────
    if polarity > 0.05:
        sent_label = "Positive"
    elif polarity < -0.05:
        sent_label = "Negative"
    else:
        sent_label = "Neutral"

    # ── Intent flags ──────────────────────────────────────────────
    complaint_cnt   = _keyword_count(text_clean, COMPLAINT_KEYWORDS)
    urgency_raw     = _keyword_count(text_clean, URGENCY_KEYWORDS)
    frustration_f   = _keyword_flag(text_clean,  FRUSTRATION_KEYWORDS)
    pricing_f       = _keyword_flag(text_clean,  PRICING_KEYWORDS)
    trust_f         = _keyword_flag(text_clean,  TRUST_KEYWORDS)
    confusion_f     = _keyword_flag(text_clean,  CONFUSION_KEYWORDS)

    # Normalise urgency to 0-1 (cap at 5 signals)
    urgency_score = round(min(urgency_raw / 5.0, 1.0), 4)

    # ── Dominant intent label ─────────────────────────────────────
    intent_scores = {k: _keyword_count(text_clean, v) for k, v in INTENT_RULES.items()}
    if max(intent_scores.values()) == 0:
        intent_label = "neutral"
    else:
        intent_label = max(intent_scores, key=intent_scores.get)

    return {
        "sentiment_score":   polarity,
        "subjectivity":      subjectivity,
        "sentiment_label":   sent_label,
        "complaint_count":   complaint_cnt,
        "urgency_score":     urgency_score,
        "frustration_flag":  frustration_f,
        "pricing_concern":   pricing_f,
        "trust_issue":       trust_f,
        "confusion_flag":    confusion_f,
        "intent_label":      intent_label,
        "clean_text":        text_clean,
    }


# ── Main: enrich dataframe ────────────────────────────────────────
def run_nlp_pipeline(
    df: pd.DataFrame,
    text_col: str = "review_text",
) -> pd.DataFrame:
    """
    Adds all NLP-derived columns to df.
    If text_col is absent or empty, uses neutral defaults.
    Returns enriched dataframe.
    """
    df = df.copy()

    # ── Find the best available text column ──────────────────────
    candidate_cols = [
        "review_text", "feedback", "comments",
        "customer_review", "product_review", "survey_text",
    ]
    actual_col = None
    for col in [text_col] + candidate_cols:
        if col in df.columns and df[col].notna().sum() > 10:
            actual_col = col
            break

    if actual_col is None:
        # No usable text column — fill with neutral defaults
        df["sentiment_score"]  = 0.0
        df["subjectivity"]     = 0.5
        df["sentiment_label"]  = "Neutral"
        df["complaint_count"]  = 0
        df["urgency_score"]    = 0.0
        df["frustration_flag"] = 0
        df["pricing_concern"]  = 0
        df["trust_issue"]      = 0
        df["confusion_flag"]   = 0
        df["intent_label"]     = "neutral"
        df["clean_text"]       = ""
        df["topic_label"]      = "no_text"
        df["_nlp_source"]      = "no_text_column"
        print("[NLP] No text column found — using neutral defaults.")
        return df

    print(f"[NLP] Using text column: '{actual_col}' ({df[actual_col].notna().sum()} rows)")

    # ── Row-level scoring (vectorised-style via apply) ────────────
    texts = df[actual_col].fillna("").astype(str)
    scores = texts.apply(_score_row)
    score_df = pd.DataFrame(scores.tolist(), index=df.index)

    for col in score_df.columns:
        df[col] = score_df[col]

    df["_nlp_source"] = actual_col

    # ── TF-IDF keyword extraction ─────────────────────────────────
    clean_texts = df["clean_text"].tolist()
    non_empty   = [t for t in clean_texts if t.strip()]

    if len(non_empty) >= 10:
        tfidf_result = run_tfidf(clean_texts)
        df["topic_label"] = tfidf_result.get("row_topics", "unknown")
        # Save vectorizer for dashboard keyword display
        save_model(tfidf_result.get("vectorizer"), "tfidf_vectorizer")
        save_model(tfidf_result.get("nmf_model"),  "nmf_model")
        save_model(tfidf_result.get("topic_terms"), "topic_terms")
    else:
        df["topic_label"] = "insufficient_text"

    return df


# ── TF-IDF + NMF topic modelling ─────────────────────────────────
def run_tfidf(
    texts: list,
    n_topics: int = 6,
    n_top_words: int = 8,
) -> dict:
    """
    Fits TF-IDF + NMF topic model on the provided texts.
    Returns:
      vectorizer   : fitted TfidfVectorizer
      nmf_model    : fitted NMF
      topic_terms  : {topic_id: [word, word, ...]}
      row_topics   : list of dominant topic label per row (same len as texts)
      tfidf_matrix : sparse matrix
    """
    # Filter empties but keep index alignment
    valid_mask = [bool(t and t.strip()) for t in texts]
    valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]

    if len(valid_texts) < 5:
        return {
            "row_topics": ["no_text"] * len(texts),
            "topic_terms": {},
            "vectorizer": None,
            "nmf_model": None,
        }

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(valid_texts)

    # NMF topic decomposition
    actual_topics = min(n_topics, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
    actual_topics = max(actual_topics, 2)

    nmf = NMF(
        n_components=actual_topics,
        random_state=42,
        max_iter=300,
    )
    W = nmf.fit_transform(tfidf_matrix)   # shape: (n_docs, n_topics)

    # Get top words per topic
    feature_names = vectorizer.get_feature_names_out()
    topic_terms = {}
    topic_labels_map = {}
    for i, component in enumerate(nmf.components_):
        top_idx  = component.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[j] for j in top_idx]
        topic_terms[i] = top_words
        # Use top 2 words as label
        topic_labels_map[i] = f"{top_words[0]}_{top_words[1]}"

    # Assign dominant topic to each valid row
    dominant_topics_valid = W.argmax(axis=1)
    row_topic_labels_valid = [topic_labels_map[t] for t in dominant_topics_valid]

    # Re-expand to original length (fill empties with "no_text")
    row_topics = []
    valid_iter = iter(row_topic_labels_valid)
    for ok in valid_mask:
        row_topics.append(next(valid_iter) if ok else "no_text")

    return {
        "row_topics":   row_topics,
        "topic_terms":  topic_terms,
        "topic_labels_map": topic_labels_map,
        "vectorizer":   vectorizer,
        "nmf_model":    nmf,
        "tfidf_matrix": tfidf_matrix,
        "W":            W,
    }


# ── BERT / Sentence-Transformer embeddings ────────────────────────
_SBERT_MODEL = None   # module-level cache so we load once per session

def get_embeddings(
    texts: list,
    model_name: str = "all-MiniLM-L6-v2",   # 384-d, fast, good quality
    batch_size: int = 64,
    cache_key:  str = "sbert_embeddings",
) -> np.ndarray:
    """
    Returns L2-normalised BERT embeddings as float32 numpy array.
    Shape: (len(texts), 384)

    Falls back to TF-IDF dense vectors if sentence-transformers
    is not installed.

    Results are cached to models/sbert_embeddings.pkl so the
    dashboard doesn't re-embed on every reload.
    """
    global _SBERT_MODEL

    # Try loading cached embeddings first
    cached = load_model(cache_key)
    if cached is not None and cached.shape[0] == len(texts):
        print(f"[NLP] Loaded cached embeddings: {cached.shape}")
        return cached

    clean = [clean_text(t) if t else "" for t in texts]

    if SBERT_OK:
        if _SBERT_MODEL is None:
            print(f"[NLP] Loading SBERT model: {model_name}")
            _SBERT_MODEL = SentenceTransformer(model_name)

        embeddings = _SBERT_MODEL.encode(
            clean,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
    else:
        # Fallback: TF-IDF dense vectors
        print("[NLP] sentence-transformers not available. Using TF-IDF fallback.")
        vec = TfidfVectorizer(max_features=384, stop_words="english")
        non_empty = [t if t else "empty" for t in clean]
        sparse = vec.fit_transform(non_empty)
        embeddings = sparse.toarray().astype(np.float32)

    # L2 normalise so cosine similarity = dot product
    embeddings = normalize(embeddings, norm="l2").astype(np.float32)

    save_model(embeddings, cache_key)
    print(f"[NLP] Embeddings computed and cached: {embeddings.shape}")
    return embeddings


# ── Dashboard helper: top TF-IDF keywords ────────────────────────
def get_top_keywords(
    df: pd.DataFrame,
    n: int = 20,
    filter_sentiment: str = None,    # "Positive" | "Negative" | None
    filter_segment:   str = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [keyword, tfidf_score]
    suitable for a bar chart in the dashboard.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    if filter_sentiment and "sentiment_label" in df.columns:
        mask &= df["sentiment_label"] == filter_sentiment
    if filter_segment and "segment_label" in df.columns:
        mask &= df["segment_label"] == filter_segment

    texts = df.loc[mask, "clean_text"].fillna("").tolist() if "clean_text" in df.columns else []
    if not texts or all(t == "" for t in texts):
        return pd.DataFrame({"keyword": [], "tfidf_score": []})

    vec = TfidfVectorizer(max_features=200, stop_words="english", ngram_range=(1, 2))
    try:
        X = vec.fit_transform(texts)
        scores = X.mean(axis=0).A1
        terms  = vec.get_feature_names_out()
        kw_df  = pd.DataFrame({"keyword": terms, "tfidf_score": scores})
        return kw_df.sort_values("tfidf_score", ascending=False).head(n).reset_index(drop=True)
    except Exception:
        return pd.DataFrame({"keyword": [], "tfidf_score": []})


# ── Complaint theme summary ───────────────────────────────────────
def get_complaint_themes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a breakdown of complaint signal counts across the dataframe.
    Used for the NLP Insights page theme bar chart.
    """
    theme_cols = {
        "Complaint signals":  "complaint_count",
        "Frustration":        "frustration_flag",
        "Pricing concerns":   "pricing_concern",
        "Trust issues":       "trust_issue",
        "Confusion / UX":     "confusion_flag",
    }
    rows = []
    for theme, col in theme_cols.items():
        if col in df.columns:
            if col == "complaint_count":
                count = int((df[col] > 0).sum())
            else:
                count = int(df[col].sum())
            rows.append({"theme": theme, "count": count,
                         "pct": round(count / len(df) * 100, 1)})
    return pd.DataFrame(rows)


# ── Sentiment distribution summary ───────────────────────────────
def get_sentiment_summary(df: pd.DataFrame) -> dict:
    if "sentiment_label" not in df.columns:
        return {}
    vc = df["sentiment_label"].value_counts()
    total = len(df)
    return {
        label: {"count": int(vc.get(label, 0)),
                "pct": round(vc.get(label, 0) / total * 100, 1)}
        for label in ["Positive", "Neutral", "Negative"]
    }