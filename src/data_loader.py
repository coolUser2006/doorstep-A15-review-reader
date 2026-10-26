"""Loading, validation and clause splitting. Shared by every other script."""
import random, re
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ASPECTS = ["punctuality", "wash_quality", "staff_behaviour", "price", "booking_experience"]
LABELS = ASPECTS + ["other"]
SENTIMENTS = ["negative", "neutral", "positive"]

_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")
_SPLIT = re.compile(r"(?<=[.!?])\s+|\s+(?:but|however|although)\s+", re.I)


def split_clauses(text: str) -> list[str]:
    """One review -> clause-level units, so each aspect gets its own sentiment."""
    text = _WS.sub(" ", _URL.sub(" ", str(text))).strip()
    return [p.strip() for p in _SPLIT.split(text) if p and len(p.strip()) > 2]


def load_reviews(path: str) -> pd.DataFrame:
    """Columns: review_id, text, rating (rating optional, 1-5; empty = unknown)."""
    df = pd.read_csv(path)
    for c in ("review_id", "text"):
        if c not in df.columns:
            raise ValueError(f"{path}: missing column '{c}'")
    return df.dropna(subset=["text"]).drop_duplicates("review_id").reset_index(drop=True)


def load_labelled(path: str) -> pd.DataFrame:
    """Clause-level gold labels: review_id, text, aspect, sentiment."""
    df = pd.read_csv(path)
    need = {"review_id", "text", "aspect", "sentiment"}
    if need - set(df.columns):
        raise ValueError(f"{path}: missing columns {need - set(df.columns)}")
    df = df.dropna(subset=["text", "aspect", "sentiment"])
    bad_a = set(df.aspect) - set(LABELS)
    bad_s = set(df.sentiment) - set(SENTIMENTS)
    if bad_a or bad_s:
        raise ValueError(f"Unknown labels. aspects={bad_a} sentiments={bad_s}")
    return df.reset_index(drop=True)


def explode_clauses(reviews: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in reviews.itertuples(index=False):
        for c in split_clauses(r.text):
            rows.append({"review_id": r.review_id, "review_text": r.text, "text": c,
                         "rating": getattr(r, "rating", np.nan)})
    return pd.DataFrame(rows)
