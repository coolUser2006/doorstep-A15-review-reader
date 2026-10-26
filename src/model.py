"""Baselines and the supervised model. Majority -> keyword -> TF-IDF + logistic regression."""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SEED = 42

KEYWORDS = {
    "punctuality": ["late", "on time", "slot", "delayed", "arrived", "waited", "wait", "punctual", "early", "turn up", "showed up"],
    "wash_quality": ["dirt", "dirty", "spotless", "clean", "cleaned", "shine", "shining", "finish", "streaky", "water marks",
                     "polish", "interior", "windshield", "paint", "scratch", "scratches"],
    "staff_behaviour": ["crew", "staff", "rude", "polite", "friendly", "guy", "team", "respectful", "professional", "argued"],
    "price": ["price", "expensive", "cheap", "cheaper", "money", "charged", "cost", "value", "worth", "extra", "bill"],
    "booking_experience": ["app", "booking", "book", "reschedule", "rescheduling", "payment", "website", "cancel", "cancelling"],
}
_PAT = {a: re.compile(r"\b(?:" + "|".join(map(re.escape, k)) + r")\b", re.I) for a, k in KEYWORDS.items()}


def keyword_aspect(text: str) -> str:
    """Most keyword hits wins; ties go to the first aspect listed; no hit -> 'other'."""
    hits = {a: len(p.findall(text)) for a, p in _PAT.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else "other"


_sia = SentimentIntensityAnalyzer()
_sia.lexicon.update({"late": -1.5, "delayed": -2.0, "rude": -2.5, "dirt": -1.5, "dirty": -2.0, "streaky": -2.0,
                     "expensive": -2.0, "crashed": -2.0, "failed": -2.0, "careless": -2.0, "missed": -1.5,
                     "barely": -1.0, "scratches": -2.0, "ignored": -2.0, "hassle": -1.5, "forever": -1.5,
                     "reliable": 1.5, "unreliable": -2.0, "problem": -1.5, "problems": -1.5, "issue": -1.0, "issues": -1.0,
                     "broke": -2.0, "repair": -1.0, "repairs": -1.0, "recall": -1.5, "comfortable": 1.5, "uncomfortable": -2.0,
                     "leak": -2.0, "leaks": -2.0, "overpriced": -2.0, "spotless": 2.5, "shining": 2.0, "punctual": 2.0, "polite": 2.0, "respectful": 2.0,
                     "reasonable": 1.5, "easy": 1.5, "simple": 1.5, "smooth": 1.5, "quick": 1.5})


def keyword_sentiment(text: str, thr: float = 0.25) -> str:
    c = _sia.polarity_scores(text)["compound"]
    return "positive" if c >= thr else "negative" if c <= -thr else "neutral"


def make_tfidf_lr(seed: int = SEED) -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))])
