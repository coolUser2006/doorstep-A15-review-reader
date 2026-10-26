import os, sys, tempfile
sys.path.insert(0, "src")
import pandas as pd
from data_loader import split_clauses
from model import keyword_aspect, keyword_sentiment
from agreement import agreement


def test_split_clauses():
    assert len(split_clauses("Arrived late. The crew was rude.")) == 2
    assert len(split_clauses("Nice wash but too expensive")) == 2


def test_keyword_baseline():
    assert keyword_aspect("The crew arrived an hour late.") == "punctuality"
    assert keyword_aspect("Parked in the basement.") == "other"
    assert keyword_sentiment("The crew was rude and careless.") == "negative"


def test_kappa_perfect():
    d = pd.DataFrame({"review_id": ["RV00001", "RV00002", "RV00003"], "text": ["a", "b", "c"],
                      "aspect": ["price", "other", "punctuality"], "sentiment": ["negative", "neutral", "positive"]})
    with tempfile.TemporaryDirectory() as t:
        d.to_csv(f"{t}/a.csv", index=False); d.to_csv(f"{t}/b.csv", index=False)
        r = agreement(f"{t}/a.csv", f"{t}/b.csv")
    assert r["kappa_aspect"] == 1.0 and r["kappa_sentiment"] == 1.0
