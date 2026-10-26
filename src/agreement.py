"""Inter-annotator agreement on the shared 100-review sample (Cohen's kappa)."""
import sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score


def agreement(a_path: str, b_path: str) -> dict:
    a, b = pd.read_csv(a_path), pd.read_csv(b_path)
    m = a.merge(b, on=["review_id", "text"], suffixes=("_a", "_b"))
    return {"n": len(m),
            "kappa_aspect": cohen_kappa_score(m.aspect_a, m.aspect_b),
            "kappa_sentiment": cohen_kappa_score(m.sentiment_a, m.sentiment_b)}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python src/agreement.py annotator_a.csv annotator_b.csv")
    print(agreement(*sys.argv[1:]))
