"""After hand-filling aspect_correct / sentiment_correct (y or n) in results/validation_sample.csv."""
import sys
import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/validation_sample.csv")
for col in ("aspect_correct", "sentiment_correct"):
    v = d[col].astype(str).str.strip().str.lower().map({"y": 1, "n": 0}).dropna()
    if len(v) == 0:
        print(f"{col}: not filled yet"); continue
    p = v.mean(); se = np.sqrt(p * (1 - p) / len(v))
    print(f"{col}: {p:.1%} correct on {len(v)} clauses (95% CI {p - 1.96 * se:.1%} to {p + 1.96 * se:.1%})")
