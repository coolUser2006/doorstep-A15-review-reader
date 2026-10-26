"""Produce results/review_aspects.csv (the handover), failure cases, and the driver analysis."""
import argparse, os, random, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import ASPECTS, explode_clauses, load_labelled, load_reviews
from model import make_tfidf_lr, keyword_aspect, keyword_sentiment

SEED = 42
random.seed(SEED); np.random.seed(SEED)


def predict_all(reviews, lab):
    cl = explode_clauses(reviews)
    m_a, m_s = make_tfidf_lr().fit(lab.text, lab.aspect), make_tfidf_lr().fit(lab.text, lab.sentiment)
    cl["aspect"], cl["sentiment"] = m_a.predict(cl.text), m_s.predict(cl.text)
    # Clauses that were hand-labelled get out-of-fold predictions so evaluation is not in-sample.
    cv = GroupKFold(5)
    oof_a = cross_val_predict(make_tfidf_lr(), lab.text, lab.aspect, groups=lab.review_id, cv=cv)
    oof_s = cross_val_predict(make_tfidf_lr(), lab.text, lab.sentiment, groups=lab.review_id, cv=cv)
    key = {(r, t): (a, s) for r, t, a, s in zip(lab.review_id, lab.text, oof_a, oof_s)}
    for i, (r, t) in enumerate(zip(cl.review_id, cl.text)):
        if (r, t) in key:
            cl.loc[i, ["aspect", "sentiment"]] = key[(r, t)]
    return cl, oof_a, oof_s


def to_review_aspects(cl):
    """Handover. One row per (review, aspect). Required columns first; extras appended at the end."""
    rank = {"negative": 2, "positive": 1, "neutral": 0}
    rows = []
    for (rid, asp), g in cl[cl.aspect != "other"].groupby(["review_id", "aspect"], sort=False):
        sent = max(g.sentiment, key=rank.get)  # any negative clause makes the aspect negative
        rows.append({"review": g.review_text.iloc[0], "aspect": asp, "sentiment": sent,
                     "review_id": rid, "rating": g.rating.iloc[0]})
    out = pd.DataFrame(rows)
    out["rating"] = out["rating"].astype("Int64")  # missing stays an empty cell
    return out


def driver_analysis(ra, aspects=ASPECTS, n_boot=300):
    """Which aspect most strongly drives a bad (<=2) rating? Odds ratios with bootstrap 95% CIs."""
    if ra["rating"].isna().all():
        return None
    wide = ra.assign(neg=(ra.sentiment == "negative").astype(int)).pivot_table(
        index="review_id", columns="aspect", values="neg", aggfunc="max", fill_value=0)
    for a in aspects:
        if a not in wide:
            wide[a] = 0
    wide = wide[list(aspects)]
    rating = ra.drop_duplicates("review_id").set_index("review_id")["rating"].astype(float).reindex(wide.index)
    keep = rating.notna()
    X, y = wide[keep].values, (rating[keep] <= 2).astype(int).values
    fit = lambda X_, y_: np.exp(LogisticRegression(C=10, max_iter=1000).fit(X_, y_).coef_[0])
    point, rng, boots = fit(X, y), np.random.default_rng(SEED), []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(fit(X[i], y[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    r = rating[keep].values
    out = pd.DataFrame({"aspect": list(aspects), "n_reviews_negative": X.sum(0),
                        "mean_rating_if_negative": [r[X[:, j] == 1].mean() if X[:, j].any() else np.nan for j in range(len(aspects))],
                        "mean_rating_otherwise": [r[X[:, j] == 0].mean() for j in range(len(aspects))],
                        "odds_ratio": point, "ci_low": lo, "ci_high": hi}).round(3)
    out.attrs["bad_rate"] = float(y.mean())
    return out.sort_values("odds_ratio", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default="data/processed/reviews.csv")
    ap.add_argument("--labelled", default="data/processed/labelled.csv")
    a = ap.parse_args()
    for p in (a.reviews, a.labelled):
        if not os.path.exists(p):
            sys.exit(f"{p} not found (for a dry run use the *_synthetic.csv files from src/make_synthetic.py)")
    reviews, lab = load_reviews(a.reviews), load_labelled(a.labelled)
    cl, oof_a, oof_s = predict_all(reviews, lab)
    ra = to_review_aspects(cl)
    ra.to_csv("results/review_aspects.csv", index=False)

    wrong = lab.assign(pred_aspect=oof_a, pred_sentiment=oof_s,
                       keyword_aspect=lab.text.map(keyword_aspect), keyword_sentiment=lab.text.map(keyword_sentiment))
    wrong = wrong[(wrong.aspect != wrong.pred_aspect) | (wrong.sentiment != wrong.pred_sentiment)]
    wrong.to_csv("results/failure_cases.csv", index=False)

    drv = driver_analysis(ra)
    if drv is not None:
        drv.to_csv("results/driver_analysis.csv", index=False)
        d = drv.iloc[::-1]
        plt.figure(figsize=(7, 4))
        plt.barh(d.aspect, d.odds_ratio, xerr=[d.odds_ratio - d.ci_low, d.ci_high - d.odds_ratio], capsize=3)
        plt.axvline(1, color="gray", linestyle="--")
        plt.xlabel("Odds ratio of a bad (<=2 star) rating when aspect is negative")
        plt.title(f"What drives bad ratings (baseline bad rate {drv.attrs['bad_rate']:.0%})")
        plt.tight_layout(); plt.savefig("results/plots/driver_odds.png", dpi=140); plt.close()
        print(drv.to_string(index=False))
    print(f"\nreview_aspects.csv: {len(ra)} rows | failure cases: {len(wrong)} of {len(lab)} labelled clauses")


if __name__ == "__main__":
    main()
