"""Unlabelled pipeline on real reviews (Edmunds car-owner reviews). No hand labels needed:
keyword aspects + VADER sentiment per clause -> review_aspects -> driver analysis -> brand breakdown.
Accuracy is NOT measured here (no gold labels); use results/validation_sample.csv + score_validation.py."""
import argparse, os, random, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import explode_clauses, load_reviews
from model import keyword_sentiment
from evaluate import driver_analysis

SEED = 42
random.seed(SEED); np.random.seed(SEED)

CAR_ASPECTS = {
    "reliability_repairs": ["reliable", "reliability", "unreliable", "broke", "broken", "breakdown", "repair", "repairs", "problem", "problems",
                            "issue", "issues", "recall", "leak", "leaks", "maintenance", "trouble", "failed", "defect", "stalled"],
    "comfort_interior": ["comfortable", "comfort", "uncomfortable", "seat", "seats", "interior", "quiet", "noise", "noisy", "ride", "cabin",
                         "leather", "legroom", "roomy", "spacious", "trunk", "cargo"],
    "performance_handling": ["power", "powerful", "acceleration", "fast", "handling", "handles", "engine", "transmission", "torque", "horsepower",
                             "braking", "brakes", "steering", "drive", "driving", "speed"],
    "fuel_economy": ["mpg", "gas", "fuel", "mileage", "economy", "efficient", "efficiency", "hybrid"],
    "price_value": ["price", "value", "cost", "expensive", "cheap", "money", "afford", "affordable", "deal", "worth", "payment", "overpriced"],
    "dealer_service": ["dealer", "dealership", "dealers", "salesman", "salesperson", "service department", "customer service", "warranty"],
    "tech_features": ["infotainment", "navigation", "bluetooth", "screen", "radio", "technology", "camera", "touchscreen", "carplay", "android auto"],
}
ASPECTS = list(CAR_ASPECTS)
_PAT = {a: re.compile(r"\b(?:" + "|".join(map(re.escape, k)) + r")\b", re.I) for a, k in CAR_ASPECTS.items()}


def detect(clause: str) -> list[str]:
    """All aspects mentioned in a clause (a clause can touch more than one)."""
    return [a for a, p in _PAT.items() if p.search(clause)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", default="data/raw/reviews_cars.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    reviews = load_reviews(a.reviews)
    brand = reviews.set_index("review_id")["brand"] if "brand" in reviews else None
    cl = explode_clauses(reviews)
    cl["sentiment"] = cl.text.map(keyword_sentiment)
    cl["aspects"] = cl.text.map(detect)
    m = cl.explode("aspects").dropna(subset=["aspects"]).rename(columns={"aspects": "aspect"})

    rank = {"negative": 2, "positive": 1, "neutral": 0}
    rows = []
    for (rid, asp), g in m.groupby(["review_id", "aspect"], sort=False):
        rows.append({"review_id": rid, "aspect": asp, "sentiment": max(g.sentiment, key=rank.get),
                     "rating": int(g.rating.iloc[0]), "brand": brand[rid] if brand is not None else "",
                     "snippet": g.review_text.iloc[0][:200]})
    ra = pd.DataFrame(rows)
    os.makedirs(f"{a.out}/plots", exist_ok=True)
    ra.to_csv(f"{a.out}/review_aspects_cars.csv", index=False)

    print(f"{len(reviews)} reviews, {len(cl)} clauses, {len(ra)} (review, aspect) rows; "
          f"{(cl.aspects.str.len() == 0).mean():.0%} of clauses mention no aspect")
    cover = ra.groupby("aspect").agg(reviews=("review_id", "nunique"),
                                     negative_share=("sentiment", lambda s: (s == "negative").mean())).round(3)
    print(cover.sort_values("reviews", ascending=False).to_string())
    cover.to_csv(f"{a.out}/aspect_coverage.csv")

    drv = driver_analysis(ra, ASPECTS)
    drv.to_csv(f"{a.out}/driver_analysis_cars.csv", index=False)
    print(f"\nBad-rating (<=2 stars) base rate: {drv.attrs['bad_rate']:.1%}\n", drv.to_string(index=False))
    d = drv.iloc[::-1]
    plt.figure(figsize=(8, 4.2))
    plt.barh(d.aspect, d.odds_ratio, xerr=[d.odds_ratio - d.ci_low, d.ci_high - d.odds_ratio], capsize=3)
    plt.axvline(1, color="gray", linestyle="--")
    plt.xlabel("Odds ratio of a <=2 star rating when the aspect is negative (95% bootstrap CI)")
    plt.title(f"What drives bad owner ratings (base rate {drv.attrs['bad_rate']:.0%})")
    plt.tight_layout(); plt.savefig(f"{a.out}/plots/driver_odds_cars.png", dpi=140); plt.close()

    if brand is not None:  # brand league table for the most-complained aspect, min 20 mentions
        neg = ra.assign(neg=(ra.sentiment == "negative"))
        t = neg.groupby(["brand", "aspect"]).agg(n=("neg", "size"), negative_rate=("neg", "mean")).reset_index()
        t[t.n >= 20].to_csv(f"{a.out}/brand_aspect_negative_rate.csv", index=False)

    # validation sample: fill the two empty columns by hand (y/n), then run score_validation.py
    v = m.sample(min(150, len(m)), random_state=SEED)[["review_id", "text", "aspect", "sentiment"]]
    v = v.rename(columns={"aspect": "pred_aspect", "sentiment": "pred_sentiment"})
    v["aspect_correct"], v["sentiment_correct"] = "", ""
    v.to_csv(f"{a.out}/validation_sample.csv", index=False)


if __name__ == "__main__":
    main()
