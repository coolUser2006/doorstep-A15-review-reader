"""SYNTHETIC Doorstep-style reviews, for developing the pipeline before real labels exist.
Do NOT report numbers from this data as findings. Ratings are generated with KNOWN aspect
weights (punctuality > wash quality > staff > price > booking) so the driver analysis can
be sanity-checked. 6% of labels are randomly flipped to mimic annotator disagreement."""
import os, random, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import SEED, split_clauses

random.seed(SEED); np.random.seed(SEED)

T = {  # aspect -> sentiment -> clauses (some deliberately avoid obvious keywords)
 "punctuality": {
  "negative": ["The crew arrived an hour late.", "Showed up well after the slot I booked.", "Waited forever for the van to turn up.",
               "They were delayed and never called.", "Nobody came until the afternoon."],
  "positive": ["Arrived right on time.", "The crew turned up exactly at my slot.", "Punctual and quick to start.", "They were early, which I appreciated."]},
 "wash_quality": {
  "negative": ["The car still had dirt on the bumpers.", "Left water marks all over the windshield.", "The interior was barely cleaned.",
               "Missed spots and the polish looked streaky.", "My paint has fresh scratches now."],
  "positive": ["The car was spotless and shining.", "Great finish on the paint.", "The interior looked brand new.", "Windows were crystal clear."]},
 "staff_behaviour": {
  "negative": ["The crew was rude and dismissive.", "One guy kept arguing with me.", "Staff were careless and unprofessional.",
               "They ignored my instructions completely."],
  "positive": ["The crew was polite and respectful.", "Friendly and helpful team.", "They explained everything patiently.", "Very professional staff."]},
 "price": {
  "negative": ["Way too expensive for a basic wash.", "Charged extra without telling me.", "Not worth the money.", "The bill was higher than quoted."],
  "positive": ["Very reasonable price.", "Good value for the money.", "Cheaper than going to a car wash.", "Fair rates for the quality."]},
 "booking_experience": {
  "negative": ["The app crashed while booking.", "Could not reschedule my appointment.", "Payment failed twice in the app.", "Cancelling was a hassle."],
  "positive": ["Booking was quick and easy.", "Loved how simple the app is.", "Rescheduling took seconds.", "Paying through the app was smooth."]},
}
OTHER = ["I have been using this for a month.", "I parked in the office basement.", "Would tell my colleagues about it.", "It was a Tuesday morning."]
WEIGHT = {"punctuality": 1.7, "wash_quality": 1.4, "staff_behaviour": 1.0, "price": 0.5, "booking_experience": 0.35}
P_NEG = {"punctuality": .35, "wash_quality": .3, "staff_behaviour": .25, "price": .3, "booking_experience": .25}
ASPECTS = list(T)


def make(n=1200, out_dir="data/processed", noise=0.06):
    reviews, labelled = [], []
    for i in range(n):
        rid = f"RV{i:05d}"
        k = random.choice([1, 2, 2, 3])
        clauses, score = [], 4.6
        for asp in random.sample(ASPECTS, k):
            neg = random.random() < P_NEG[asp]
            sent = "negative" if neg else "positive"
            clauses.append((random.choice(T[asp][sent]), asp, sent))
            score += -WEIGHT[asp] if neg else 0.1
        if random.random() < 0.3:
            clauses.append((random.choice(OTHER), "other", "neutral"))
        random.shuffle(clauses)
        text = " ".join(c[0] for c in clauses)
        rating = int(np.clip(round(score + random.gauss(0, 0.5)), 1, 5))
        reviews.append({"review_id": rid, "text": text, "rating": rating})
        assert len(split_clauses(text)) == len(clauses)
        for c, asp, sent in clauses:
            if random.random() < noise:  # simulated annotator disagreement
                asp = random.choice(ASPECTS + ["other"]); sent = random.choice(["negative", "neutral", "positive"])
            labelled.append({"review_id": rid, "text": c, "aspect": asp, "sentiment": sent})
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(reviews).to_csv(f"{out_dir}/reviews_synthetic.csv", index=False)
    pd.DataFrame(labelled).to_csv(f"{out_dir}/labelled_synthetic.csv", index=False)
    print(f"Wrote {n} synthetic reviews and {len(labelled)} labelled clauses to {out_dir}/")


if __name__ == "__main__":
    make()
