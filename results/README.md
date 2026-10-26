# results/ - what A15 hands over

## review_aspects.csv  (the handover)
| Column | Meaning |
|---|---|
| review | Full text of the customer review (UTF-8) |
| aspect | One of: punctuality, wash_quality, staff_behaviour, price, booking_experience |
| sentiment | negative, neutral or positive for that aspect in that review. If a review has several clauses on one aspect, any negative clause makes it negative |
| review_id | Review identifier, format RV00042 (appended column) |
| rating | Star rating 1-5 given with the review; empty cell = not provided (appended column) |

One row per (review, aspect). A review mentioning three aspects gives three rows. Reviews that mention no aspect produce no rows.

## How to regenerate
```
python src/evaluate.py --reviews data/processed/reviews.csv --labelled data/processed/labelled.csv
```

## Other files
- `results_table.md` - baseline vs model, 10 seeds (`python src/train.py`)
- `driver_analysis.csv`, `plots/driver_odds.png` - which aspect drives bad ratings
- `failure_cases.csv` - every hand-labelled clause the model got wrong (out-of-fold)

## What it assumes
1. Aspects are detected per clause, so a clause that covers two aspects is forced into one.
2. Sentiment is about the aspect, not the whole review ("great wash, arrived late" gives wash_quality positive, punctuality negative).
3. Reviews are English. Other languages or heavy code-mixing will degrade accuracy.
4. A "bad rating" in the driver analysis means 2 stars or fewer.
5. The `review` column holds text, not an ID. If another team expects an ID there, use `review_id`.
