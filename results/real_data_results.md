# Real-data results: Edmunds car-owner reviews

**Data:** Kaggle "Scraped Car Review" (Edmunds owner reviews, 50 brands). Sample used: 6,770 reviews (up to 160 per brand, seed 42), 51,307 clauses.
**Method:** each clause gets aspects by keyword match and a sentiment from VADER with a car/service domain lexicon. No hand labels, so aspect and
sentiment accuracy are **not measured yet** (see `validation_sample.csv`). 55% of clauses mention none of the 7 aspects.

## Which aspect drives a bad (<=2 star) rating?
Base rate of bad ratings: 9.6%. Odds ratio of a bad rating when the aspect is negative in a review (logistic regression, all aspects together; 95% bootstrap CI over reviews):

| Aspect | Reviews negative | Mean rating if negative | Mean rating otherwise | Odds ratio (95% CI) |
|---|---|---|---|---|
| reliability_repairs | 1577 | 3.32 | 4.14 | 3.61 (2.95-4.48) |
| performance_handling | 1094 | 3.35 | 4.06 | 2.68 (2.26-3.26) |
| dealer_service | 496 | 3.04 | 4.01 | 2.22 (1.74-2.86) |
| comfort_interior | 601 | 3.48 | 3.98 | 1.81 (1.40-2.33) |
| price_value | 450 | 3.41 | 3.98 | 1.75 (1.32-2.33) |
| fuel_economy | 499 | 3.57 | 3.97 | 1.21 (0.91-1.69) |
| tech_features | 164 | 3.56 | 3.95 | 1.03 (0.61-1.58) |

**Reading it:** reliability and repair complaints are the strongest signal. Negative dealer/service comments pull the average rating down the most
(3.04), but only about 500 reviews mention them. Fuel economy and tech features are not distinguishable from "no effect" at this sample size.

## Validation (150 random clauses)
Labelled by Claude (an AI), **not by human annotators**; the team should spot-check a subset before relying on these numbers. A label counts as correct when
the predicted aspect is genuinely discussed in the clause, and when the predicted sentiment matches the clause's tone.

| Task | Correct | 95% CI |
|---|---|---|
| Aspect | 80.7% (121/150) | 74.3% to 87.0% |
| Sentiment | 73.3% (110/150) | 66.3% to 80.4% |

Where it fails: predicted-neutral sentiment was right only 47% of the time (many complaints such as "ride too rough" or "wish mileage was better" are
labelled neutral or even positive), while predicted-negative was right 92%. So the model **under-detects negativity**, which may bias the odds ratios above (direction not tested). Aspect errors come from generic keywords ("drive", "gas", "dealer", "deal").

## Limits (read before quoting any number)
1. Validation above is AI-labelled with one labeller, so there is no inter-annotator agreement. Run `python src/score_validation.py` again after
   human review.
2. Negative share per aspect is inflated for words like "problem" and "issue"; the odds ratios are more trustworthy than raw negative shares.
3. This is an association, not causation, and it describes owner reviews of cars (Edmunds), not a car-wash service.
4. The sample is capped per brand, so it is not representative of brand volume. `brand_aspect_negative_rate.csv` is descriptive only (min 20 mentions).
