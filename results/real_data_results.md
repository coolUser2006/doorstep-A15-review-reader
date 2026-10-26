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

## Limits (read before quoting any number)
1. Keyword aspects and VADER sentiment are unvalidated. Fill the `aspect_correct` / `sentiment_correct` columns in `validation_sample.csv` (y/n) and run
   `python src/score_validation.py` to get accuracy with a confidence interval.
2. Negative share per aspect is inflated for words like "problem" and "issue"; the odds ratios are more trustworthy than raw negative shares.
3. This is an association, not causation, and it describes owner reviews of cars (Edmunds), not a car-wash service.
4. The sample is capped per brand, so it is not representative of brand volume. `brand_aspect_negative_rate.csv` is descriptive only (min 20 mentions).
