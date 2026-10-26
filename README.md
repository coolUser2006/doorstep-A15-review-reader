# A15 - Review reader (doorstep-A15-review-reader)

## What this module does

Reads customer reviews of a mobile car-wash service and reports which part of the service each one is about
(punctuality, wash quality, staff behaviour, price, booking experience) and whether the feeling is positive,
neutral or negative. It also answers the business question: which aspect most strongly drives a bad rating?

## Team

## &#x20;  ## Team

## &#x20;  - Himani Gupta (Lead): ...

## &#x20;  - Jai: ...Data

TODO: source of the public reviews, size, cleaning steps, and the hand-labelled count (target: several hundred
clauses, plus 100 reviews labelled by two people for kappa).
`src/make\_synthetic.py` generates SYNTHETIC reviews only so the pipeline can run before real labels exist.

## How to run

```
pip install -r requirements.txt
python src/make\_labelling\_sheet.py data/raw/reviews.csv data/processed/to\_label.csv 300   # hand-label this
python src/agreement.py annotator\_a.csv annotator\_b.csv                                   # kappa on shared 100
python src/train.py --who YourName          # baselines vs model, 10 seeds, logs experiments.csv
python src/evaluate.py                      # writes results/review\_aspects.csv and driver analysis
pytest
```

Dry run without real data: `python src/make\_synthetic.py`, then pass the `\*\_synthetic.csv` files via
`--labelled` / `--reviews`.

## Results (SYNTHETIC dry run - replace with real-data numbers)

Macro-F1, 10 seeds, split grouped by review.

|Method|Aspect|Sentiment|
|-|-|-|
|Majority class|0.053 +/- 0.002|0.253 +/- 0.003|
|Keyword baseline (+VADER)|0.838 +/- 0.009|0.618 +/- 0.011|
|TF-IDF + logistic regression|0.948 +/- 0.011|0.945 +/- 0.008|
|Fine-tuned transformer|not run yet (`src/transformer\_finetune.py`)|not run yet|

Synthetic data is templated, so these scores are far higher than real reviews will give. The pipeline check
that did matter: the driver analysis recovered the planted order (punctuality > wash quality > staff > price > booking).

## Output

`results/review\_aspects.csv` - see `results/README.md`.

## Limits

Clause-level, single-aspect-per-clause; English only; keyword baseline misses paraphrases; synthetic data only so far.

