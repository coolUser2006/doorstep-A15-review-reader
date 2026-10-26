# Integration note - A15

## What I produce
`results/review_aspects.csv`: review, aspect, sentiment, review_id, rating. Regenerated on demand (not streaming).

## What I consume
Nothing from other modules. Public service-business review datasets, hand-labelled by the team.

## Assumptions that could break
1. A consumer (e.g. A9 churn warning) may expect one row per review; I emit one row per (review, aspect).
2. `review` is text, not an ID; `review_id` is appended at the end.
3. Aspect names are fixed to five; adding an aspect adds new values to the column.

## Tested against
TODO (Week 9): run against another team's real file. Synthetic data only so far.

## Known incompatibilities
TODO (Week 14).
