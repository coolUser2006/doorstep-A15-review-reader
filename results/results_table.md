Data: `data/processed/labelled_synthetic.csv` (2743 clauses, 1200 reviews). Macro-F1, 10 seeds, grouped split by review (25% test).

| Task | Method | Mean | SD | Min-Max |
|---|---|---|---|---|
| aspect | majority | 0.053 | 0.002 | 0.048-0.056 |
| aspect | keyword | 0.838 | 0.009 | 0.825-0.856 |
| aspect | tfidf_lr | 0.948 | 0.011 | 0.931-0.968 |
| sentiment | majority | 0.253 | 0.003 | 0.249-0.256 |
| sentiment | keyword | 0.618 | 0.011 | 0.599-0.633 |
| sentiment | tfidf_lr | 0.945 | 0.008 | 0.932-0.957 |
