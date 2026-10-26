"""Compare baselines vs TF-IDF+LR over 10 seeds (grouped by review). Logs to experiments.csv."""
import argparse, datetime, os, random, sys
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_labelled
from model import keyword_aspect, keyword_sentiment, make_tfidf_lr

SEED = 42
random.seed(SEED); np.random.seed(SEED)
TASKS = {"aspect": keyword_aspect, "sentiment": keyword_sentiment}


def evaluate_split(tr, te, seed):
    out = {}
    for task, kw in TASKS.items():
        y_tr, y_te = tr[task], te[task]
        maj = DummyClassifier(strategy="most_frequent").fit(tr.text, y_tr)
        lr = make_tfidf_lr(seed).fit(tr.text, y_tr)
        preds = {"majority": maj.predict(te.text), "keyword": te.text.map(kw).values, "tfidf_lr": lr.predict(te.text)}
        for name, p in preds.items():
            out[(task, name)] = f1_score(y_te, p, average="macro", zero_division=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default="data/processed/labelled.csv")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--who", default="unset", help="name to record in experiments.csv")
    a = ap.parse_args()
    if not os.path.exists(a.labelled):
        sys.exit(f"{a.labelled} not found. Hand-label real reviews (see README), or for a dry run use "
                 f"--labelled data/processed/labelled_synthetic.csv after python src/make_synthetic.py")
    df = load_labelled(a.labelled)
    scores = {}
    for s in range(a.seeds):
        tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=s).split(df, groups=df.review_id))
        for k, v in evaluate_split(df.iloc[tr_i], df.iloc[te_i], s).items():
            scores.setdefault(k, []).append(v)

    os.makedirs("results", exist_ok=True)
    log_path = "experiments.csv"
    old = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()
    n0, today, rows = len(old), datetime.date.today().isoformat(), []
    for (task, name), vals in scores.items():
        for s, v in enumerate(vals):
            rows.append({"run_id": f"R{n0 + len(rows) + 1:03d}", "date": today, "who": a.who,
                         "what_changed": f"{name} on {os.path.basename(a.labelled)}", "main_metric": f"macroF1_{task}",
                         "value": round(v, 4), "seed": s, "notes": ""})
    pd.concat([old, pd.DataFrame(rows)]).to_csv(log_path, index=False)

    lines = [f"Data: `{a.labelled}` ({len(df)} clauses, {df.review_id.nunique()} reviews). Macro-F1, {a.seeds} seeds, "
             "grouped split by review (25% test).", "", "| Task | Method | Mean | SD | Min-Max |", "|---|---|---|---|---|"]
    for (task, name), v in scores.items():
        lines.append(f"| {task} | {name} | {np.mean(v):.3f} | {np.std(v, ddof=1):.3f} | {min(v):.3f}-{max(v):.3f} |")
    open("results/results_table.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
