"""Fine-tune a small transformer for aspect OR sentiment. NOT RUN in the authoring sandbox (no access to
the Hugging Face hub there) - run it on your own machine/Colab, then log results in experiments.csv.
Usage: python src/transformer_finetune.py --task aspect --seed 0   (repeat for 10 seeds, like train.py)"""
import argparse, os, sys
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_labelled

SEED = 42


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, y): self.enc, self.y = enc, y
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.y[i]); return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labelled", default="data/processed/labelled.csv")
    ap.add_argument("--task", choices=["aspect", "sentiment"], default="aspect")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    df = load_labelled(a.labelled)
    classes = sorted(df[a.task].unique()); idx = {c: i for i, c in enumerate(classes)}
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.25, random_state=a.seed).split(df, groups=df.review_id))  # same split as train.py
    tr, te = df.iloc[tr_i], df.iloc[te_i]
    tok = AutoTokenizer.from_pretrained(a.model)
    enc = lambda d: tok(list(d.text), truncation=True, padding=True, max_length=64)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=len(classes))
    args = TrainingArguments(output_dir=f"models/{a.task}_{a.seed}", num_train_epochs=a.epochs, seed=a.seed,
                             per_device_train_batch_size=16, learning_rate=3e-5, report_to=[], save_strategy="no")
    trainer = Trainer(model=model, args=args, train_dataset=DS(enc(tr), [idx[c] for c in tr[a.task]]))
    trainer.train()
    pred = trainer.predict(DS(enc(te), [idx[c] for c in te[a.task]])).predictions.argmax(1)
    print(f"transformer {a.task} seed={a.seed} macroF1 = {f1_score([idx[c] for c in te[a.task]], pred, average='macro'):.4f}")


if __name__ == "__main__":
    main()
