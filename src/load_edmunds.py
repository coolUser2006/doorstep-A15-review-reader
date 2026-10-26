"""Load the Kaggle 'Scraped Car Review' (Edmunds owner reviews) zip/folder into data/raw/reviews_cars.csv.
Columns written: review_id, text, rating, brand. Samples up to --per-brand reviews per brand (seed 42)."""
import argparse, glob, io, os, random, re, sys, zipfile
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED); np.random.seed(SEED)


def read_one(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)), engine="python", on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError("cannot decode file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="path to the archive .zip or a folder of Scraped_Car_Review_*.csv")
    ap.add_argument("--out", default="data/raw/reviews_cars.csv")
    ap.add_argument("--per-brand", type=int, default=160)
    a = ap.parse_args()
    files = {}
    if a.src.endswith(".zip"):
        with zipfile.ZipFile(a.src) as z:
            for n in z.namelist():
                if n.endswith(".csv"):
                    files[n] = z.read(n)
    else:
        for n in glob.glob(os.path.join(a.src, "*.csv")):
            files[n] = open(n, "rb").read()
    parts = []
    for name, raw in sorted(files.items()):
        d = read_one(raw)
        if "Review" not in d.columns or "Rating" not in d.columns:
            continue
        brand = re.sub(r"^scra\w*_car_reviews?_|\.csv$", "", os.path.basename(name).lower())
        d = pd.DataFrame({"text": d["Review"].astype(str).str.strip(),
                          "rating": pd.to_numeric(d["Rating"], errors="coerce"), "brand": brand})
        d = d[(d.text.str.len() >= 40) & d.rating.between(1, 5)]
        parts.append(d.sample(min(len(d), a.per_brand), random_state=SEED))
    df = pd.concat(parts, ignore_index=True).drop_duplicates("text").reset_index(drop=True)
    df.insert(0, "review_id", [f"RV{i:06d}" for i in range(len(df))])
    df["rating"] = df["rating"].astype(int)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} reviews from {df.brand.nunique()} brands -> {a.out}")
    print("rating distribution:\n", df.rating.value_counts(normalize=True).sort_index().round(3).to_string())


if __name__ == "__main__":
    main()
