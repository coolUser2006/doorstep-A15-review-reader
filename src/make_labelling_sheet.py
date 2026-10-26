"""Turn raw reviews into a clause-level sheet to hand-label. Fill the aspect and sentiment columns.
aspect in: punctuality|wash_quality|staff_behaviour|price|booking_experience|other
sentiment in: negative|neutral|positive"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import explode_clauses, load_reviews

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python src/make_labelling_sheet.py reviews.csv out_sheet.csv [n_reviews]")
    rv = load_reviews(sys.argv[1])
    if len(sys.argv) > 3:
        rv = rv.sample(int(sys.argv[3]), random_state=42)
    sheet = explode_clauses(rv)[["review_id", "text"]]
    sheet["aspect"], sheet["sentiment"] = "", ""
    sheet.to_csv(sys.argv[2], index=False)
    print(f"{len(sheet)} clauses from {len(rv)} reviews -> {sys.argv[2]}")
