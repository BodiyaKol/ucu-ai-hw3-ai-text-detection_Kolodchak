"""Validate a submission CSV against the full sample order (sample ids are taken from the TF-IDF submission,
which was written via sample_submission.csv on Kaggle)."""
import sys

import pandas as pd

from src.submission import check_submission

sub = pd.read_csv(sys.argv[1])
ref = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "outputs/03_tfidf/m2_tfidf/submission.csv")
sub = ref[["id"]].merge(sub, on="id", how="left")
check_submission(sub, ref)
sub.to_csv(sys.argv[1], index=False)
print(f"OK {sys.argv[1]}: rows={len(sub)} share={sub['label'].value_counts(normalize=True).round(3).to_dict()}")
