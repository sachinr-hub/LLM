import argparse
import os
import csv
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional: BERTScore (install if you want richer evaluation)
try:
    from bert_score import score as bertscore_score  # type: ignore
    HAS_BERTSCORE = True
except Exception:
    HAS_BERTSCORE = False


def load_pairs(csv_path: str) -> Tuple[List[str], List[str]]:
    """Load pairs of (prediction, reference) from CSV with headers: pred, ref."""
    preds, refs = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds.append((row.get("pred") or "").strip())
            refs.append((row.get("ref") or "").strip())
    return preds, refs


def tfidf_cosine(preds: List[str], refs: List[str]) -> float:
    """Compute average TF-IDF cosine similarity across pairs."""
    scores = []
    for p, r in zip(preds, refs):
        vect = TfidfVectorizer(stop_words="english")
        X = vect.fit_transform([p, r])
        score = cosine_similarity(X[0], X[1])[0][0]
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


def bertscore(preds: List[str], refs: List[str]) -> Tuple[float, float, float]:
    if not HAS_BERTSCORE:
        raise RuntimeError("BERTScore is not installed. Run: pip install bert-score")
    P, R, F1 = bertscore_score(preds, refs, lang="en", rescale_with_baseline=True)
    return float(P.mean()), float(R.mean()), float(F1.mean())


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated emails vs references.")
    parser.add_argument("--pairs_csv", required=True, help="CSV with columns: pred, ref")
    parser.add_argument("--use_bertscore", action="store_true", help="Use BERTScore if available")
    args = parser.parse_args()

    preds, refs = load_pairs(args.pairs_csv)

    if args.use_bertscore and HAS_BERTSCORE:
        p, r, f1 = bertscore(preds, refs)
        print(f"BERTScore -> P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}")
    else:
        cs = tfidf_cosine(preds, refs)
        print(f"TF-IDF Cosine Similarity (avg): {cs:.4f}")


if __name__ == "__main__":
    main()
