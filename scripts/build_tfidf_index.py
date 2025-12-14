import argparse
import json
import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def load_records(jsonl_path: Path) -> List[dict]:
    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_documents(records: List[dict]) -> List[str]:
    docs = []
    for rec in records:
        text_parts = [rec.get("question", "") or ""]
        starter = rec.get("starter_code")
        if starter:
            text_parts.append(starter)
        docs.append("\n".join(text_parts))
    return docs


def main():
    parser = argparse.ArgumentParser(description="Build TF-IDF index for APPS reference DB")
    parser.add_argument("--jsonl", default="data/reference_db/apps.jsonl")
    parser.add_argument("--artifact_dir", default="data/reference_db")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing {jsonl_path}. Run scripts/build_reference_db.py first.")

    records = load_records(jsonl_path)
    docs = build_documents(records)
    logging.info("Building TF-IDF on %d documents", len(docs))

    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=50000)
    char_vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, max_features=50000)

    word_matrix = word_vectorizer.fit_transform(docs)
    char_matrix = char_vectorizer.fit_transform(docs)
    combined_matrix = sparse.hstack([word_matrix, char_matrix]).tocsr()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(artifact_dir / "tfidf_matrix.npz", combined_matrix)
    joblib.dump(word_vectorizer, artifact_dir / "tfidf_vectorizer_words.joblib")
    joblib.dump(char_vectorizer, artifact_dir / "tfidf_vectorizer_chars.joblib")

    stats = {
        "num_docs": len(docs),
        "vocab_words": len(word_vectorizer.vocabulary_),
        "vocab_chars": len(char_vectorizer.vocabulary_),
    }
    logging.info("Saved TF-IDF artifacts to %s", artifact_dir)
    logging.info("Stats: %s", stats)


if __name__ == "__main__":
    main()
