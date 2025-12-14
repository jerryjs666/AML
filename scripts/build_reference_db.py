import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from datasets import get_dataset_config_info, load_dataset

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

REQUIRED_FIELDS = ["question", "input_output", "solutions", "difficulty", "url", "starter_code", "problem_id"]


def validate_schema(features: Dict[str, Any]):
    logging.info("Dataset features: %s", features)
    missing = [f for f in REQUIRED_FIELDS if f not in features]
    if missing:
        raise RuntimeError(f"Dataset missing required fields: {missing}")


def normalize_record(example: Dict[str, Any]) -> Dict[str, Any]:
    record = {key: example.get(key) for key in REQUIRED_FIELDS}
    return record


def main():
    parser = argparse.ArgumentParser(description="Build APPS reference database JSONL")
    parser.add_argument("--split", default="test", help="Dataset split to use (test/train/validation)")
    parser.add_argument("--limit", type=int, default=500, help="Number of records to include")
    parser.add_argument("--out", default="data/reference_db/apps.jsonl", help="Output JSONL path")
    parser.add_argument("--allow-offline", action="store_true", help="Use bundled offline sample if download fails")
    parser.add_argument("--offline-jsonl", default="sample_data/offline_apps.jsonl", help="Path to offline fallback data")
    args = parser.parse_args()

    # Validate schema from HF metadata even if download is blocked
    info = get_dataset_config_info("codeparrot/apps", trust_remote_code=True)
    validate_schema(info.features)

    dataset = None
    try:
        logging.info("Loading dataset split=%s", args.split)
        dataset = load_dataset("codeparrot/apps", split=args.split)
    except Exception as exc:  # noqa: BLE001
        if not args.allow_offline:
            raise
        logging.warning("Falling back to offline sample due to download error: %s", exc)

    records: List[Dict[str, Any]] = []
    if dataset is not None:
        subset = dataset.select(range(min(args.limit, len(dataset))))
        for example in subset:
            records.append(normalize_record(example))
    else:
        offline_path = Path(args.offline_jsonl)
        if not offline_path.exists():
            raise FileNotFoundError(f"Offline file missing: {offline_path}")
        with offline_path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(records) >= args.limit:
                    break
                records.append(json.loads(line))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")
            count += 1
    logging.info("Wrote %d records to %s", count, out_path)


if __name__ == "__main__":
    main()
