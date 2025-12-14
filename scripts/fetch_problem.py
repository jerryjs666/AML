import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fetch a problem from apps.jsonl for quick copy")
    parser.add_argument("--jsonl", default="data/reference_db/apps.jsonl")
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    path = Path(args.jsonl)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/build_reference_db.py first.")

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == args.idx:
                record = json.loads(line)
                payload = {
                    "question": record.get("question"),
                    "input_output": record.get("input_output"),
                    "solutions": record.get("solutions"),
                    "starter_code": record.get("starter_code"),
                    "difficulty": record.get("difficulty"),
                    "url": record.get("url"),
                    "problem_id": record.get("problem_id"),
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return
    raise IndexError(f"Index {args.idx} out of range")


if __name__ == "__main__":
    main()
