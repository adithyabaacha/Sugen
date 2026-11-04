import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gold", default="data/dev_notes.jsonl")
parser.add_argument("--out", default="in/notes.jsonl")
args = parser.parse_args()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.gold, encoding="utf-8") as src, open(args.out, "w", encoding="utf-8") as dest:
    count = 0
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        dest.write(json.dumps({"id": row["id"], "text": row["text"]}, ensure_ascii=False) + "\n")
        count += 1
print(f"Prepared {count} unlabeled notes -> {args.out}")
