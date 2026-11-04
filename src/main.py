"""Entrypoint: python -m src.main INPUT.jsonl OUTPUT.jsonl"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from .catalog import ICDCatalog
from .retrieval import LexicalRetriever

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = Path(os.environ.get("ICD_MMS_PATH", str(ROOT / "data" / "mms-2026-01.json")))


def run(input_path: str, output_path: str) -> int:
    catalog = ICDCatalog(CATALOG_PATH)
    model = LexicalRetriever(catalog)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen = set()
    with open(input_path, encoding="utf-8") as source, open(output_path, "w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            note_id = row["id"]
            if note_id in seen:
                raise ValueError(f"Repeated input id {note_id!r} at line {line_number}")
            seen.add(note_id)
            text = row.get("text", "")
            codes = catalog.sanitize(model.predict(text))
            target.write(json.dumps({"id": note_id, "codes": codes}, ensure_ascii=False) + "\n")
            target.flush()  # Progressive output in case the run is interrupted.
            count += 1
    print(f"Wrote {count} predictions to {output}", file=sys.stderr)
    return count


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m src.main INPUT.jsonl OUTPUT.jsonl", file=sys.stderr)
        raise SystemExit(2)
    run(sys.argv[1], sys.argv[2])

if __name__ == "__main__":
    main()
