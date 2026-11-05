
"""Entrypoint: python -m src.main INPUT.jsonl OUTPUT.jsonl"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .catalog import ICDCatalog
from .cross_encoder_reranker import LocalCrossEncoderReranker
from .llm_reranker import LLMReranker
from .retrieval import LexicalRetriever

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = Path(
    os.getenv(
        "ICD_MMS_PATH",
        str(ROOT / "data" / "mms-2026-01.json")
    )
)

MODES = {"lexical", "cross", "llm", "cross_llm"}


def run(input_path: str, output_path: str) -> int:
    mode = os.getenv(
        "ICD_RERANK_MODE", "lexical"
    ).strip().lower()

    if mode not in MODES:
        raise ValueError(
            f"Invalid ICD_RERANK_MODE={mode!r}; "
            f"choose one of {sorted(MODES)}"
        )

    catalog = ICDCatalog(CATALOG_PATH)
    retriever = LexicalRetriever(catalog)

    cross = (
        LocalCrossEncoderReranker(catalog)
        if mode in {"cross", "cross_llm"} else None
    )

    llm = (
        LLMReranker(catalog)
        if mode in {"llm", "cross_llm"} else None
    )

    limit = max(
        3,
        cross.candidate_limit if cross else 3,
        llm.candidate_limit if llm else 3,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    seen = set()

    with open(input_path, encoding="utf-8") as source, \
         open(output, "w", encoding="utf-8") as target:

        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue

            row = json.loads(line)
            note_id = row["id"]

            if note_id in seen:
                raise ValueError(
                    f"Repeated input id {note_id!r} "
                    f"at line {line_number}"
                )

            seen.add(note_id)
            note = row.get("text", "")

            candidates = retriever.rank(
                note,
                limit=limit
            )

            ranked = (
                cross.rank(note, candidates)
                if cross else candidates
            )

            llm_codes = (
                llm.rank(note, ranked)
                if llm else []
            )

            codes = catalog.sanitize(
                llm_codes,
                fallback=ranked
            )

            target.write(
                json.dumps(
                    {"id": note_id, "codes": codes},
                    ensure_ascii=False
                ) + "\n"
            )
            target.flush()
            count += 1

    print(
        f"Wrote {count} predictions to {output}; mode={mode}",
        file=sys.stderr
    )

    if cross:
        print(cross.report(), file=sys.stderr)

    if llm:
        print(llm.report(), file=sys.stderr)

    return count


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python -m src.main INPUT.jsonl OUTPUT.jsonl",
            file=sys.stderr
        )
        raise SystemExit(2)

    run(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
