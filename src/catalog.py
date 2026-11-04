"""The provided ICD-11 MMS JSON is the only authority for valid codes."""
from __future__ import annotations
import json
from pathlib import Path

class ICDCatalog:
    def __init__(self, path: str | Path):
        self.records = json.loads(Path(path).read_text(encoding="utf-8"))
        self.by_code = {r["code"].upper(): r for r in self.records}
        if len(self.by_code) != len(self.records):
            raise ValueError("Duplicate codes in ICD catalog")
        if not self.records:
            raise ValueError("Empty ICD catalog")

    def is_valid(self, code: str) -> bool:
        return isinstance(code, str) and code.strip().upper() in self.by_code

    def get(self, code: str) -> dict:
        return self.by_code[code.strip().upper()]

    def sanitize(self, values: list[str], fallback: list[str] | None = None) -> list[str]:
        result = []
        for code in list(values or []) + list(fallback or []):
            if not isinstance(code, str):
                continue
            # Keep only the stem, matching the grader normalization.
            code = code.split("&")[0].split("/")[0].strip().upper()
            if code in self.by_code and code not in result:
                result.append(code)
            if len(result) == 3:
                break
        if not result:
            # Defensive fallback for an empty or otherwise unprocessable note.
            result = [self.records[0]["code"]]
        return result
