
"""Tests for the local cross-encoder reranker."""

import json
from pathlib import Path

import numpy as np

from src.catalog import ICDCatalog
from src.cross_encoder_reranker import LocalCrossEncoderReranker
from src.main import run

ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    """Simulates a cross-encoder without loading a real model."""

    def __init__(self, values):
        self.values = values
        self.pairs = []

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        return np.asarray(self.values, dtype=float)


def test_cross_encoder_can_change_candidate_order(monkeypatch):
    monkeypatch.setenv("ICD_CE_WEIGHT", "1")

    catalog = ICDCatalog(
        ROOT / "data/mms-2026-01.json"
    )

    fake = FakeModel([-2.0, 4.0, 1.0])

    cross = LocalCrossEncoderReranker(
        catalog,
        model=fake
    )

    order = cross.rank(
        "Diagnosis: malaria",
        ["1A00", "1F40.Z", "1A01"]
    )

    assert order == ["1F40.Z", "1A01", "1A00"]
    assert len(fake.pairs) == 3
    assert cross.reranked == 1


def test_cross_encoder_invalid_scores_fall_back(monkeypatch):
    monkeypatch.setenv("ICD_CE_WEIGHT", "1")

    catalog = ICDCatalog(
        ROOT / "data/mms-2026-01.json"
    )

    cross = LocalCrossEncoderReranker(
        catalog,
        model=FakeModel([float("nan"), 2])
    )

    original = ["1A00", "1F40.Z"]

    assert cross.rank(
        "some clinical note",
        original
    ) == original

    assert cross.failures == 1


def test_lexical_mode_ignores_cross_encoder_and_llm(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("ICD_RERANK_MODE", "lexical")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    notes = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"

    notes.write_text(
        '{"id":"x","text":"Dx: Cholera"}\n',
        encoding="utf-8"
    )

    assert run(str(notes), str(output)) == 1

    data = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert data["id"] == "x"
    assert len(data["codes"]) == 3


def test_cross_mode_uses_fake_cross_encoder(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("ICD_RERANK_MODE", "cross")

    import src.main as main_module

    class StubCross:
        candidate_limit = 3

        def __init__(self, catalog):
            pass

        def rank(self, note, candidates):
            return list(reversed(candidates))

        def report(self):
            return "stub cross"

    monkeypatch.setattr(
        main_module,
        "LocalCrossEncoderReranker",
        StubCross
    )

    notes = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"

    notes.write_text(
        '{"id":"x","text":"Dx: Cholera"}\n',
        encoding="utf-8"
    )

    assert main_module.run(str(notes), str(output)) == 1

    data = json.loads(
        output.read_text(encoding="utf-8")
    )

    expected = list(
        reversed(
            main_module.LexicalRetriever(
                ICDCatalog(
                    ROOT / "data/mms-2026-01.json"
                )
            ).rank("Dx: Cholera", limit=3)
        )
    )

    assert data["codes"] == expected
