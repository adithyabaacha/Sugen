
import json
from pathlib import Path
from types import SimpleNamespace

from src.catalog import ICDCatalog
from src.llm_reranker import LLMReranker
from src.main import run

ROOT = Path(__file__).resolve().parents[1]


def fake_response(codes):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({"codes": codes})
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=250,
            completion_tokens=25
        ),
    )


class FakeClient:
    def __init__(self, output):
        self.output = output
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


def test_llm_reranker_filters_hallucinated_and_duplicate_codes(
    monkeypatch,
):
    monkeypatch.setenv("ICD_USE_LLM", "auto")
    catalog = ICDCatalog(ROOT / "data/mms-2026-01.json")

    model = LLMReranker(
        catalog,
        client=FakeClient(
            fake_response([
                "FAKE",
                "1F40.Z",
                "1F40.Z",
                "1A00"
            ])
        ),
    )

    assert model.rank(
        "malaria diagnosed",
        ["1F40.Z", "1A00"]
    ) == ["1F40.Z", "1A00"]

    assert model.successes == 1
    assert model.estimated_cost > 0


def test_llm_failure_falls_back_and_trips_breaker(
    monkeypatch,
):
    monkeypatch.setenv("ICD_USE_LLM", "auto")
    catalog = ICDCatalog(ROOT / "data/mms-2026-01.json")

    model = LLMReranker(
        catalog,
        client=FakeClient(
            RuntimeError("provider unavailable")
        ),
    )

    for _ in range(3):
        assert model.rank("cholera", ["1A00"]) == []

    assert not model.enabled
    assert model.calls == 3

    assert model.rank("cholera", ["1A00"]) == []
    assert model.calls == 3


def test_with_api_disabled_file_runner_still_works(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("ICD_USE_LLM", "0")
    monkeypatch.setenv(
        "OPENAI_API_KEY", "unused-key"
    )

    notes = tmp_path / "notes.jsonl"
    output = tmp_path / "out.jsonl"

    notes.write_text(
        '{"id":"one","text":"Diagnosis: cholera"}\n',
        encoding="utf-8",
    )

    assert run(str(notes), str(output)) == 1

    result = json.loads(
        output.read_text().strip()
    )

    assert result["id"] == "one"
    assert len(result["codes"]) == 3

    catalog = ICDCatalog(
        ROOT / "data/mms-2026-01.json"
    )
    assert all(
        catalog.is_valid(c)
        for c in result["codes"]
    )
