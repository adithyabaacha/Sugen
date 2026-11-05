"""Provider selection and offline fallback tests."""

import json

from pathlib import Path
from types import SimpleNamespace

from src.catalog import ICDCatalog
from src.llm_reranker import LLMReranker
from src.main import run

ROOT = Path(__file__).resolve().parents[1]

class FakeClient:

    def __init__(self, codes):
        self.codes = codes
        self.kwargs = None

        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **kwargs):
        self.kwargs = kwargs

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({
                            "codes": self.codes
                        })
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=300,
                completion_tokens=30
            ),
        )

def catalog():
    return ICDCatalog(
        ROOT / "data/mms-2026-01.json"
    )

def test_groq_uses_correct_model_and_valid_codes(
    monkeypatch,
):
    monkeypatch.setenv(
        "ICD_LLM_PROVIDER", "groq"
    )
    monkeypatch.delenv(
        "ICD_LLM_MODEL", raising=False
    )
    monkeypatch.setenv(
        "ICD_USE_LLM", "auto"
    )

    client = FakeClient([
        "FAKE",
        "1A00",
        "1A00",
        "1F40.Z",
    ])

    reranker = LLMReranker(
        catalog(),
        client=client
    )

    assert reranker.model == "openai/gpt-oss-20b"

    result = reranker.rank(
        "Cholera",
        ["1A00", "1F40.Z"]
    )

    assert result == ["1A00", "1F40.Z"]
    assert reranker.successes == 1

    assert (
        client.kwargs["extra_body"][
            "reasoning_effort"
        ] == "low"
    )

def test_openai_default_provider(monkeypatch):
    monkeypatch.delenv(
        "ICD_LLM_PROVIDER", raising=False
    )
    monkeypatch.delenv(
        "ICD_LLM_MODEL", raising=False
    )
    monkeypatch.setenv(
        "ICD_USE_LLM", "auto"
    )

    client = FakeClient(["1A00"])

    reranker = LLMReranker(
        catalog(),
        client=client
    )

    assert reranker.provider == "openai"
    assert reranker.model == "gpt-4.1-mini"

    assert reranker.rank(
        "cholera", ["1A00"]
    ) == ["1A00"]

    assert "extra_body" not in client.kwargs

def test_missing_openai_key_uses_fallback(monkeypatch):
    monkeypatch.setenv(
        "ICD_LLM_PROVIDER", "openai"
    )
    monkeypatch.setenv(
        "ICD_USE_LLM", "auto"
    )

    monkeypatch.delenv(
        "OPENAI_API_KEY", raising=False
    )

    monkeypatch.setenv(
        "GROQ_API_KEY", "unused-key"
    )

    reranker = LLMReranker(catalog())

    assert not reranker.enabled
    assert reranker.rank(
        "cholera", ["1A00"]
    ) == []
    assert reranker.calls == 0

def test_file_runner_without_api_key(
    monkeypatch,
    tmp_path
):
    monkeypatch.setenv(
        "ICD_RERANK_MODE", "llm"
    )
    monkeypatch.setenv(
        "ICD_LLM_PROVIDER", "openai"
    )

    monkeypatch.delenv(
        "OPENAI_API_KEY", raising=False
    )

    notes = tmp_path / "input.jsonl"
    output = tmp_path / "output.jsonl"

    notes.write_text(
        '{"id":"n1","text":"Diagnosis: cholera"}\n',
        encoding="utf-8"
    )

    assert run(
        str(notes),
        str(output)
    ) == 1

    result = json.loads(
        output.read_text(encoding="utf-8")
    )

    assert result["id"] == "n1"
    assert 1 <= len(result["codes"]) <= 3

    cat = catalog()
    assert all(
        cat.is_valid(code)
        for code in result["codes"]
    )
