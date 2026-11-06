
import json
from pathlib import Path
from types import SimpleNamespace

from src.catalog import ICDCatalog
from src.llm_reranker import LLMReranker
from src.note_normalizer import (
    NoteNormalizer,
    merge_retrieval_lists
)

ROOT = Path(__file__).resolve().parents[1]


def catalog():
    return ICDCatalog(
        ROOT / "data/mms-2026-01.json"
    )


def response(payload):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(
                    content=payload,
                    refusal=None
                )
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20
        )
    )


class SequenceClient:
    def __init__(self, values):
        self.values = iter(values)
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.create
            )
        )

    def create(self, **params):
        self.requests.append(params)
        result = next(self.values)

        if isinstance(result, Exception):
            raise result

        return result


def test_openai_strict_schema_candidate_enum(monkeypatch):
    monkeypatch.setenv("ICD_LLM_PROVIDER", "openai")
    monkeypatch.setenv("ICD_LLM_MODEL", "gpt-4.1-mini")

    client = SequenceClient([
        response('{"codes":["1A00"]}')
    ])

    model = LLMReranker(catalog(), client=client)

    assert model.rank(
        "cholera", ["1A00", "1F40.Z"]
    ) == ["1A00"]

    fmt = client.requests[0]["response_format"]

    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert (
        fmt["json_schema"]["schema"]["properties"]
        ["codes"]["items"]["enum"]
        == ["1A00", "1F40.Z"]
    )


def test_invalid_response_retried_once(monkeypatch, capsys):
    monkeypatch.setenv("ICD_LLM_PROVIDER", "openai")

    client = SequenceClient([
        response('{"wrong":[]}'),
        response('{"codes":["1A00"]}')
    ])

    model = LLMReranker(catalog(), client=client)

    assert model.rank(
        "PRIVATE PATIENT NOTE", ["1A00"]
    ) == ["1A00"]

    assert model.calls == 2
    assert model.retries == 1
    assert model.successes == 1
    assert (
        "PRIVATE PATIENT NOTE"
        not in capsys.readouterr().err
    )


def test_validation_failure_falls_back(monkeypatch, capsys):
    monkeypatch.setenv("ICD_LLM_PROVIDER", "openai")

    client = SequenceClient([
        response("{bad json"),
        response('{"codes":[]}')
    ])

    model = LLMReranker(catalog(), client=client)

    assert model.rank(
        "SECRET NOTE", ["1A00"]
    ) == []

    log = capsys.readouterr().err

    assert "validation:no_valid_candidate_codes" in log
    assert "SECRET NOTE" not in log
    assert model.failures == 1
    assert model.calls == 2


def test_transient_rate_limit_retry(monkeypatch):
    monkeypatch.setenv("ICD_LLM_PROVIDER", "openai")
    monkeypatch.setenv("ICD_LLM_BACKOFF_SECONDS", "0")

    class TemporaryError(Exception):
        status_code = 429

    client = SequenceClient([
        TemporaryError("sensitive details"),
        response('{"codes":["1A00"]}')
    ])

    model = LLMReranker(catalog(), client=client)

    assert model.rank(
        "cholera", ["1A00"]
    ) == ["1A00"]

    assert model.calls == 2
    assert model.retries == 1


def test_auth_error_disables_without_retry(monkeypatch, capsys):
    monkeypatch.setenv("ICD_LLM_PROVIDER", "openai")

    class BadKey(Exception):
        status_code = 401

    client = SequenceClient([
        BadKey("secret key")
    ])

    model = LLMReranker(catalog(), client=client)

    assert model.rank("note PHI", ["1A00"]) == []
    assert model.calls == 1
    assert not model.enabled

    assert "secret key" not in capsys.readouterr().err


def test_optional_normalization(monkeypatch):
    monkeypatch.setenv("ICD_NORMALIZE_MODE", "all")
    monkeypatch.setenv("ICD_LLM_PROVIDER", "openai")

    client = SequenceClient([
        response('{"query":"pneumonia"}')
    ])

    normalizer = NoteNormalizer(client=client)

    assert normalizer.normalize("neumonía") == "pneumonia"
    assert normalizer.calls == 1
    assert normalizer.successes == 1

    combined = merge_retrieval_lists(
        ["A", "B", "C", "D"],
        ["D", "X", "Y"],
        quota=2
    )

    assert combined == ["A", "B", "D", "X"]


def test_normalization_disabled(monkeypatch):
    monkeypatch.setenv("ICD_NORMALIZE_MODE", "off")

    normalizer = NoteNormalizer()

    assert normalizer.normalize("neumonía") == ""
    assert normalizer.calls == 0
