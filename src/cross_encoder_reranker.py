
"""Local CPU cross-encoder reranker. No runtime model downloads."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .clinical import get_diagnosis_focus

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "cross_encoder"


class LocalCrossEncoderReranker:
    def __init__(self, catalog: Any, model: Any = None):
        self.catalog = catalog
        self.model_path = Path(
            os.getenv("ICD_CE_MODEL_PATH", str(DEFAULT_MODEL_PATH))
        )
        self.candidate_limit = max(
            3, min(100, int(os.getenv("ICD_CE_CANDIDATES", "30")))
        )
        self.batch_size = max(
            1, min(64, int(os.getenv("ICD_CE_BATCH_SIZE", "8")))
        )
        self.max_length = max(
            64, min(512, int(os.getenv("ICD_CE_MAX_LENGTH", "256")))
        )
        self.max_query_chars = max(
            100, min(6000, int(os.getenv("ICD_CE_QUERY_CHARS", "1800")))
        )
        self.weight = min(
            1.0, max(0.0, float(os.getenv("ICD_CE_WEIGHT", "0.75")))
        )
        self.time_budget = max(
            0.0,
            float(os.getenv("ICD_CE_TIME_BUDGET_SECONDS", "2400"))
        )

        self._model = model
        self._load_attempted = model is not None
        self.started = time.monotonic()
        self.reranked = 0
        self.failures = 0
        self.disabled_reason = ""

    @property
    def enabled(self) -> bool:
        return not bool(self.disabled_reason)

    def _get_model(self):
        if self._model is not None:
            return self._model
        if self._load_attempted:
            return None

        self._load_attempted = True

        if (
            not self.model_path.is_dir()
            or not (self.model_path / "config.json").exists()
        ):
            self.disabled_reason = (
                f"model not found at {self.model_path}"
            )
            print(
                f"Cross-encoder unavailable: {self.disabled_reason}",
                file=sys.stderr
            )
            return None

        try:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            import torch
            from sentence_transformers import CrossEncoder

            torch.set_num_threads(
                max(
                    1,
                    min(
                        8,
                        int(os.getenv("ICD_CE_THREADS", "4"))
                    )
                )
            )

            self._model = CrossEncoder(
                str(self.model_path),
                device="cpu",
                max_length=self.max_length,
                local_files_only=True,
            )
            return self._model

        except Exception as exc:
            self.failures += 1
            self.disabled_reason = (
                f"could not load local model "
                f"({type(exc).__name__})"
            )
            print(
                f"Cross-encoder unavailable: {self.disabled_reason}",
                file=sys.stderr
            )
            return None

    def rank(self, note: str, candidates: list[str]) -> list[str]:
        """Rerank candidates, preserving lexical order on failure."""

        candidates = [
            c for c in candidates if self.catalog.is_valid(c)
        ]

        if (
            len(candidates) < 2
            or not isinstance(note, str)
            or not note.strip()
        ):
            return candidates

        if not self.enabled:
            return candidates

        if time.monotonic() - self.started > self.time_budget:
            self.disabled_reason = "cross-encoder time budget reached"
            print(
                f"Cross-encoder disabled: {self.disabled_reason}",
                file=sys.stderr
            )
            return candidates

        model = self._get_model()
        if model is None:
            return candidates

        head = candidates[:self.candidate_limit]
        tail = candidates[self.candidate_limit:]

        focus = get_diagnosis_focus(note)
        query = (
            f"Assessment: {focus}. Consultation: {note}"
            if focus else note
        )
        query = query[:self.max_query_chars]

        pairs = [
            (query, self.catalog.get(code)["title"])
            for code in head
        ]

        try:
            scores = np.asarray(
                model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ),
                dtype=float,
            ).reshape(-1)

            if (
                len(scores) != len(head)
                or not np.all(np.isfinite(scores))
            ):
                raise ValueError("invalid cross-encoder scores")

            ce_order = np.argsort(-scores, kind="stable")
            ce_rank = np.empty(len(head), dtype=int)
            ce_rank[ce_order] = np.arange(len(head))

            lex_rank = np.arange(len(head))

            # Fuse ranks, not raw scores from different models.
            combined = (
                self.weight * ce_rank
                + (1.0 - self.weight) * lex_rank
            )

            order = np.argsort(combined, kind="stable")

            self.reranked += 1

            return [head[int(i)] for i in order] + tail

        except Exception as exc:
            self.failures += 1
            print(
                f"Cross-encoder inference failed "
                f"({type(exc).__name__}); using lexical order",
                file=sys.stderr
            )

            if self.failures >= 3:
                self.disabled_reason = "3 cross-encoder failures"

            return candidates

    def report(self) -> str:
        status = (
            f"disabled ({self.disabled_reason})"
            if self.disabled_reason else "available"
        )

        return (
            f"Cross-encoder {status}; "
            f"reranked={self.reranked}, "
            f"failures={self.failures}, "
            f"candidates={self.candidate_limit}, "
            f"weight={self.weight:.2f}"
        )
