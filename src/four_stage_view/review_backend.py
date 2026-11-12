"""Schemas and the bounded call contract shared by the Codex and VLM transports (step 1, video analysis).

Concrete backends only implement the call transport; procedural Codex geometry stays in ``codex_backend``.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

from pydantic import Field

from .models import StrictModel
from .store import atomic_json


class AgentCallError(RuntimeError):
    """A failed or uncertain call is never retried implicitly."""


T = TypeVar("T", bound=StrictModel)


class ObjectProfile(StrictModel):
    """Model-observed object attributes for reference-conditioned generation."""

    category: str = Field(min_length=1, max_length=300)
    shape: str = Field(min_length=1, max_length=1000)
    proportions: str = Field(min_length=1, max_length=600)
    colors: list[str] = Field(max_length=12)
    material_appearance: str = Field(min_length=1, max_length=600)
    visible_parts: list[str] = Field(max_length=24)
    markings: list[str] = Field(max_length=24)
    observed_views: list[str] = Field(max_length=12)
    unobserved_or_occluded: list[str] = Field(max_length=16)


def redact(text: str) -> str:
    for key, value in os.environ.items():
        if len(value) > 7 and any(
            word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"\b(?:sk-|AIza)[A-Za-z0-9_-]+", "[REDACTED]", text)
    return re.sub(r"(?i)(bearer\s+)[A-Za-z0-9_.~+/-]{8,}", r"\1[REDACTED]", text)


def strict_schema(model: type[T]) -> dict:
    """Strict JSON output schemas require every declared property to be required."""
    schema = model.model_json_schema()

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "object":
                value["additionalProperties"] = False
                value["required"] = list(value.get("properties", {}))
            value.pop("default", None)
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(schema)
    return schema


def reject_response(metadata: dict, message: str):
    metadata.update(status="invalid_response", error=message)
    atomic_json(Path(metadata["call_directory"]) / "metadata.json", metadata)
    raise AgentCallError(message)


class ReviewBackend(ABC):
    """Transport-independent decisions with the same local quality gates."""

    @abstractmethod
    def _call(
        self,
        model: type[T],
        prompt: str,
        images: list[Path],
        directory: Path,
        purpose: str,
        output_schema: dict | None = None,
    ) -> tuple[T, dict]:
        """Return a validated model response and its persisted call metadata."""
        raise NotImplementedError

    def _bound_review_call(self, model, prompt, images, directory, purpose, evidence_digest):
        if not re.fullmatch(r"[a-f0-9]{64}", evidence_digest):
            raise ValueError("Evidence digest must be a SHA-256 hex string")
        schema = strict_schema(model)
        schema["properties"]["evidence_digest"]["enum"] = [evidence_digest]
        result, metadata = self._call(model, prompt, images, directory, purpose, output_schema=schema)
        if result.evidence_digest != evidence_digest:
            reject_response(metadata, "Agent review referred to different evidence")
        return result, metadata
