from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lake_agent.config import OCRSettings
from lake_agent.domain.indexing_models import ImageSection
from lake_agent.indexing.text.chunking import (
    build_basic_search_text,
    chunk_markdown_text,
    normalize_text,
)


@dataclass(frozen=True, slots=True)
class OCRExtractionOptions:
    batch_size: int = 3
    max_chars_per_chunk: int = 2400
    min_chunk_chars: int = 400


class OCRMarkdownExtractor:
    def __init__(
        self,
        client: Any,
        handle_file: Any,
        options: OCRExtractionOptions | None = None,
    ) -> None:
        self._client = client
        self._handle_file = handle_file
        self._options = options or OCRExtractionOptions()

    @classmethod
    def from_env(
        cls,
        options: OCRExtractionOptions | None = None,
    ) -> "OCRMarkdownExtractor":
        settings = OCRSettings.from_env()
        client, handle_file = _load_gradio_client(settings.model_url)
        return cls(client=client, handle_file=handle_file, options=options)

    def extract_sections(self, image_path: str | Path, *, source_id: str) -> list[ImageSection]:
        markdown = self._invoke_ocr_batch([Path(image_path)])[0]
        normalized = normalize_text(markdown)
        if not normalized:
            return []

        chunks = chunk_markdown_text(
            normalized,
            max_chars=self._options.max_chars_per_chunk,
            min_chars=self._options.min_chunk_chars,
        )
        return [
            ImageSection(
                section_id=_stable_id(
                    f"{source_id}:ocr:{chunk.chunk_index}:{chunk.heading or ''}",
                    prefix="imgsec",
                ),
                section_type="ocr_chunk",
                chunk_index=chunk.chunk_index,
                heading=chunk.heading,
                content=chunk.content,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
                char_count=len(chunk.content),
                search_text=build_basic_search_text(chunk.heading, chunk.content),
            )
            for chunk in chunks
        ]

    def extract_sections_batch(
        self,
        image_paths: list[str | Path],
        *,
        source_ids: list[str],
    ) -> dict[str, list[ImageSection]]:
        if len(image_paths) != len(source_ids):
            raise ValueError("image_paths and source_ids must have the same length")
        if not image_paths:
            return {}

        markdown_results = self._invoke_ocr_batch([Path(path) for path in image_paths])
        sections_by_source: dict[str, list[ImageSection]] = {}
        for source_id, markdown in zip(source_ids, markdown_results, strict=True):
            normalized = normalize_text(markdown)
            if not normalized:
                sections_by_source[source_id] = []
                continue

            chunks = chunk_markdown_text(
                normalized,
                max_chars=self._options.max_chars_per_chunk,
                min_chars=self._options.min_chunk_chars,
            )
            sections_by_source[source_id] = [
                ImageSection(
                    section_id=_stable_id(
                        f"{source_id}:ocr:{chunk.chunk_index}:{chunk.heading or ''}",
                        prefix="imgsec",
                    ),
                    section_type="ocr_chunk",
                    chunk_index=chunk.chunk_index,
                    heading=chunk.heading,
                    content=chunk.content,
                    line_start=chunk.line_start,
                    line_end=chunk.line_end,
                    char_count=len(chunk.content),
                    search_text=build_basic_search_text(chunk.heading, chunk.content),
                )
                for chunk in chunks
            ]
        return sections_by_source

    @property
    def batch_size(self) -> int:
        return self._options.batch_size

    def _invoke_ocr_batch(self, image_paths: list[Path]) -> list[str]:
        result = self._client.predict(
            [self._handle_file(str(image_path)) for image_path in image_paths],
            api_name="/ocr",
        )
        return _coerce_ocr_batch_result(result, expected_count=len(image_paths))

_OCR_CLIENT_CACHE: dict[str, tuple[Any, Any]] = {}


def _load_gradio_client(model_url: str) -> tuple[Any, Any]:
    cached = _OCR_CLIENT_CACHE.get(model_url)
    if cached is not None:
        return cached
    try:
        from gradio_client import Client, handle_file
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "OCR extraction requires gradio_client. Install the project dependencies first."
        ) from exc
    loaded = (Client(model_url), handle_file)
    _OCR_CLIENT_CACHE[model_url] = loaded
    return loaded


def _coerce_ocr_batch_result(result: Any, *, expected_count: int) -> list[str]:
    if expected_count == 1:
        return [_extract_string(result).strip()]

    if isinstance(result, (list, tuple)) and len(result) == expected_count:
        return [_extract_string(item).strip() for item in result]

    raise RuntimeError(
        "OCR batch response shape did not match the number of requested images. "
        f"expected_count={expected_count}, response_type={type(result).__name__!r}, "
        f"response_preview={str(result)[:400]!r}"
    )


def _extract_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        preferred_keys = ("markdown", "text", "content", "output", "result")
        for key in preferred_keys:
            if key in value:
                extracted = _extract_string(value[key])
                if extracted:
                    return extracted
        parts = [_extract_string(item) for item in value.values()]
        return "\n".join(part for part in parts if part)
    if isinstance(value, (list, tuple)):
        parts = [_extract_string(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value) if value is not None else ""


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
