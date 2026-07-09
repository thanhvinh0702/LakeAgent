from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from lake_agent.domain.indexing_models import ImageSection
from lake_agent.indexing.text.chunking import (
    build_basic_search_text,
    chunk_markdown_text,
    normalize_text,
)

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter


class SupportsImageOCR(Protocol):
    def extract_sections_batch(
        self,
        image_paths: list[str | Path],
        *,
        source_ids: list[str],
    ) -> dict[str, list[ImageSection]]: ...


@dataclass(frozen=True, slots=True)
class DoclingOCRExtractionOptions:
    batch_size: int = 3
    max_chars_per_chunk: int = 2400
    min_chunk_chars: int = 400


class DoclingOCRMarkdownExtractor:
    def __init__(
        self,
        converter: DocumentConverter,
        options: DoclingOCRExtractionOptions | None = None,
    ) -> None:
        self._converter = converter
        self._options = options or DoclingOCRExtractionOptions()

    @classmethod
    def from_default(
        cls,
        options: DoclingOCRExtractionOptions | None = None,
    ) -> "DoclingOCRMarkdownExtractor":
        return cls(converter=_load_docling_converter(), options=options)

    def extract_sections(self, image_path: str | Path, *, source_id: str) -> list[ImageSection]:
        return self.extract_sections_batch([image_path], source_ids=[source_id]).get(source_id, [])

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

        from docling_core.types.doc import ImageRefMode

        sections_by_source: dict[str, list[ImageSection]] = {}
        results = list(
            self._converter.convert_all(
                [str(Path(path)) for path in image_paths],
                raises_on_error=True,
            )
        )
        if len(results) != len(source_ids):
            raise RuntimeError(
                "Docling OCR response count did not match the number of requested images. "
                f"expected_count={len(source_ids)}, actual_count={len(results)}"
            )

        for source_id, result in zip(source_ids, results, strict=True):
            markdown = result.document.export_to_markdown(
                image_mode=ImageRefMode.PLACEHOLDER,
                traverse_pictures=True,
            )
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


_DOCLING_CONVERTER_CACHE: DocumentConverter | None = None


def _load_docling_converter() -> DocumentConverter:
    global _DOCLING_CONVERTER_CACHE
    if _DOCLING_CONVERTER_CACHE is not None:
        return _DOCLING_CONVERTER_CACHE

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Docling OCR extraction requires docling. Install the project dependencies first."
        ) from exc

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
    )
    _DOCLING_CONVERTER_CACHE = DocumentConverter(
        format_options={
            InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
        }
    )
    return _DOCLING_CONVERTER_CACHE


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
