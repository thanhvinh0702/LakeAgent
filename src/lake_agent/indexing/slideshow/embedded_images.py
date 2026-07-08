from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lake_agent.domain.indexing_models import (
    ImageIndexResult,
    SlideshowEmbeddedImage,
    SlideshowIndexResult,
    SlideshowSection,
)
from lake_agent.indexing.image import (
    DoclingOCRExtractionOptions,
    SupportsImageOCR,
    ImageVLMEnricher,
)
from lake_agent.indexing.text.chunking import build_basic_search_text


@dataclass(frozen=True, slots=True)
class SlideshowEmbeddedImageProcessingOptions:
    ocr_batch_size: int = DoclingOCRExtractionOptions().batch_size
    vlm_batch_size: int = 3


class SlideshowEmbeddedImageProcessor:
    def __init__(
        self,
        *,
        ocr_extractor: SupportsImageOCR | None = None,
        vlm_enricher: ImageVLMEnricher | None = None,
        options: SlideshowEmbeddedImageProcessingOptions | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._ocr_extractor = ocr_extractor
        self._vlm_enricher = vlm_enricher
        self._options = options or SlideshowEmbeddedImageProcessingOptions()
        self._log_callback = log_callback

    def enrich_batch(self, results: list[SlideshowIndexResult]) -> list[SlideshowIndexResult]:
        if not results:
            return results

        image_entries: list[tuple[SlideshowIndexResult, SlideshowEmbeddedImage]] = []
        for result in results:
            for image in result.embedded_images:
                image_entries.append((result, image))

        if not image_entries:
            self._log("No embedded images detected in the current slideshow batch.")
            return results

        if self._ocr_extractor is not None:
            try:
                self._apply_ocr_sections(image_entries)
            except Exception as exc:
                warning = f"OCR extraction failed: {exc}"
                self._log(warning)
                for result in results:
                    result.parse_warnings.append(warning)
        else:
            self._log("Skipping OCR for embedded slideshow images because OCR is disabled.")
        if self._vlm_enricher is not None:
            try:
                self._apply_vlm_sections(image_entries)
            except Exception as exc:
                warning = f"VLM enrichment failed: {exc}"
                self._log(warning)
                for result in results:
                    result.parse_warnings.append(warning)
        else:
            self._log("Skipping VLM summaries for embedded slideshow images because VLM is disabled.")
        return results

    def _apply_ocr_sections(
        self,
        image_entries: list[tuple[SlideshowIndexResult, SlideshowEmbeddedImage]],
    ) -> None:
        total_batches = (len(image_entries) + self._options.ocr_batch_size - 1) // self._options.ocr_batch_size
        for batch_start in range(0, len(image_entries), self._options.ocr_batch_size):
            batch = image_entries[batch_start : batch_start + self._options.ocr_batch_size]
            batch_number = (batch_start // self._options.ocr_batch_size) + 1
            image_paths = [entry.path for _, entry in batch]
            source_ids = [entry.image_id for _, entry in batch]
            self._log(
                "Sending OCR batch "
                f"{batch_number}/{total_batches} for {len(batch)} embedded slideshow images."
            )
            sections_by_image = self._ocr_extractor.extract_sections_batch(
                image_paths,
                source_ids=source_ids,
            )
            for result, image in batch:
                raw_sections = sections_by_image.get(image.image_id, [])
                base_index = len(result.sections)
                for offset, raw_section in enumerate(raw_sections, start=1):
                    result.sections.append(
                        SlideshowSection(
                            section_id=raw_section.section_id,
                            section_type="image_ocr",
                            chunk_index=base_index + offset,
                            heading=raw_section.heading or image.caption,
                            content=raw_section.content,
                            slide_start=image.slide_start,
                            slide_end=image.slide_end,
                            char_count=raw_section.char_count,
                            search_text=raw_section.search_text,
                            image_id=image.image_id,
                            image_index=image.image_index,
                            warnings=list(image.warnings) + list(raw_section.warnings),
                        )
                    )

    def _apply_vlm_sections(
        self,
        image_entries: list[tuple[SlideshowIndexResult, SlideshowEmbeddedImage]],
    ) -> None:
        total_batches = (len(image_entries) + self._options.vlm_batch_size - 1) // self._options.vlm_batch_size
        for batch_start in range(0, len(image_entries), self._options.vlm_batch_size):
            batch = image_entries[batch_start : batch_start + self._options.vlm_batch_size]
            batch_number = (batch_start // self._options.vlm_batch_size) + 1
            image_paths = [entry.path for _, entry in batch]
            image_results = [_build_image_index_result(result, image) for result, image in batch]
            self._log(
                "Sending VLM batch "
                f"{batch_number}/{total_batches} for {len(batch)} embedded slideshow images."
            )
            enriched_images = self._vlm_enricher.enrich_batch(image_paths, image_results)
            for (result, image), enriched in zip(batch, enriched_images, strict=True):
                if not enriched.file_summary:
                    continue
                result.sections.append(
                    SlideshowSection(
                        section_id=f"{image.image_id}:summary",
                        section_type="image_summary",
                        chunk_index=len(result.sections) + 1,
                        heading=image.caption or f"Image {image.image_index}",
                        content=enriched.file_summary,
                        slide_start=image.slide_start,
                        slide_end=image.slide_end,
                        char_count=len(enriched.file_summary),
                        search_text=_build_image_summary_search_text(enriched, image, result),
                        image_id=image.image_id,
                        image_index=image.image_index,
                        warnings=list(image.warnings),
                    )
                )

    def _log(self, message: str) -> None:
        if self._log_callback is not None:
            self._log_callback(message)


def _build_image_index_result(
    result: SlideshowIndexResult,
    image: SlideshowEmbeddedImage,
) -> ImageIndexResult:
    relative_path = f"{result.relative_path}#image-{image.image_index}"
    return ImageIndexResult(
        source_id=image.image_id,
        relative_path=relative_path,
        filename=image.filename,
        file_format="png",
        width=image.width,
        height=image.height,
        color_mode=image.color_mode,
        has_alpha="A" in image.color_mode,
        is_animated=False,
        frame_count=1,
        parse_warnings=list(image.warnings),
        file_search_text=None,
    )


def _build_image_summary_search_text(
    enriched: ImageIndexResult,
    image: SlideshowEmbeddedImage,
    result: SlideshowIndexResult,
) -> str:
    parts = [
        result.filename,
        result.relative_path,
        image.filename,
        image.caption,
        enriched.file_summary,
    ]
    if enriched.file_keywords:
        parts.append(", ".join(enriched.file_keywords))
    return build_basic_search_text(None, "\n".join(part for part in parts if part))
