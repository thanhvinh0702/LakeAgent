from __future__ import annotations

import unittest

from lake_agent.domain.indexing_models import (
    DocumentIndexResult,
    DocumentSection,
    SlideshowIndexResult,
    SlideshowSection,
    TextIndexResult,
    TextSection,
)
from lake_agent.indexing.document.enrichment import (
    DocumentEnrichmentOptions,
    DocumentLLMEnricher,
)
from lake_agent.indexing.slideshow.enrichment import (
    SlideshowEnrichmentOptions,
    SlideshowLLMEnricher,
)
from lake_agent.indexing.text.enrichment import (
    TextEnrichmentOptions,
    TextLLMEnricher,
)


def _dummy_invoke(system_prompt: str, user_prompt: str):
    raise AssertionError("should not be called in payload sampling tests")


class SummarySamplingTest(unittest.TestCase):
    def test_document_payload_samples_head_middle_tail(self) -> None:
        result = DocumentIndexResult(
            source_id="source_doc",
            relative_path="docs/report.pdf",
            filename="report.pdf",
            file_format="pdf",
            sections=[
                DocumentSection(
                    section_id=f"section_{index}",
                    section_type="document_chunk",
                    chunk_index=index,
                    content=f"content {index}",
                )
                for index in range(1, 10)
            ],
        )
        enricher = DocumentLLMEnricher(
            invoke_enrichment=_dummy_invoke,
            options=DocumentEnrichmentOptions(section_count_limit=4),
        )

        payload = enricher._build_payload(result)

        self.assertEqual([1, 3, 6, 9], [section["chunk_index"] for section in payload["sections"]])

    def test_slideshow_payload_samples_head_middle_tail(self) -> None:
        result = SlideshowIndexResult(
            source_id="source_slide",
            relative_path="slides/deck.pptx",
            filename="deck.pptx",
            file_format="pptx",
            sections=[
                SlideshowSection(
                    section_id=f"section_{index}",
                    section_type="slide_chunk",
                    chunk_index=index,
                    content=f"content {index}",
                )
                for index in range(1, 10)
            ],
        )
        enricher = SlideshowLLMEnricher(
            invoke_enrichment=_dummy_invoke,
            options=SlideshowEnrichmentOptions(section_count_limit=4),
        )

        payload = enricher._build_payload(result)

        self.assertEqual([1, 3, 6, 9], [section["chunk_index"] for section in payload["sections"]])

    def test_text_payload_samples_head_middle_tail(self) -> None:
        result = TextIndexResult(
            source_id="source_text",
            relative_path="notes/report.md",
            filename="report.md",
            file_format="md",
            sections=[
                TextSection(
                    section_id=f"section_{index}",
                    chunk_index=index,
                    content=f"content {index}",
                )
                for index in range(1, 10)
            ],
        )
        enricher = TextLLMEnricher(
            invoke_enrichment=_dummy_invoke,
            options=TextEnrichmentOptions(section_count_limit=4),
        )

        payload = enricher._build_payload(result)

        self.assertEqual([1, 3, 6, 9], [section["chunk_index"] for section in payload["sections"]])


if __name__ == "__main__":
    unittest.main()
