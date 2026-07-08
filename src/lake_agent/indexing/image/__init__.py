"""Minimal deterministic indexing for image files."""

from lake_agent.indexing.image.deterministic import (
    DeterministicImageParser,
    ImageParseOptions,
)
from lake_agent.indexing.image.docling_ocr import (
    DoclingOCRExtractionOptions,
    DoclingOCRMarkdownExtractor,
    SupportsImageOCR,
)
from lake_agent.indexing.image.ocr import (
    OCRExtractionOptions,
    OCRMarkdownExtractor,
)
from lake_agent.indexing.image.vlm import (
    ImageEnrichmentOptions,
    ImageVLMEnricher,
)
from lake_agent.indexing.image.vector_store import (
    add_image_result,
    add_image_results,
    build_batch_image_documents,
    build_image_documents,
    build_openai_embeddings,
    build_pgvector_store,
)
from lake_agent.indexing.image.service import (
    ImageIndexingError,
    ImageIndexingProgress,
    ImageIndexingService,
)

__all__ = [
    "DeterministicImageParser",
    "DoclingOCRExtractionOptions",
    "DoclingOCRMarkdownExtractor",
    "ImageIndexingError",
    "ImageIndexingProgress",
    "ImageIndexingService",
    "ImageEnrichmentOptions",
    "ImageVLMEnricher",
    "ImageParseOptions",
    "OCRExtractionOptions",
    "OCRMarkdownExtractor",
    "SupportsImageOCR",
    "add_image_result",
    "add_image_results",
    "build_batch_image_documents",
    "build_image_documents",
    "build_openai_embeddings",
    "build_pgvector_store",
]
