"""Deterministic slideshow indexing using Docling hierarchical chunking."""

from lake_agent.indexing.slideshow.deterministic import (
    DeterministicSlideshowParser,
    SlideshowParseOptions,
)
from lake_agent.indexing.slideshow.embedded_images import (
    SlideshowEmbeddedImageProcessingOptions,
    SlideshowEmbeddedImageProcessor,
)
from lake_agent.indexing.slideshow.enrichment import (
    SlideshowEnrichmentOptions,
    SlideshowLLMEnricher,
)
from lake_agent.indexing.slideshow.service import (
    SlideshowIndexingError,
    SlideshowIndexingProgress,
    SlideshowIndexingService,
)
from lake_agent.indexing.slideshow.vector_store import (
    add_slideshow_result,
    add_slideshow_results,
    build_batch_slideshow_documents,
    build_openai_embeddings,
    build_pgvector_store,
    build_slideshow_documents,
)

__all__ = [
    "DeterministicSlideshowParser",
    "SlideshowEmbeddedImageProcessingOptions",
    "SlideshowEmbeddedImageProcessor",
    "SlideshowEnrichmentOptions",
    "SlideshowIndexingError",
    "SlideshowIndexingProgress",
    "SlideshowIndexingService",
    "SlideshowLLMEnricher",
    "SlideshowParseOptions",
    "add_slideshow_result",
    "add_slideshow_results",
    "build_batch_slideshow_documents",
    "build_openai_embeddings",
    "build_pgvector_store",
    "build_slideshow_documents",
]
