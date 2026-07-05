"""Deterministic text indexing for plain text and markdown files."""

from lake_agent.indexing.text.deterministic import (
    DeterministicTextParser,
    TextParseOptions,
)
from lake_agent.indexing.text.enrichment import (
    TextEnrichmentOptions,
    TextLLMEnricher,
)
from lake_agent.indexing.text.service import (
    TextIndexingError,
    TextIndexingProgress,
    TextIndexingService,
)
from lake_agent.indexing.text.vector_store import (
    add_text_result,
    add_text_results,
    build_batch_text_documents,
    build_openai_embeddings,
    build_pgvector_store,
    build_text_documents,
)

__all__ = [
    "DeterministicTextParser",
    "TextEnrichmentOptions",
    "TextIndexingError",
    "TextIndexingProgress",
    "TextIndexingService",
    "TextLLMEnricher",
    "TextParseOptions",
    "add_text_result",
    "add_text_results",
    "build_batch_text_documents",
    "build_openai_embeddings",
    "build_pgvector_store",
    "build_text_documents",
]
