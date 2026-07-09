"""Deterministic web indexing for local HTML files."""

from lake_agent.indexing.web.deterministic import (
    DeterministicWebParser,
    WebParseOptions,
)
from lake_agent.indexing.web.enrichment import (
    WebEnrichmentOptions,
    WebLLMEnricher,
)
from lake_agent.indexing.web.service import (
    WebIndexingError,
    WebIndexingProgress,
    WebIndexingService,
)
from lake_agent.indexing.web.vector_store import (
    add_web_result,
    add_web_results,
    build_batch_web_documents,
    build_openai_embeddings,
    build_pgvector_store,
    build_web_documents,
)

__all__ = [
    "DeterministicWebParser",
    "WebEnrichmentOptions",
    "WebIndexingError",
    "WebIndexingProgress",
    "WebIndexingService",
    "WebLLMEnricher",
    "WebParseOptions",
    "add_web_result",
    "add_web_results",
    "build_batch_web_documents",
    "build_openai_embeddings",
    "build_pgvector_store",
    "build_web_documents",
]
