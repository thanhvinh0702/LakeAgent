"""Deterministic document indexing using Docling hierarchical chunking."""

from lake_agent.indexing.document.deterministic import (
    DeterministicDocumentParser,
    DocumentParseOptions,
)
from lake_agent.indexing.document.embedded_images import (
    DocumentEmbeddedImageProcessingOptions,
    DocumentEmbeddedImageProcessor,
)
from lake_agent.indexing.document.enrichment import (
    DocumentEnrichmentOptions,
    DocumentLLMEnricher,
)
from lake_agent.indexing.document.service import (
    DocumentIndexingError,
    DocumentIndexingProgress,
    DocumentIndexingService,
)
from lake_agent.indexing.document.vector_store import (
    add_document_result,
    add_document_results,
    build_batch_document_documents,
    build_document_documents,
    build_openai_embeddings,
    build_pgvector_store,
)

__all__ = [
    "DeterministicDocumentParser",
    "DocumentEmbeddedImageProcessingOptions",
    "DocumentEmbeddedImageProcessor",
    "DocumentEnrichmentOptions",
    "DocumentIndexingError",
    "DocumentIndexingProgress",
    "DocumentIndexingService",
    "DocumentLLMEnricher",
    "DocumentParseOptions",
    "add_document_result",
    "add_document_results",
    "build_batch_document_documents",
    "build_document_documents",
    "build_openai_embeddings",
    "build_pgvector_store",
]
