from __future__ import annotations

import warnings
from typing import Any

from langchain_core.documents import Document
from langchain_postgres.v2.indexes import HNSWIndex, IVFFlatIndex

from lake_agent.config import EmbeddingSettings, PostgresSettings
from lake_agent.domain.indexing_models import DocumentIndexResult, DocumentSection
from lake_agent.indexing.tabular.vector_store import (
    build_openai_embeddings,
    build_pgvector_store as build_base_pgvector_store,
)


def build_pgvector_store(
    table_name: str,
    *,
    embedding_settings: EmbeddingSettings | None = None,
    postgres_settings: PostgresSettings | None = None,
) -> Any:
    embedding_settings = embedding_settings or EmbeddingSettings.from_env()
    vector_store = build_base_pgvector_store(
        table_name,
        embedding_settings=embedding_settings,
        postgres_settings=postgres_settings,
    )
    # _ensure_approx_index(
    #     vector_store,
    #     table_name=table_name,
    #     embedding_dimensions=embedding_settings.dimensions,
    # )
    return vector_store


def build_document_documents(result: DocumentIndexResult) -> list[Document]:
    documents: list[Document] = []

    if result.file_search_text:
        documents.append(
            Document(
                page_content=result.file_search_text,
                metadata={
                    "record_type": "file",
                    "source_id": result.source_id,
                    "relative_path": result.relative_path,
                    "filename": result.filename,
                    "file_format": result.file_format,
                },
            )
        )

    for section in result.sections:
        page_content = _section_page_content(section)
        if not page_content:
            continue
        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "record_type": "section",
                    "source_id": result.source_id,
                    "relative_path": result.relative_path,
                    "filename": result.filename,
                    "file_format": result.file_format,
                    "section_id": section.section_id,
                    "section_type": section.section_type,
                    "chunk_index": section.chunk_index,
                    "heading": section.heading,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "image_id": section.image_id,
                    "image_index": section.image_index,
                },
            )
        )

    return documents


def build_batch_document_documents(results: list[DocumentIndexResult]) -> list[Document]:
    documents: list[Document] = []
    for result in results:
        documents.extend(build_document_documents(result))
    return documents


def add_document_result(vector_store: Any, result: DocumentIndexResult) -> list[str]:
    return add_document_results(vector_store, [result])


def add_document_results(vector_store: Any, results: list[DocumentIndexResult]) -> list[str]:
    documents = build_batch_document_documents(results)
    if not documents:
        return []

    ids = [_document_id(document) for document in documents]
    vector_store.add_documents(documents=documents, ids=ids)
    return ids


def _document_id(document: Document) -> str:
    metadata = document.metadata
    if metadata.get("record_type") == "file":
        return f"{metadata['source_id']}:file"
    return str(metadata["section_id"])


def _section_page_content(section: DocumentSection) -> str | None:
    return section.search_text.strip() if section.search_text else None


def _ensure_approx_index(
    vector_store: Any,
    *,
    table_name: str,
    embedding_dimensions: int | None,
) -> None:
    if embedding_dimensions is None:
        return

    if embedding_dimensions <= 2000:
        _ensure_hnsw_index(vector_store, table_name=table_name)
        return

    warnings.warn(
        "Embedding dimensions exceed pgvector HNSW limit for vector columns; "
        f"falling back to IVFFlat for {table_name!r}. "
        f"dimensions={embedding_dimensions}",
        RuntimeWarning,
        stacklevel=2,
    )
    _ensure_ivfflat_index(vector_store, table_name=table_name)


def _ensure_hnsw_index(vector_store: Any, *, table_name: str) -> None:
    index_name = f"{table_name}_hnsw_idx"
    if vector_store.is_valid_index(index_name):
        return
    vector_store.apply_vector_index(
        HNSWIndex(),
        name=index_name,
    )


def _ensure_ivfflat_index(vector_store: Any, *, table_name: str) -> None:
    index_name = f"{table_name}_ivfflat_idx"
    if vector_store.is_valid_index(index_name):
        return
    vector_store.apply_vector_index(
        IVFFlatIndex(),
        name=index_name,
    )
