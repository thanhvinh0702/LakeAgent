from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from lake_agent.config import EmbeddingSettings, PostgresSettings
from lake_agent.domain.indexing_models import TableProfile, TabularIndexResult


def build_openai_embeddings(
    settings: EmbeddingSettings | None = None,
) -> OpenAIEmbeddings:
    settings = settings or EmbeddingSettings.from_env()
    return OpenAIEmbeddings(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
    )


def build_pgvector_store(
    collection_name: str,
    *,
    embedding_settings: EmbeddingSettings | None = None,
    postgres_settings: PostgresSettings | None = None,
) -> Any:
    postgres_settings = postgres_settings or PostgresSettings.from_env()
    embeddings = build_openai_embeddings(embedding_settings)
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=postgres_settings.dsn_vector,
        use_jsonb=True,
    )


def build_tabular_documents(result: TabularIndexResult) -> list[Document]:
    documents: list[Document] = []

    file_page_content = _file_page_content(result)
    if file_page_content:
        documents.append(
            Document(
                page_content=file_page_content,
                metadata={
                    "record_type": "file",
                    "source_id": result.source_id,
                    "relative_path": result.relative_path,
                    "filename": result.filename,
                    "file_format": result.file_format,
                },
            )
        )

    for table in result.tables:
        if table.is_context_sheet:
            continue

        page_content = _table_page_content(table)
        if not page_content:
            continue

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "record_type": "table",
                    "source_id": result.source_id,
                    "relative_path": result.relative_path,
                    "filename": result.filename,
                    "file_format": result.file_format,
                    "table_id": table.table_id,
                    "table_name": table.table_name,
                    "sheet_name": table.sheet_name,
                    "sheet_description": table.sheet_description,
                    "header_row_index": table.header_row_index,
                },
            )
        )

    return documents


def add_tabular_result(
    vector_store: Any,
    result: TabularIndexResult,
) -> list[str]:
    documents = build_tabular_documents(result)
    if not documents:
        return []

    ids = [_document_id(document) for document in documents]
    vector_store.add_documents(documents=documents, ids=ids)
    return ids


def _document_id(document: Document) -> str:
    metadata = document.metadata

    if metadata.get("record_type") == "file":
        return f"{metadata['source_id']}:file"

    return str(metadata["table_id"])


def _file_page_content(result: TabularIndexResult) -> str | None:
    parts: list[str] = []

    if result.file_summary:
        parts.append(result.file_summary)

    if result.file_keywords:
        parts.append(", ".join(result.file_keywords))

    return "\n\n".join(part for part in parts if part).strip() or None


def _table_page_content(table: TableProfile) -> str | None:
    return table.table_search_text.strip() if table.table_search_text else None