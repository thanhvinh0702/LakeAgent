from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from sqlalchemy.exc import ProgrammingError

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
        dimensions=settings.dimensions,
    )


def build_pgvector_store(
    table_name: str,
    *,
    embedding_settings: EmbeddingSettings | None = None,
    postgres_settings: PostgresSettings | None = None,
) -> Any:
    postgres_settings = postgres_settings or PostgresSettings.from_env()
    embedding_settings = embedding_settings or EmbeddingSettings.from_env()
    vector_size = _vector_size(embedding_settings)
    embeddings = build_openai_embeddings(embedding_settings)
    engine = PGEngine.from_connection_string(postgres_settings.dsn_vector)
    _ensure_vector_table(
        engine=engine,
        table_name=table_name,
        vector_size=vector_size,
    )
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=table_name,
        id_column="langchain_id",
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


def build_batch_tabular_documents(results: list[TabularIndexResult]) -> list[Document]:
    documents: list[Document] = []
    for result in results:
        documents.extend(build_tabular_documents(result))
    return documents


def add_tabular_result(
    vector_store: Any,
    result: TabularIndexResult,
) -> list[str]:
    return add_tabular_results(vector_store, [result])


def add_tabular_results(
    vector_store: Any,
    results: list[TabularIndexResult],
) -> list[str]:
    documents = build_batch_tabular_documents(results)
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


def _vector_size(settings: EmbeddingSettings) -> int:
    if settings.dimensions is None:
        raise ValueError(
            "Missing required environment variable: "
            "OPENAI_EMBEDDING_DIMENSIONS or EMBEDDING_DIMENSIONS. "
            "PGVectorStore needs a fixed embedding dimension to create the table."
        )
    return settings.dimensions


def _ensure_vector_table(
    *,
    engine: PGEngine,
    table_name: str,
    vector_size: int,
) -> None:
    try:
        engine.init_vectorstore_table(
            table_name=table_name,
            vector_size=vector_size,
            id_column=Column("langchain_id", "TEXT", nullable=False),
        )
    except ProgrammingError as exc:
        message = str(exc).lower()
        if "already exists" not in message and "duplicatetable" not in message:
            raise
