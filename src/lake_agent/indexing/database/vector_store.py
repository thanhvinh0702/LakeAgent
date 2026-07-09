from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from lake_agent.domain.indexing_models import DatabaseIndexResult, DbTableProfile
from lake_agent.indexing.tabular.vector_store import (
    build_openai_embeddings,
    build_pgvector_store,
)


def build_database_documents(result: DatabaseIndexResult) -> list[Document]:
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

    for table in result.tables:
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
                },
            )
        )

    return documents


def build_batch_database_documents(results: list[DatabaseIndexResult]) -> list[Document]:
    documents: list[Document] = []
    for result in results:
        documents.extend(build_database_documents(result))
    return documents


def add_database_result(
    vector_store: Any,
    result: DatabaseIndexResult,
) -> list[str]:
    return add_database_results(vector_store, [result])


def add_database_results(
    vector_store: Any,
    results: list[DatabaseIndexResult],
) -> list[str]:
    documents = build_batch_database_documents(results)
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


def _table_page_content(table: DbTableProfile) -> str | None:
    return table.table_search_text.strip() if table.table_search_text else None
