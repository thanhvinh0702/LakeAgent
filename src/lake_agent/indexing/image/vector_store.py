from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from lake_agent.domain.indexing_models import ImageIndexResult, ImageSection
from lake_agent.indexing.tabular.vector_store import (
    build_openai_embeddings,
    build_pgvector_store,
)


def build_image_documents(result: ImageIndexResult) -> list[Document]:
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
                    "width": result.width,
                    "height": result.height,
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
                    "line_start": section.line_start,
                    "line_end": section.line_end,
                },
            )
        )

    return documents


def build_batch_image_documents(results: list[ImageIndexResult]) -> list[Document]:
    documents: list[Document] = []
    for result in results:
        documents.extend(build_image_documents(result))
    return documents


def add_image_result(vector_store: Any, result: ImageIndexResult) -> list[str]:
    return add_image_results(vector_store, [result])


def add_image_results(vector_store: Any, results: list[ImageIndexResult]) -> list[str]:
    documents = build_batch_image_documents(results)
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


def _section_page_content(section: ImageSection) -> str | None:
    return section.search_text.strip() if section.search_text else None
