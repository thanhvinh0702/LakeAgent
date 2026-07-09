from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from lake_agent.domain.indexing_models import EpubIndexResult, EpubSection
from lake_agent.indexing.tabular.vector_store import (
    build_openai_embeddings,
    build_pgvector_store,
)


def build_epub_documents(result: EpubIndexResult) -> list[Document]:
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
                    "title": result.title,
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
                    "chapter_index": section.chapter_index,
                    "chapter_title": section.chapter_title,
                    "chapter_href": section.chapter_href,
                    "image_id": section.image_id,
                    "image_index": section.image_index,
                    "image_href": section.image_href,
                },
            )
        )

    return documents


def build_batch_epub_documents(results: list[EpubIndexResult]) -> list[Document]:
    documents: list[Document] = []
    for result in results:
        documents.extend(build_epub_documents(result))
    return documents


def add_epub_result(vector_store: Any, result: EpubIndexResult) -> list[str]:
    return add_epub_results(vector_store, [result])


def add_epub_results(vector_store: Any, results: list[EpubIndexResult]) -> list[str]:
    documents = build_batch_epub_documents(results)
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


def _section_page_content(section: EpubSection) -> str | None:
    return section.search_text.strip() if section.search_text else None
