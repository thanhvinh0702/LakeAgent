from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - lightweight fallback for deterministic tests
    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)

from lake_agent.domain.indexing_models.json import JsonIndexResult, JsonSection


def build_openai_embeddings(settings: Any | None = None) -> Any:
    from lake_agent.indexing.tabular.vector_store import build_openai_embeddings

    return build_openai_embeddings(settings)


def build_pgvector_store(
    table_name: str,
    *,
    embedding_settings: Any | None = None,
    postgres_settings: Any | None = None,
) -> Any:
    _ensure_windows_selector_event_loop_policy()

    from lake_agent.indexing.tabular.vector_store import build_pgvector_store

    return build_pgvector_store(
        table_name,
        embedding_settings=embedding_settings,
        postgres_settings=postgres_settings,
    )


def build_json_documents(result: JsonIndexResult) -> list[Document]:
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
                    "top_level_type": result.top_level_type,
                    "entry_count": result.entry_count,
                    "max_depth": result.max_depth,
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
                    "chunk_index": section.chunk_index,
                    "path_start": section.path_start,
                    "path_end": section.path_end,
                    "entry_count": section.entry_count,
                },
            )
        )

    return documents


def build_batch_json_documents(results: list[JsonIndexResult]) -> list[Document]:
    documents: list[Document] = []
    for result in results:
        documents.extend(build_json_documents(result))
    return documents


def add_json_result(vector_store: Any, result: JsonIndexResult) -> list[str]:
    return add_json_results(vector_store, [result])


def add_json_results(vector_store: Any, results: list[JsonIndexResult]) -> list[str]:
    documents = build_batch_json_documents(results)
    if not documents:
        return []

    ids = [_document_id(document) for document in documents]
    vector_store.add_documents(documents=documents, ids=ids)
    return ids


def _document_id(document: Document) -> str:
    metadata = document.metadata
    if metadata.get("record_type") == "file":
        return f"{metadata['source_id']}:file"
    return f"{metadata['source_id']}:section_{metadata['chunk_index']}"


def _section_page_content(section: JsonSection) -> str | None:
    return section.search_text.strip() if section.search_text else None


def _ensure_windows_selector_event_loop_policy() -> None:
    if sys.platform != "win32":
        return

    policy_factory = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_factory is None:
        return

    if isinstance(asyncio.get_event_loop_policy(), policy_factory):
        return

    asyncio.set_event_loop_policy(policy_factory())
