from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any

from lake_agent.domain.indexing_models import (
    DocumentEmbeddedImage,
    DocumentIndexResult,
    DocumentSection,
)


def serialize_document_result(
    result: DocumentIndexResult,
    *,
    size_bytes: int,
    last_modified: datetime | None,
    indexed_at: datetime,
) -> dict[str, Any]:
    payload = asdict(result)
    payload["artifact_dir"] = None
    return {
        "record_type": "document_result",
        "indexed_at": indexed_at.isoformat(),
        "size_bytes": size_bytes,
        "last_modified": last_modified.isoformat() if last_modified is not None else None,
        "result": payload,
    }


def serialize_document_error(
    *,
    source_id: str,
    relative_path: str,
    filename: str,
    file_format: str,
    size_bytes: int,
    last_modified: datetime | None,
    indexed_at: datetime,
    error_message: str,
) -> dict[str, Any]:
    return {
        "record_type": "document_error",
        "indexed_at": indexed_at.isoformat(),
        "source_id": source_id,
        "relative_path": relative_path,
        "filename": filename,
        "file_format": file_format,
        "size_bytes": size_bytes,
        "last_modified": last_modified.isoformat() if last_modified is not None else None,
        "error_message": error_message,
    }


def deserialize_document_result(payload: dict[str, Any]) -> tuple[DocumentIndexResult, int, datetime | None, datetime]:
    result_payload = dict(payload["result"])
    result = DocumentIndexResult(
        source_id=result_payload["source_id"],
        relative_path=result_payload["relative_path"],
        filename=result_payload["filename"],
        file_format=result_payload["file_format"],
        sections=[
            _deserialize_section(section_payload)
            for section_payload in result_payload.get("sections", [])
        ],
        parser_version=result_payload.get("parser_version", "docling_hierarchical_v1"),
        parse_warnings=list(result_payload.get("parse_warnings", [])),
        file_summary=result_payload.get("file_summary"),
        file_keywords=list(result_payload.get("file_keywords", [])),
        file_search_text=result_payload.get("file_search_text"),
        embedded_images=[
            _deserialize_embedded_image(image_payload)
            for image_payload in result_payload.get("embedded_images", [])
        ],
        artifact_dir=None,
    )
    return (
        result,
        int(payload["size_bytes"]),
        _parse_datetime(payload.get("last_modified")),
        _parse_datetime(payload["indexed_at"]) or datetime.now(),
    )


def _deserialize_section(payload: dict[str, Any]) -> DocumentSection:
    return DocumentSection(
        section_id=payload["section_id"],
        section_type=payload["section_type"],
        chunk_index=int(payload["chunk_index"]),
        heading=payload.get("heading"),
        content=payload.get("content", ""),
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        char_count=int(payload.get("char_count", 0)),
        search_text=payload.get("search_text"),
        image_id=payload.get("image_id"),
        image_index=payload.get("image_index"),
        warnings=list(payload.get("warnings", [])),
    )


def _deserialize_embedded_image(payload: dict[str, Any]) -> DocumentEmbeddedImage:
    return DocumentEmbeddedImage(
        image_id=payload["image_id"],
        image_index=int(payload["image_index"]),
        path=payload.get("path", ""),
        filename=payload["filename"],
        width=int(payload["width"]),
        height=int(payload["height"]),
        color_mode=payload["color_mode"],
        page_start=payload.get("page_start"),
        page_end=payload.get("page_end"),
        caption=payload.get("caption"),
        warnings=list(payload.get("warnings", [])),
    )


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)
