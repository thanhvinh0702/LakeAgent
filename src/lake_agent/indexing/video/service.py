from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from lake_agent.domain.indexing_models import VideoIndexResult
from lake_agent.indexing.video.deterministic import DeterministicVideoParser
from lake_agent.indexing.video.vector_store import add_video_results
from lake_agent.persistence.repositories import VideoIndexRepository

_SUPPORTED_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


@dataclass(frozen=True, slots=True)
class IndexedVideoFile:
    relative_path: str
    size_bytes: int
    last_modified: datetime


@dataclass(frozen=True, slots=True)
class VideoIndexingProgress:
    event: str
    relative_path: str | None
    processed_count: int
    total_count: int
    indexed_count: int
    unchanged_count: int
    error_count: int
    vector_document_count: int
    message: str | None = None


@dataclass(frozen=True, slots=True)
class VideoIndexingError:
    relative_path: str
    message: str


class VideoIndexingService:
    def __init__(
        self,
        root_dir: str | Path,
        parser: DeterministicVideoParser,
        repository: VideoIndexRepository,
        *,
        vector_store: Any | None = None,
        vector_batch_size: int = 25,
        progress_callback: Callable[[VideoIndexingProgress], None] | None = None,
    ) -> None:
        if vector_batch_size <= 0:
            raise ValueError("vector_batch_size must be positive")
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._parser = parser
        self._repository = repository
        self._vector_store = vector_store
        self._vector_batch_size = vector_batch_size
        self._progress_callback = progress_callback

    def run(
        self,
        prefix: str = "",
        *,
        force: bool = False,
    ) -> dict[str, int | str | list[VideoIndexingError]]:
        normalized_prefix = _normalize_prefix(prefix)
        indexed_at = datetime.now(timezone.utc)
        indexed_count = 0
        unchanged_count = 0
        error_count = 0
        vector_document_count = 0
        processed_count = 0
        errors: list[VideoIndexingError] = []
        vector_batch: list[VideoIndexResult] = []
        files = self._scan_files(normalized_prefix)
        total_count = len(files)

        self._emit_progress(
            event="start",
            relative_path=None,
            processed_count=0,
            total_count=total_count,
            indexed_count=0,
            unchanged_count=0,
            error_count=0,
            vector_document_count=0,
        )

        for indexed_file in files:
            previous = self._repository.find_file(indexed_file.relative_path)
            if not force and _is_unchanged(previous, indexed_file):
                unchanged_count += 1
                processed_count += 1
                self._emit_progress(
                    event="unchanged",
                    relative_path=indexed_file.relative_path,
                    processed_count=processed_count,
                    total_count=total_count,
                    indexed_count=indexed_count,
                    unchanged_count=unchanged_count,
                    error_count=error_count,
                    vector_document_count=vector_document_count,
                )
                continue

            absolute_path = self._root_dir / indexed_file.relative_path
            file_format = absolute_path.suffix.lower().removeprefix(".")
            source_id = _stable_source_id(indexed_file.relative_path)
            try:
                result = self._parser.parse_file(
                    absolute_path,
                    relative_path=indexed_file.relative_path,
                    source_id=source_id,
                )
                self._repository.save(
                    result,
                    size_bytes=indexed_file.size_bytes,
                    last_modified=indexed_file.last_modified,
                    indexed_at=indexed_at,
                )
                vector_batch.append(result)
                indexed_count += 1
                if len(vector_batch) >= self._vector_batch_size:
                    vector_document_count += self._flush_vector_batch(vector_batch)
                    vector_batch.clear()
                processed_count += 1
                self._emit_progress(
                    event="indexed",
                    relative_path=indexed_file.relative_path,
                    processed_count=processed_count,
                    total_count=total_count,
                    indexed_count=indexed_count,
                    unchanged_count=unchanged_count,
                    error_count=error_count,
                    vector_document_count=vector_document_count,
                )
            except Exception as exc:
                error_message = str(exc)
                error_count += 1
                self._repository.save_error(
                    source_id=source_id,
                    relative_path=indexed_file.relative_path,
                    filename=absolute_path.name,
                    file_format=file_format,
                    size_bytes=indexed_file.size_bytes,
                    last_modified=indexed_file.last_modified,
                    error_message=error_message,
                    indexed_at=indexed_at,
                )
                errors.append(
                    VideoIndexingError(
                        relative_path=indexed_file.relative_path,
                        message=error_message,
                    )
                )
                processed_count += 1
                self._emit_progress(
                    event="error",
                    relative_path=indexed_file.relative_path,
                    processed_count=processed_count,
                    total_count=total_count,
                    indexed_count=indexed_count,
                    unchanged_count=unchanged_count,
                    error_count=error_count,
                    vector_document_count=vector_document_count,
                    message=error_message,
                )

        vector_document_count += self._flush_vector_batch(vector_batch)
        self._repository.mark_missing(normalized_prefix, indexed_at)
        self._emit_progress(
            event="done",
            relative_path=None,
            processed_count=processed_count,
            total_count=total_count,
            indexed_count=indexed_count,
            unchanged_count=unchanged_count,
            error_count=error_count,
            vector_document_count=vector_document_count,
        )
        return {
            "prefix": normalized_prefix,
            "discovered_count": len(files),
            "indexed_count": indexed_count,
            "unchanged_count": unchanged_count,
            "error_count": error_count,
            "vector_document_count": vector_document_count,
            "errors": errors,
        }

    def _scan_files(self, prefix: str) -> list[IndexedVideoFile]:
        base_dir = self._root_dir if not prefix else (self._root_dir / prefix).resolve()
        if not base_dir.exists():
            return []

        files: list[IndexedVideoFile] = []
        for path in sorted(base_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            stat = path.stat()
            files.append(
                IndexedVideoFile(
                    relative_path=path.relative_to(self._root_dir).as_posix(),
                    size_bytes=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )
        return files

    def _flush_vector_batch(self, batch: list[VideoIndexResult]) -> int:
        if self._vector_store is None or not batch:
            return 0
        document_ids = add_video_results(self._vector_store, batch)
        return len(document_ids)

    def _emit_progress(
        self,
        *,
        event: str,
        relative_path: str | None,
        processed_count: int,
        total_count: int,
        indexed_count: int,
        unchanged_count: int,
        error_count: int,
        vector_document_count: int,
        message: str | None = None,
    ) -> None:
        if self._progress_callback is None:
            return
        self._progress_callback(
            VideoIndexingProgress(
                event=event,
                relative_path=relative_path,
                processed_count=processed_count,
                total_count=total_count,
                indexed_count=indexed_count,
                unchanged_count=unchanged_count,
                error_count=error_count,
                vector_document_count=vector_document_count,
                message=message,
            )
        )


def _is_unchanged(previous: Any, current: IndexedVideoFile) -> bool:
    if not previous:
        return False
    if previous.get("status") != "indexed":
        return False
    if previous.get("size_bytes") != current.size_bytes:
        return False
    return previous.get("last_modified") == current.last_modified


def _normalize_prefix(prefix: str) -> str:
    cleaned = prefix.strip().replace("\\", "/").strip("/")
    if not cleaned:
        return ""
    normalized = PurePosixPath(cleaned).as_posix()
    if normalized in {".", ""}:
        return ""
    return normalized


def _stable_source_id(relative_path: str) -> str:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"source_{digest}"
