from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from lake_agent.domain.indexing_models import DatabaseIndexResult
from lake_agent.indexing.database.deterministic import DeterministicDatabaseParser
from lake_agent.indexing.database.enrichment import DatabaseLLMEnricher
from lake_agent.indexing.database.vector_store import add_database_results
from lake_agent.persistence.repositories import DatabaseIndexRepository

_SUPPORTED_SUFFIXES = {".db", ".sqlite", ".sqlite3"}


@dataclass(frozen=True, slots=True)
class IndexedDatabaseFile:
    relative_path: str
    size_bytes: int
    last_modified: datetime


@dataclass(frozen=True, slots=True)
class DatabaseIndexingProgress:
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
class DatabaseIndexingError:
    relative_path: str
    message: str


class DatabaseIndexingService:
    def __init__(
        self,
        root_dir: str | Path,
        parser: DeterministicDatabaseParser,
        repository: DatabaseIndexRepository,
        *,
        enricher: DatabaseLLMEnricher | None = None,
        vector_store: Any | None = None,
        enrich_batch_size: int = 10,
        vector_batch_size: int = 25,
        progress_callback: Callable[[DatabaseIndexingProgress], None] | None = None,
    ) -> None:
        if enrich_batch_size <= 0:
            raise ValueError("enrich_batch_size must be positive")
        if vector_batch_size <= 0:
            raise ValueError("vector_batch_size must be positive")
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._parser = parser
        self._repository = repository
        self._enricher = enricher
        self._vector_store = vector_store
        self._enrich_batch_size = enrich_batch_size
        self._vector_batch_size = vector_batch_size
        self._progress_callback = progress_callback

    def run(self, prefix: str = "") -> dict[str, int | str | list[DatabaseIndexingError]]:
        normalized_prefix = _normalize_prefix(prefix)
        indexed_at = datetime.now(timezone.utc)
        discovered_count = 0
        indexed_count = 0
        unchanged_count = 0
        error_count = 0
        vector_document_count = 0
        errors: list[DatabaseIndexingError] = []
        enrich_pending: list[tuple[IndexedDatabaseFile, DatabaseIndexResult]] = []
        vector_batch: list[DatabaseIndexResult] = []
        files = self._scan_files(normalized_prefix)
        total_count = len(files)
        processed_count = 0

        self._emit_progress(
            event="start",
            relative_path=None,
            processed_count=processed_count,
            total_count=total_count,
            indexed_count=indexed_count,
            unchanged_count=unchanged_count,
            error_count=error_count,
            vector_document_count=vector_document_count,
        )

        for indexed_file in files:
            discovered_count += 1
            previous = self._repository.find_file(indexed_file.relative_path)
            if _is_unchanged(previous, indexed_file):
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
                enrich_pending.append((indexed_file, result))
                if self._enricher is None or len(enrich_pending) >= self._enrich_batch_size:
                    try:
                        (
                            indexed_count,
                            processed_count,
                            vector_document_count,
                        ) = self._flush_enrich_batch(
                            enrich_pending,
                            vector_batch=vector_batch,
                            indexed_at=indexed_at,
                            total_count=total_count,
                            indexed_count=indexed_count,
                            unchanged_count=unchanged_count,
                            error_count=error_count,
                            vector_document_count=vector_document_count,
                            processed_count=processed_count,
                        )
                    except Exception as exc:
                        error_count, processed_count = self._handle_pending_error_batch(
                            enrich_pending,
                            error_message=str(exc),
                            indexed_at=indexed_at,
                            total_count=total_count,
                            indexed_count=indexed_count,
                            unchanged_count=unchanged_count,
                            error_count=error_count,
                            vector_document_count=vector_document_count,
                            processed_count=processed_count,
                            errors=errors,
                        )
                    enrich_pending.clear()
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
                    DatabaseIndexingError(
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

        if enrich_pending:
            try:
                (
                    indexed_count,
                    processed_count,
                    vector_document_count,
                ) = self._flush_enrich_batch(
                    enrich_pending,
                    vector_batch=vector_batch,
                    indexed_at=indexed_at,
                    total_count=total_count,
                    indexed_count=indexed_count,
                    unchanged_count=unchanged_count,
                    error_count=error_count,
                    vector_document_count=vector_document_count,
                    processed_count=processed_count,
                )
            except Exception as exc:
                error_count, processed_count = self._handle_pending_error_batch(
                    enrich_pending,
                    error_message=str(exc),
                    indexed_at=indexed_at,
                    total_count=total_count,
                    indexed_count=indexed_count,
                    unchanged_count=unchanged_count,
                    error_count=error_count,
                    vector_document_count=vector_document_count,
                    processed_count=processed_count,
                    errors=errors,
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
            "discovered_count": discovered_count,
            "indexed_count": indexed_count,
            "unchanged_count": unchanged_count,
            "error_count": error_count,
            "vector_document_count": vector_document_count,
            "errors": errors,
        }

    def _flush_enrich_batch(
        self,
        pending: list[tuple[IndexedDatabaseFile, DatabaseIndexResult]],
        *,
        vector_batch: list[DatabaseIndexResult],
        indexed_at: datetime,
        total_count: int,
        indexed_count: int,
        unchanged_count: int,
        error_count: int,
        vector_document_count: int,
        processed_count: int,
    ) -> tuple[int, int, int]:
        if not pending:
            return indexed_count, processed_count, vector_document_count

        results = [result for _, result in pending]
        if self._enricher is not None:
            results = self._enricher.enrich_batch(results)

        indexed_files = [item for item, _ in pending]
        for indexed_file, result in zip(indexed_files, results, strict=True):
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

        return indexed_count, processed_count, vector_document_count

    def _handle_pending_error_batch(
        self,
        pending: list[tuple[IndexedDatabaseFile, DatabaseIndexResult]],
        *,
        error_message: str,
        indexed_at: datetime,
        total_count: int,
        indexed_count: int,
        unchanged_count: int,
        error_count: int,
        vector_document_count: int,
        processed_count: int,
        errors: list[DatabaseIndexingError],
    ) -> tuple[int, int]:
        for indexed_file, result in pending:
            file_path = self._root_dir / indexed_file.relative_path
            self._repository.save_error(
                source_id=result.source_id,
                relative_path=indexed_file.relative_path,
                filename=file_path.name,
                file_format=file_path.suffix.lower().removeprefix("."),
                size_bytes=indexed_file.size_bytes,
                last_modified=indexed_file.last_modified,
                error_message=error_message,
                indexed_at=indexed_at,
            )
            errors.append(
                DatabaseIndexingError(
                    relative_path=indexed_file.relative_path,
                    message=error_message,
                )
            )
            error_count += 1
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
        return error_count, processed_count

    def _scan_files(self, prefix: str) -> list[IndexedDatabaseFile]:
        base_dir = self._root_dir if not prefix else (self._root_dir / prefix).resolve()
        if not base_dir.exists():
            return []

        files: list[IndexedDatabaseFile] = []
        for path in sorted(base_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            stat = path.stat()
            files.append(
                IndexedDatabaseFile(
                    relative_path=path.relative_to(self._root_dir).as_posix(),
                    size_bytes=stat.st_size,
                    last_modified=datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ),
                )
            )
        return files

    def _flush_vector_batch(self, batch: list[DatabaseIndexResult]) -> int:
        if self._vector_store is None or not batch:
            return 0
        document_ids = add_database_results(self._vector_store, batch)
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
            DatabaseIndexingProgress(
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


def _is_unchanged(previous: Any, current: IndexedDatabaseFile) -> bool:
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
    import hashlib

    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"source_{digest}"
