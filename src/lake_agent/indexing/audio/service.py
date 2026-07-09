from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from lake_agent.domain.indexing_models import AudioIndexResult
from lake_agent.indexing.audio.deterministic import AudioTranscriptParser
from lake_agent.indexing.audio.enrichment import AudioLLMEnricher
from lake_agent.indexing.audio.vector_store import add_audio_results
from lake_agent.persistence.repositories import AudioIndexRepository

_SUPPORTED_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


@dataclass(frozen=True, slots=True)
class IndexedAudioFile:
    relative_path: str
    size_bytes: int
    last_modified: datetime


@dataclass(frozen=True, slots=True)
class AudioIndexingProgress:
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
class AudioIndexingError:
    relative_path: str
    message: str


class AudioIndexingService:
    def __init__(
        self,
        root_dir: str | Path,
        parser: AudioTranscriptParser,
        repository: AudioIndexRepository,
        *,
        enricher: AudioLLMEnricher | None = None,
        vector_store: Any | None = None,
        enrich_batch_size: int = 10,
        vector_batch_size: int = 25,
        progress_callback: Callable[[AudioIndexingProgress], None] | None = None,
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

    def run(
        self,
        prefix: str = "",
        *,
        force: bool = False,
    ) -> dict[str, int | str | list[AudioIndexingError]]:
        normalized_prefix = _normalize_prefix(prefix)
        indexed_at = datetime.now(timezone.utc)
        discovered_count = 0
        indexed_count = 0
        unchanged_count = 0
        error_count = 0
        vector_document_count = 0
        processed_count = 0
        errors: list[AudioIndexingError] = []
        enrich_pending: list[tuple[IndexedAudioFile, AudioIndexResult]] = []
        vector_batch: list[AudioIndexResult] = []
        files = self._scan_files(normalized_prefix)
        total_count = len(files)

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
                enrich_pending.append((indexed_file, result))
                if self._enricher is None or len(enrich_pending) >= self._enrich_batch_size:
                    indexed_count, processed_count, vector_document_count = self._flush_enrich_batch(
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
                    if self._vector_store is not None and len(vector_batch) >= self._vector_batch_size:
                        vector_added, vector_errors = self._flush_vector_batch_with_fallback(vector_batch)
                        vector_document_count += vector_added
                        if vector_errors:
                            error_count += len(vector_errors)
                            errors.extend(vector_errors)
                        vector_batch.clear()
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
                    AudioIndexingError(
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
            indexed_count, processed_count, vector_document_count = self._flush_enrich_batch(
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
        vector_added, vector_errors = self._flush_vector_batch_with_fallback(vector_batch)
        vector_document_count += vector_added
        if vector_errors:
            error_count += len(vector_errors)
            errors.extend(vector_errors)
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
        pending: list[tuple[IndexedAudioFile, AudioIndexResult]],
        *,
        vector_batch: list[AudioIndexResult],
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

        indexed_files = [item for item, _ in pending]
        results = [result for _, result in pending]
        if self._enricher is not None:
            results = self._enricher.enrich_batch(results)

        for indexed_file, result in zip(indexed_files, results, strict=True):
            self._repository.save(
                result,
                size_bytes=indexed_file.size_bytes,
                last_modified=indexed_file.last_modified,
                indexed_at=indexed_at,
            )
            vector_batch.append(result)
            indexed_count += 1
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

    def _scan_files(self, prefix: str) -> list[IndexedAudioFile]:
        base_dir = self._root_dir if not prefix else (self._root_dir / prefix).resolve()
        if not base_dir.exists():
            return []

        files: list[IndexedAudioFile] = []
        for path in sorted(base_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            stat = path.stat()
            files.append(
                IndexedAudioFile(
                    relative_path=path.relative_to(self._root_dir).as_posix(),
                    size_bytes=stat.st_size,
                    last_modified=datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ),
                )
            )
        return files

    def _flush_vector_batch(self, batch: list[AudioIndexResult]) -> int:
        if self._vector_store is None or not batch:
            return 0
        document_ids = add_audio_results(self._vector_store, batch)
        return len(document_ids)

    def _flush_vector_batch_with_fallback(
        self,
        batch: list[AudioIndexResult],
    ) -> tuple[int, list[AudioIndexingError]]:
        if self._vector_store is None or not batch:
            return 0, []

        try:
            return self._retry_vector_flush(batch), []
        except Exception as batch_exc:
            added = 0
            errors: list[AudioIndexingError] = []
            for result in batch:
                try:
                    added += self._retry_vector_flush([result])
                except Exception as item_exc:
                    errors.append(
                        AudioIndexingError(
                            relative_path=result.relative_path,
                            message=(
                                "Vector indexing failed after retry/fallback: "
                                f"batch_error={batch_exc}; item_error={item_exc}"
                            ),
                        )
                    )
            return added, errors

    def _retry_vector_flush(self, batch: list[AudioIndexResult]) -> int:
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                return self._flush_vector_batch(batch)
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(1.5 * attempt)
        assert last_error is not None
        raise last_error

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
            AudioIndexingProgress(
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


def _is_unchanged(previous: Any, current: IndexedAudioFile) -> bool:
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
