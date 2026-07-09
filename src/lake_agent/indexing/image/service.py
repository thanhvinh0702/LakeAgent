from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from lake_agent.domain.indexing_models import ImageIndexResult
from lake_agent.indexing.image.deterministic import DeterministicImageParser
from lake_agent.indexing.image.ocr import OCRMarkdownExtractor
from lake_agent.indexing.image.vlm import ImageVLMEnricher
from lake_agent.indexing.image.vector_store import add_image_results
from lake_agent.persistence.repositories import ImageIndexRepository

_SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True, slots=True)
class IndexedImageFile:
    relative_path: str
    size_bytes: int
    last_modified: datetime


@dataclass(frozen=True, slots=True)
class ImageIndexingProgress:
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
class ImageIndexingError:
    relative_path: str
    message: str


class ImageIndexingService:
    def __init__(
        self,
        root_dir: str | Path,
        parser: DeterministicImageParser,
        repository: ImageIndexRepository,
        *,
        ocr_extractor: OCRMarkdownExtractor | None = None,
        vlm_enricher: ImageVLMEnricher | None = None,
        vector_store: Any | None = None,
        vl_batch_size: int = 5,
        vector_batch_size: int = 25,
        progress_callback: Callable[[ImageIndexingProgress], None] | None = None,
    ) -> None:
        if vl_batch_size <= 0:
            raise ValueError("vl_batch_size must be positive")
        if vector_batch_size <= 0:
            raise ValueError("vector_batch_size must be positive")
        self._root_dir = Path(root_dir).expanduser().resolve()
        self._parser = parser
        self._repository = repository
        self._ocr_extractor = ocr_extractor
        self._vlm_enricher = vlm_enricher
        self._vector_store = vector_store
        self._vl_batch_size = vl_batch_size
        self._vector_batch_size = vector_batch_size
        self._progress_callback = progress_callback

    def run(self, prefix: str = "") -> dict[str, int | str | list[ImageIndexingError]]:
        normalized_prefix = _normalize_prefix(prefix)
        indexed_at = datetime.now(timezone.utc)
        discovered_count = 0
        indexed_count = 0
        unchanged_count = 0
        error_count = 0
        vector_document_count = 0
        errors: list[ImageIndexingError] = []
        files = self._scan_files(normalized_prefix)
        total_count = len(files)
        processed_count = 0
        pending: list[tuple[IndexedImageFile, ImageIndexResult]] = []
        vector_batch: list[ImageIndexResult] = []

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
                self._emit_progress(
                    event="parsing",
                    relative_path=indexed_file.relative_path,
                    processed_count=processed_count,
                    total_count=total_count,
                    indexed_count=indexed_count,
                    unchanged_count=unchanged_count,
                    error_count=error_count,
                    vector_document_count=vector_document_count,
                    message="Parsing deterministic image metadata",
                )
                result = self._parser.parse_file(
                    absolute_path,
                    relative_path=indexed_file.relative_path,
                    source_id=source_id,
                )
                self._emit_progress(
                    event="parsed",
                    relative_path=indexed_file.relative_path,
                    processed_count=processed_count,
                    total_count=total_count,
                    indexed_count=indexed_count,
                    unchanged_count=unchanged_count,
                    error_count=error_count,
                    vector_document_count=vector_document_count,
                    message=(
                        f"Parsed image metadata: {result.width}x{result.height}, "
                        f"format={result.file_format}, mode={result.color_mode}"
                    ),
                )
                pending.append((indexed_file, result))
                if len(pending) >= self._pending_flush_size():
                    indexed_count, processed_count, vector_document_count = self._flush_pending(
                        pending,
                        vector_batch=vector_batch,
                        indexed_at=indexed_at,
                        total_count=total_count,
                        indexed_count=indexed_count,
                        unchanged_count=unchanged_count,
                        error_count=error_count,
                        vector_document_count=vector_document_count,
                        processed_count=processed_count,
                    )
                    pending.clear()
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
                    ImageIndexingError(
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

        if pending:
            indexed_count, processed_count, vector_document_count = self._flush_pending(
                pending,
                vector_batch=vector_batch,
                indexed_at=indexed_at,
                total_count=total_count,
                indexed_count=indexed_count,
                unchanged_count=unchanged_count,
                error_count=error_count,
                vector_document_count=vector_document_count,
                processed_count=processed_count,
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

    def _flush_pending(
        self,
        pending: list[tuple[IndexedImageFile, ImageIndexResult]],
        *,
        vector_batch: list[ImageIndexResult],
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

        self._emit_progress(
            event="flush_pending",
            relative_path=pending[0][0].relative_path,
            processed_count=processed_count,
            total_count=total_count,
            indexed_count=indexed_count,
            unchanged_count=unchanged_count,
            error_count=error_count,
            vector_document_count=vector_document_count,
            message=f"Flushing pending batch of {len(pending)} image(s)",
        )

        sections_by_source: dict[str, list] = {}
        if self._ocr_extractor is not None:
            try:
                self._emit_progress(
                    event="ocr_batch",
                    relative_path=pending[0][0].relative_path,
                    processed_count=processed_count,
                    total_count=total_count,
                    indexed_count=indexed_count,
                    unchanged_count=unchanged_count,
                    error_count=error_count,
                    vector_document_count=vector_document_count,
                    message=f"Running OCR batch for {len(pending)} image(s)",
                )
                sections_by_source = self._ocr_extractor.extract_sections_batch(
                    [self._root_dir / item.relative_path for item, _ in pending],
                    source_ids=[result.source_id for _, result in pending],
                )
            except Exception as exc:
                warning = f"OCR extraction failed: {exc}"
                for _, result in pending:
                    result.parse_warnings.append(warning)

        for indexed_file, result in pending:
            if sections_by_source:
                result.sections = sections_by_source.get(result.source_id, [])

        if self._vlm_enricher is not None:
            try:
                self._emit_progress(
                    event="vlm_batch",
                    relative_path=pending[0][0].relative_path,
                    processed_count=processed_count,
                    total_count=total_count,
                    indexed_count=indexed_count,
                    unchanged_count=unchanged_count,
                    error_count=error_count,
                    vector_document_count=vector_document_count,
                    message=f"Running VLM batch for {len(pending)} image(s)",
                )
                self._vlm_enricher.enrich_batch(
                    [self._root_dir / item.relative_path for item, _ in pending],
                    [result for _, result in pending],
                )
            except Exception as exc:
                warning = f"VLM enrichment failed: {exc}"
                for _, result in pending:
                    result.parse_warnings.append(warning)

        for indexed_file, result in pending:
            self._emit_progress(
                event="saving",
                relative_path=indexed_file.relative_path,
                processed_count=processed_count,
                total_count=total_count,
                indexed_count=indexed_count,
                unchanged_count=unchanged_count,
                error_count=error_count,
                vector_document_count=vector_document_count,
                message="Saving indexed image result",
            )
            self._repository.save(
                result,
                size_bytes=indexed_file.size_bytes,
                last_modified=indexed_file.last_modified,
                indexed_at=indexed_at,
            )
            vector_batch.append(result)
            if len(vector_batch) >= self._vector_batch_size:
                self._emit_progress(
                    event="vector_flush",
                    relative_path=indexed_file.relative_path,
                    processed_count=processed_count,
                    total_count=total_count,
                    indexed_count=indexed_count,
                    unchanged_count=unchanged_count,
                    error_count=error_count,
                    vector_document_count=vector_document_count,
                    message=f"Flushing {len(vector_batch)} vector document(s)",
                )
                vector_document_count += self._flush_vector_batch(vector_batch)
                vector_batch.clear()
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

    def _flush_vector_batch(self, batch: list[ImageIndexResult]) -> int:
        if self._vector_store is None or not batch:
            return 0
        document_ids = add_image_results(self._vector_store, batch)
        return len(document_ids)

    def _pending_flush_size(self) -> int:
        sizes: list[int] = []
        if self._ocr_extractor is not None:
            sizes.append(self._ocr_extractor.batch_size)
        if self._vlm_enricher is not None:
            sizes.append(self._vl_batch_size)
        return min(sizes) if sizes else 1

    def _scan_files(self, prefix: str) -> list[IndexedImageFile]:
        base_dir = self._root_dir if not prefix else (self._root_dir / prefix).resolve()
        if not base_dir.exists():
            return []

        files: list[IndexedImageFile] = []
        for path in sorted(base_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_SUFFIXES:
                continue
            stat = path.stat()
            files.append(
                IndexedImageFile(
                    relative_path=path.relative_to(self._root_dir).as_posix(),
                    size_bytes=stat.st_size,
                    last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
            )
        return files

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
            ImageIndexingProgress(
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


def _is_unchanged(previous: Any, current: IndexedImageFile) -> bool:
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
