from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath

from lake_agent.domain.enums import FileStatus, Modality


@dataclass(frozen=True, slots=True)
class ObjectLocator:
    """Stable address of an object in an S3-compatible object store."""

    bucket: str
    object_key: str
    version_id: str | None = None

    def __post_init__(self) -> None:
        if not self.bucket:
            raise ValueError("bucket must not be empty")
        if not self.object_key:
            raise ValueError("object_key must not be empty")

    @property
    def identity(self) -> str:
        # JSON avoids ambiguous delimiter-based identities for arbitrary object keys.
        return json.dumps(
            [self.bucket, self.object_key, self.version_id or ""],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @property
    def object_id(self) -> str:
        return hashlib.sha256(self.identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DiscoveredObject:
    """Metadata obtained from MinIO listing/stat calls."""

    locator: ObjectLocator
    etag: str | None
    size_bytes: int
    last_modified: datetime | None
    declared_content_type: str | None = None
    user_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")

    @property
    def bucket(self) -> str:
        return self.locator.bucket

    @property
    def object_key(self) -> str:
        return self.locator.object_key

    @property
    def version_id(self) -> str | None:
        return self.locator.version_id

    @property
    def filename(self) -> str:
        return PurePosixPath(self.object_key).name

    @property
    def extension(self) -> str:
        return PurePosixPath(self.filename).suffix.lower()


@dataclass(frozen=True, slots=True)
class IdentificationResult:
    """System-inferred metadata; kept separate from MinIO-declared metadata."""

    locator: ObjectLocator
    detected_mime_type: str
    detected_format: str
    modality: Modality
    encoding: str | None
    confidence: float
    sha256: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class InventoryRunResult:
    run_id: str
    bucket: str
    prefix: str
    status: FileStatus
    discovered_count: int
    identified_count: int
    unchanged_count: int
    error_count: int
