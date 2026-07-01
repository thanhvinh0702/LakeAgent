from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath

from lake_agent.domain.enums import FileStatus, Modality


@dataclass(frozen=True, slots=True)
class FileMetadata:
    """All metadata tracked for a single object in storage."""

    object_key: str
    version_id: str | None = None
    etag: str | None = None
    size_bytes: int = 0
    last_modified: datetime | None = None
    declared_content_type: str | None = None
    user_metadata: dict[str, str] = field(default_factory=dict)
    detected_mime_type: str | None = None
    detected_format: str | None = None
    modality: Modality | None = None
    encoding: str | None = None
    identification_confidence: float | None = None
    warnings: tuple[str, ...] = ()
    status: FileStatus = FileStatus.IDENTIFIED
    is_present: bool = True

    def __post_init__(self) -> None:
        if not self.object_key:
            raise ValueError("object_key must not be empty")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be non-negative")
        if (
            self.identification_confidence is not None
            and not 0.0 <= self.identification_confidence <= 1.0
        ):
            raise ValueError("identification_confidence must be between 0 and 1")

    @property
    def identity(self) -> str:
        return json.dumps(
            [self.object_key, self.version_id or ""],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @property
    def object_id(self) -> str:
        return hashlib.sha256(self.identity.encode("utf-8")).hexdigest()

    @property
    def filename(self) -> str:
        return PurePosixPath(self.object_key).name

    @property
    def extension(self) -> str:
        return PurePosixPath(self.filename).suffix.lower()
