from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Mapping

from lake_agent.domain.enums import FileStatus
from lake_agent.domain.models import FileMetadata


class InventoryRepository:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def find_object(self, object_identity: str) -> Mapping[str, Any] | None:
        return self._connection.execute(
            """
            SELECT object_id, etag, size_bytes, version_id, last_modified,
                   status
            FROM storage_objects
            WHERE object_identity = %s
            """,
            (object_identity,),
        ).fetchone()

    def save(self, obj: FileMetadata, scanned_at: datetime) -> None:
        self._connection.execute(
            """
            INSERT INTO storage_objects (
                object_id, object_identity, object_key, version_id,
                etag, filename, extension, size_bytes, last_modified,
                declared_content_type, detected_mime_type, detected_format,
                modality, encoding, identification_confidence,
                user_metadata, warnings, status, is_present,
                first_seen_at, last_seen_at
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s::jsonb, %s::jsonb, %s, %s,
                %s, %s
            )
            ON CONFLICT (object_identity) DO UPDATE SET
                etag = EXCLUDED.etag,
                filename = EXCLUDED.filename,
                extension = EXCLUDED.extension,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                declared_content_type = COALESCE(
                    EXCLUDED.declared_content_type,
                    storage_objects.declared_content_type
                ),
                detected_mime_type = COALESCE(
                    EXCLUDED.detected_mime_type,
                    storage_objects.detected_mime_type
                ),
                detected_format = COALESCE(
                    EXCLUDED.detected_format,
                    storage_objects.detected_format
                ),
                modality = COALESCE(EXCLUDED.modality, storage_objects.modality),
                encoding = COALESCE(EXCLUDED.encoding, storage_objects.encoding),
                identification_confidence = COALESCE(
                    EXCLUDED.identification_confidence,
                    storage_objects.identification_confidence
                ),
                user_metadata = CASE
                    WHEN EXCLUDED.user_metadata = '{}'::jsonb THEN storage_objects.user_metadata
                    ELSE EXCLUDED.user_metadata
                END,
                warnings = EXCLUDED.warnings,
                status = EXCLUDED.status,
                is_present = EXCLUDED.is_present,
                last_seen_at = EXCLUDED.last_seen_at,
                updated_at = NOW()
            """,
            (
                obj.object_id,
                obj.identity,
                obj.object_key,
                obj.version_id,
                obj.etag,
                obj.filename,
                obj.extension,
                obj.size_bytes,
                obj.last_modified,
                obj.declared_content_type,
                obj.detected_mime_type,
                obj.detected_format,
                obj.modality.value if obj.modality else None,
                obj.encoding,
                obj.identification_confidence,
                json.dumps(obj.user_metadata),
                json.dumps(obj.warnings),
                obj.status.value,
                obj.is_present,
                scanned_at,
                scanned_at,
            ),
        )

    def mark_missing(self, prefix: str, scanned_at: datetime) -> None:
        if prefix:
            self._connection.execute(
                """
                UPDATE storage_objects
                SET is_present = FALSE,
                    status = %s,
                    updated_at = NOW()
                WHERE (
                    object_key = %s
                    OR starts_with(object_key, %s)
                )
                  AND last_seen_at < %s
                  AND is_present = TRUE
                """,
                (
                    FileStatus.MISSING.value,
                    prefix,
                    f"{prefix}/",
                    scanned_at,
                ),
            )
            return

        self._connection.execute(
            """
            UPDATE storage_objects
            SET is_present = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE last_seen_at < %s
              AND is_present = TRUE
            """,
            (FileStatus.MISSING.value, scanned_at),
        )
