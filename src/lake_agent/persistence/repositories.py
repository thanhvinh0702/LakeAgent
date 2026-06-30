from __future__ import annotations

import json
import uuid
from typing import Any, Mapping

from lake_agent.domain.enums import FileStatus
from lake_agent.domain.models import DiscoveredObject, IdentificationResult


class InventoryRepository:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def create_run(self, bucket: str, prefix: str) -> str:
        run_id = str(uuid.uuid4())
        self._connection.execute(
            """
            INSERT INTO inventory_runs (run_id, bucket, prefix, status)
            VALUES (%s, %s, %s, %s)
            """,
            (run_id, bucket, prefix, FileStatus.DISCOVERED.value),
        )
        return run_id

    def find_object(self, object_identity: str) -> Mapping[str, Any] | None:
        return self._connection.execute(
            """
            SELECT object_id, etag, size_bytes, version_id, last_modified,
                   status, sha256
            FROM storage_objects
            WHERE object_identity = %s
            """,
            (object_identity,),
        ).fetchone()

    def mark_seen(
        self,
        obj: DiscoveredObject,
        run_id: str,
    ) -> None:
        self._connection.execute(
            """
            UPDATE storage_objects
            SET etag = %s,
                size_bytes = %s,
                last_modified = %s,
                declared_content_type = COALESCE(%s, declared_content_type),
                user_metadata = CASE
                    WHEN %s::jsonb = '{}'::jsonb THEN user_metadata
                    ELSE %s::jsonb
                END,
                last_seen_run_id = %s,
                is_present = TRUE,
                status = %s,
                updated_at = NOW()
            WHERE object_identity = %s
            """,
            (
                obj.etag,
                obj.size_bytes,
                obj.last_modified,
                obj.declared_content_type,
                json.dumps(obj.user_metadata),
                json.dumps(obj.user_metadata),
                run_id,
                FileStatus.IDENTIFIED.value,
                obj.locator.identity,
            ),
        )

    def upsert_identified(
        self,
        obj: DiscoveredObject,
        result: IdentificationResult,
        run_id: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO storage_objects (
                object_id, object_identity, bucket, object_key, version_id,
                etag, filename, extension, size_bytes, last_modified,
                declared_content_type, detected_mime_type, detected_format,
                modality, encoding, identification_confidence, sha256,
                user_metadata, warnings, status,
                first_seen_run_id, last_seen_run_id, is_present
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s,
                %s::jsonb, %s::jsonb, %s,
                %s, %s, TRUE
            )
            ON CONFLICT (object_identity) DO UPDATE SET
                etag = EXCLUDED.etag,
                filename = EXCLUDED.filename,
                extension = EXCLUDED.extension,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                declared_content_type = EXCLUDED.declared_content_type,
                detected_mime_type = EXCLUDED.detected_mime_type,
                detected_format = EXCLUDED.detected_format,
                modality = EXCLUDED.modality,
                encoding = EXCLUDED.encoding,
                identification_confidence = EXCLUDED.identification_confidence,
                sha256 = EXCLUDED.sha256,
                user_metadata = EXCLUDED.user_metadata,
                warnings = EXCLUDED.warnings,
                status = EXCLUDED.status,
                last_seen_run_id = EXCLUDED.last_seen_run_id,
                is_present = TRUE,
                updated_at = NOW()
            """,
            (
                obj.locator.object_id,
                obj.locator.identity,
                obj.bucket,
                obj.object_key,
                obj.version_id,
                obj.etag,
                obj.filename,
                obj.extension,
                obj.size_bytes,
                obj.last_modified,
                obj.declared_content_type,
                result.detected_mime_type,
                result.detected_format,
                result.modality.value,
                result.encoding,
                result.confidence,
                result.sha256,
                json.dumps(obj.user_metadata),
                json.dumps(result.warnings),
                FileStatus.IDENTIFIED.value,
                run_id,
                run_id,
            ),
        )

    def upsert_failed_object(
        self,
        obj: DiscoveredObject,
        run_id: str,
        message: str,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO storage_objects (
                object_id, object_identity, bucket, object_key, version_id,
                etag, filename, extension, size_bytes, last_modified,
                declared_content_type, user_metadata, warnings, status,
                first_seen_run_id, last_seen_run_id, is_present
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s::jsonb, %s::jsonb, %s,
                %s, %s, TRUE
            )
            ON CONFLICT (object_identity) DO UPDATE SET
                etag = EXCLUDED.etag,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                declared_content_type = EXCLUDED.declared_content_type,
                user_metadata = EXCLUDED.user_metadata,
                warnings = EXCLUDED.warnings,
                status = EXCLUDED.status,
                detected_mime_type = NULL,
                detected_format = NULL,
                modality = NULL,
                encoding = NULL,
                identification_confidence = NULL,
                sha256 = NULL,
                last_seen_run_id = EXCLUDED.last_seen_run_id,
                is_present = TRUE,
                updated_at = NOW()
            """,
            (
                obj.locator.object_id,
                obj.locator.identity,
                obj.bucket,
                obj.object_key,
                obj.version_id,
                obj.etag,
                obj.filename,
                obj.extension,
                obj.size_bytes,
                obj.last_modified,
                obj.declared_content_type,
                json.dumps(obj.user_metadata),
                json.dumps([message]),
                FileStatus.ERROR.value,
                run_id,
                run_id,
            ),
        )

    def record_error(
        self,
        run_id: str,
        bucket: str,
        object_key: str | None,
        version_id: str | None,
        stage: str,
        error: Exception,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO inventory_errors (
                error_id, run_id, bucket, object_key, version_id,
                stage, error_type, message
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                run_id,
                bucket,
                object_key,
                version_id,
                stage,
                type(error).__name__,
                str(error),
            ),
        )

    def mark_listing_completed(self, run_id: str) -> None:
        self._connection.execute(
            "UPDATE inventory_runs SET listing_completed = TRUE WHERE run_id = %s",
            (run_id,),
        )

    def mark_unseen_missing(self, run_id: str, bucket: str, prefix: str) -> None:
        self._connection.execute(
            """
            UPDATE storage_objects
            SET is_present = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE bucket = %s
              AND starts_with(object_key, %s)
              AND last_seen_run_id <> %s
              AND is_present = TRUE
            """,
            (FileStatus.MISSING.value, bucket, prefix, run_id),
        )

    def complete_run(
        self,
        run_id: str,
        *,
        discovered_count: int,
        identified_count: int,
        unchanged_count: int,
        error_count: int,
    ) -> None:
        self._connection.execute(
            """
            UPDATE inventory_runs
            SET completed_at = NOW(),
                status = %s,
                discovered_count = %s,
                identified_count = %s,
                unchanged_count = %s,
                error_count = %s
            WHERE run_id = %s
            """,
            (
                FileStatus.COMPLETED.value,
                discovered_count,
                identified_count,
                unchanged_count,
                error_count,
                run_id,
            ),
        )

    def fail_run(self, run_id: str, error: Exception) -> None:
        self._connection.execute(
            """
            UPDATE inventory_runs
            SET completed_at = NOW(), status = %s, error_message = %s
            WHERE run_id = %s
            """,
            (FileStatus.FAILED.value, str(error), run_id),
        )
