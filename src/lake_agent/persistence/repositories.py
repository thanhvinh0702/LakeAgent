from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any, Mapping

from lake_agent.domain.enums import FileStatus
from lake_agent.domain.indexing_models import (
    DatabaseIndexResult,
    DocumentIndexResult,
    ImageIndexResult,
    SlideshowIndexResult,
    SqlScriptIndexResult,
    TabularIndexResult,
    TextIndexResult,
    WebIndexResult,
)
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


class TabularIndexRepository:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def find_file(self, relative_path: str) -> Mapping[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, size_bytes, last_modified, file_format, status
            FROM tabular_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def save(
        self,
        result: TabularIndexResult,
        *,
        size_bytes: int,
        last_modified: datetime | None,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO tabular_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, parser_version, parse_warnings,
                workbook_sheet_descriptions, file_summary, file_keywords,
                file_search_text, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb,
                %s::jsonb, %s, %s::jsonb,
                %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                parser_version = EXCLUDED.parser_version,
                parse_warnings = EXCLUDED.parse_warnings,
                workbook_sheet_descriptions = EXCLUDED.workbook_sheet_descriptions,
                file_summary = EXCLUDED.file_summary,
                file_keywords = EXCLUDED.file_keywords,
                file_search_text = EXCLUDED.file_search_text,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                result.source_id,
                result.relative_path,
                result.filename,
                result.file_format,
                size_bytes,
                last_modified,
                result.parser_version,
                json.dumps(result.parse_warnings),
                json.dumps(result.workbook_sheet_descriptions),
                result.file_summary,
                json.dumps(result.file_keywords),
                result.file_search_text,
                "indexed",
                None,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM tabular_tables WHERE source_id = %s",
            (result.source_id,),
        )

        for table in result.tables:
            self._connection.execute(
                """
                INSERT INTO tabular_tables (
                    table_id, source_id, table_name, sheet_name,
                    is_context_sheet, sheet_description, header_row_index,
                    context_before_header, raw_header, row_count, column_count,
                    columns_json, preview_rows, summary, keywords,
                    table_search_text, warnings
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s, %s,
                    %s::jsonb, %s::jsonb, %s, %s::jsonb,
                    %s, %s::jsonb
                )
                """,
                (
                    table.table_id,
                    result.source_id,
                    table.table_name,
                    table.sheet_name,
                    table.is_context_sheet,
                    table.sheet_description,
                    table.header_row_index,
                    json.dumps(table.context_before_header),
                    json.dumps(table.raw_header),
                    table.row_count,
                    table.column_count,
                    json.dumps([asdict(column) for column in table.columns]),
                    json.dumps(table.preview_rows),
                    table.summary,
                    json.dumps(table.keywords),
                    table.table_search_text,
                    json.dumps(table.warnings),
                ),
            )

    def save_error(
        self,
        *,
        source_id: str,
        relative_path: str,
        filename: str,
        file_format: str,
        size_bytes: int,
        last_modified: datetime | None,
        error_message: str,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO tabular_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                source_id,
                relative_path,
                filename,
                file_format,
                size_bytes,
                last_modified,
                "error",
                error_message,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM tabular_tables WHERE source_id = %s",
            (source_id,),
        )

    def mark_missing(self, prefix: str, indexed_at: datetime) -> None:
        if prefix:
            self._connection.execute(
                """
                UPDATE tabular_files
                SET is_present = FALSE,
                    status = %s,
                    updated_at = NOW()
                WHERE (
                    relative_path = %s
                    OR starts_with(relative_path, %s)
                )
                  AND last_indexed_at < %s
                  AND is_present = TRUE
                """,
                (
                    FileStatus.MISSING.value,
                    prefix,
                    f"{prefix}/",
                    indexed_at,
                ),
            )
            return

        self._connection.execute(
            """
            UPDATE tabular_files
            SET is_present = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE last_indexed_at < %s
              AND is_present = TRUE
            """,
            (FileStatus.MISSING.value, indexed_at),
        )


class TextIndexRepository:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def find_file(self, relative_path: str) -> Mapping[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, size_bytes, last_modified, file_format, status
            FROM text_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def save(
        self,
        result: TextIndexResult,
        *,
        size_bytes: int,
        last_modified: datetime | None,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO text_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, parser_version, parse_warnings,
                file_summary, file_keywords, file_search_text, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb,
                %s, %s::jsonb, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                parser_version = EXCLUDED.parser_version,
                parse_warnings = EXCLUDED.parse_warnings,
                file_summary = EXCLUDED.file_summary,
                file_keywords = EXCLUDED.file_keywords,
                file_search_text = EXCLUDED.file_search_text,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                result.source_id,
                result.relative_path,
                result.filename,
                result.file_format,
                size_bytes,
                last_modified,
                result.parser_version,
                json.dumps(result.parse_warnings),
                result.file_summary,
                json.dumps(result.file_keywords),
                result.file_search_text,
                "indexed",
                None,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM text_sections WHERE source_id = %s",
            (result.source_id,),
        )

        for section in result.sections:
            self._connection.execute(
                """
                INSERT INTO text_sections (
                    section_id, source_id, chunk_index, heading,
                    content, line_start, line_end, char_count,
                    search_text, warnings
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s::jsonb
                )
                """,
                (
                    section.section_id,
                    result.source_id,
                    section.chunk_index,
                    section.heading,
                    section.content,
                    section.line_start,
                    section.line_end,
                    section.char_count,
                    section.search_text,
                    json.dumps(section.warnings),
                ),
            )

    def save_error(
        self,
        *,
        source_id: str,
        relative_path: str,
        filename: str,
        file_format: str,
        size_bytes: int,
        last_modified: datetime | None,
        error_message: str,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO text_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                source_id,
                relative_path,
                filename,
                file_format,
                size_bytes,
                last_modified,
                "error",
                error_message,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM text_sections WHERE source_id = %s",
            (source_id,),
        )

    def mark_missing(self, prefix: str, indexed_at: datetime) -> None:
        if prefix:
            self._connection.execute(
                """
                UPDATE text_files
                SET is_present = FALSE,
                    status = %s,
                    updated_at = NOW()
                WHERE (
                    relative_path = %s
                    OR starts_with(relative_path, %s)
                )
                  AND last_indexed_at < %s
                  AND is_present = TRUE
                """,
                (
                    FileStatus.MISSING.value,
                    prefix,
                    f"{prefix}/",
                    indexed_at,
                ),
            )
            return

        self._connection.execute(
            """
            UPDATE text_files
            SET is_present = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE last_indexed_at < %s
              AND is_present = TRUE
            """,
            (FileStatus.MISSING.value, indexed_at),
        )


class DocumentIndexRepository:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def find_file(self, relative_path: str) -> Mapping[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, size_bytes, last_modified, file_format, status
            FROM document_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def save(
        self,
        result: DocumentIndexResult,
        *,
        size_bytes: int,
        last_modified: datetime | None,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO document_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, parser_version, parse_warnings,
                file_summary, file_keywords, file_search_text, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb,
                %s, %s::jsonb, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                parser_version = EXCLUDED.parser_version,
                parse_warnings = EXCLUDED.parse_warnings,
                file_summary = EXCLUDED.file_summary,
                file_keywords = EXCLUDED.file_keywords,
                file_search_text = EXCLUDED.file_search_text,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                result.source_id,
                result.relative_path,
                result.filename,
                result.file_format,
                size_bytes,
                last_modified,
                result.parser_version,
                json.dumps(result.parse_warnings),
                result.file_summary,
                json.dumps(result.file_keywords),
                result.file_search_text,
                "indexed",
                None,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM document_sections WHERE source_id = %s",
            (result.source_id,),
        )

        for section in result.sections:
            self._connection.execute(
                """
                INSERT INTO document_sections (
                    section_id, source_id, section_type, chunk_index, heading,
                    content, page_start, page_end, char_count,
                    search_text, image_id, image_index, warnings
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s::jsonb
                )
                """,
                (
                    section.section_id,
                    result.source_id,
                    section.section_type,
                    section.chunk_index,
                    section.heading,
                    section.content,
                    section.page_start,
                    section.page_end,
                    section.char_count,
                    section.search_text,
                    section.image_id,
                    section.image_index,
                    json.dumps(section.warnings),
                ),
            )

    def save_error(
        self,
        *,
        source_id: str,
        relative_path: str,
        filename: str,
        file_format: str,
        size_bytes: int,
        last_modified: datetime | None,
        error_message: str,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO document_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                source_id,
                relative_path,
                filename,
                file_format,
                size_bytes,
                last_modified,
                "error",
                error_message,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM document_sections WHERE source_id = %s",
            (source_id,),
        )

    def mark_missing(self, prefix: str, indexed_at: datetime) -> None:
        if prefix:
            self._connection.execute(
                """
                UPDATE document_files
                SET is_present = FALSE,
                    status = %s,
                    updated_at = NOW()
                WHERE (
                    relative_path = %s
                    OR starts_with(relative_path, %s)
                )
                  AND last_indexed_at < %s
                  AND is_present = TRUE
                """,
                (
                    FileStatus.MISSING.value,
                    prefix,
                    f"{prefix}/",
                    indexed_at,
                ),
            )
            return

        self._connection.execute(
            """
            UPDATE document_files
            SET is_present = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE last_indexed_at < %s
              AND is_present = TRUE
            """,
            (FileStatus.MISSING.value, indexed_at),
        )


class SlideshowIndexRepository:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def find_file(self, relative_path: str) -> Mapping[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, size_bytes, last_modified, file_format, status
            FROM slideshow_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def save(
        self,
        result: SlideshowIndexResult,
        *,
        size_bytes: int,
        last_modified: datetime | None,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO slideshow_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, parser_version, parse_warnings,
                file_summary, file_keywords, file_search_text, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb,
                %s, %s::jsonb, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                parser_version = EXCLUDED.parser_version,
                parse_warnings = EXCLUDED.parse_warnings,
                file_summary = EXCLUDED.file_summary,
                file_keywords = EXCLUDED.file_keywords,
                file_search_text = EXCLUDED.file_search_text,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                result.source_id,
                result.relative_path,
                result.filename,
                result.file_format,
                size_bytes,
                last_modified,
                result.parser_version,
                json.dumps(result.parse_warnings),
                result.file_summary,
                json.dumps(result.file_keywords),
                result.file_search_text,
                "indexed",
                None,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM slideshow_sections WHERE source_id = %s",
            (result.source_id,),
        )

        for section in result.sections:
            self._connection.execute(
                """
                INSERT INTO slideshow_sections (
                    section_id, source_id, section_type, chunk_index, heading,
                    content, slide_start, slide_end, char_count,
                    search_text, image_id, image_index, warnings
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s::jsonb
                )
                """,
                (
                    section.section_id,
                    result.source_id,
                    section.section_type,
                    section.chunk_index,
                    section.heading,
                    section.content,
                    section.slide_start,
                    section.slide_end,
                    section.char_count,
                    section.search_text,
                    section.image_id,
                    section.image_index,
                    json.dumps(section.warnings),
                ),
            )

    def save_error(
        self,
        *,
        source_id: str,
        relative_path: str,
        filename: str,
        file_format: str,
        size_bytes: int,
        last_modified: datetime | None,
        error_message: str,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO slideshow_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                source_id,
                relative_path,
                filename,
                file_format,
                size_bytes,
                last_modified,
                "error",
                error_message,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM slideshow_sections WHERE source_id = %s",
            (source_id,),
        )

    def mark_missing(self, prefix: str, indexed_at: datetime) -> None:
        if prefix:
            self._connection.execute(
                """
                UPDATE slideshow_files
                SET is_present = FALSE,
                    status = %s,
                    updated_at = NOW()
                WHERE (
                    relative_path = %s
                    OR starts_with(relative_path, %s)
                )
                  AND last_indexed_at < %s
                  AND is_present = TRUE
                """,
                (
                    FileStatus.MISSING.value,
                    prefix,
                    f"{prefix}/",
                    indexed_at,
                ),
            )
            return

        self._connection.execute(
            """
            UPDATE slideshow_files
            SET is_present = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE last_indexed_at < %s
              AND is_present = TRUE
            """,
            (FileStatus.MISSING.value, indexed_at),
        )


class ImageIndexRepository:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def find_file(self, relative_path: str) -> Mapping[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, size_bytes, last_modified, file_format, status
            FROM image_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def save(
        self,
        result: ImageIndexResult,
        *,
        size_bytes: int,
        last_modified: datetime | None,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO image_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, width, height, color_mode,
                has_alpha, is_animated, frame_count, parser_version, parse_warnings,
                file_summary, file_keywords, file_search_text, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s::jsonb,
                %s, %s::jsonb, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                width = EXCLUDED.width,
                height = EXCLUDED.height,
                color_mode = EXCLUDED.color_mode,
                has_alpha = EXCLUDED.has_alpha,
                is_animated = EXCLUDED.is_animated,
                frame_count = EXCLUDED.frame_count,
                parser_version = EXCLUDED.parser_version,
                parse_warnings = EXCLUDED.parse_warnings,
                file_summary = EXCLUDED.file_summary,
                file_keywords = EXCLUDED.file_keywords,
                file_search_text = EXCLUDED.file_search_text,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                result.source_id,
                result.relative_path,
                result.filename,
                result.file_format,
                size_bytes,
                last_modified,
                result.width,
                result.height,
                result.color_mode,
                result.has_alpha,
                result.is_animated,
                result.frame_count,
                result.parser_version,
                json.dumps(result.parse_warnings),
                result.file_summary,
                json.dumps(result.file_keywords),
                result.file_search_text,
                "indexed",
                None,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM image_sections WHERE source_id = %s",
            (result.source_id,),
        )

        for section in result.sections:
            self._connection.execute(
                """
                INSERT INTO image_sections (
                    section_id, source_id, section_type, chunk_index,
                    heading, content, line_start, line_end,
                    char_count, search_text, warnings
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s::jsonb
                )
                """,
                (
                    section.section_id,
                    result.source_id,
                    section.section_type,
                    section.chunk_index,
                    section.heading,
                    section.content,
                    section.line_start,
                    section.line_end,
                    section.char_count,
                    section.search_text,
                    json.dumps(section.warnings),
                ),
            )

    def save_error(
        self,
        *,
        source_id: str,
        relative_path: str,
        filename: str,
        file_format: str,
        size_bytes: int,
        last_modified: datetime | None,
        error_message: str,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO image_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, width, height, color_mode,
                has_alpha, is_animated, frame_count, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                source_id,
                relative_path,
                filename,
                file_format,
                size_bytes,
                last_modified,
                0,
                0,
                "",
                False,
                False,
                1,
                "error",
                error_message,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM image_sections WHERE source_id = %s",
            (source_id,),
        )

    def mark_missing(self, prefix: str, indexed_at: datetime) -> None:
        if prefix:
            self._connection.execute(
                """
                UPDATE image_files
                SET is_present = FALSE,
                    status = %s,
                    updated_at = NOW()
                WHERE (
                    relative_path = %s
                    OR starts_with(relative_path, %s)
                )
                  AND last_indexed_at < %s
                  AND is_present = TRUE
                """,
                (
                    FileStatus.MISSING.value,
                    prefix,
                    f"{prefix}/",
                    indexed_at,
                ),
            )
            return

        self._connection.execute(
            """
            UPDATE image_files
            SET is_present = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE last_indexed_at < %s
              AND is_present = TRUE
            """,
            (FileStatus.MISSING.value, indexed_at),
        )


class WebIndexRepository:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def find_file(self, relative_path: str) -> Mapping[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, size_bytes, last_modified, file_format, status
            FROM web_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def save(
        self,
        result: WebIndexResult,
        *,
        size_bytes: int,
        last_modified: datetime | None,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO web_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, parser_version, parse_warnings,
                file_summary, file_keywords, file_search_text, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb,
                %s, %s::jsonb, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                parser_version = EXCLUDED.parser_version,
                parse_warnings = EXCLUDED.parse_warnings,
                file_summary = EXCLUDED.file_summary,
                file_keywords = EXCLUDED.file_keywords,
                file_search_text = EXCLUDED.file_search_text,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                result.source_id,
                result.relative_path,
                result.filename,
                result.file_format,
                size_bytes,
                last_modified,
                result.parser_version,
                json.dumps(result.parse_warnings),
                result.file_summary,
                json.dumps(result.file_keywords),
                result.file_search_text,
                "indexed",
                None,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM web_sections WHERE source_id = %s",
            (result.source_id,),
        )

        for section in result.sections:
            self._connection.execute(
                """
                INSERT INTO web_sections (
                    section_id, source_id, chunk_index, heading,
                    content, char_count, search_text, warnings
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s::jsonb
                )
                """,
                (
                    section.section_id,
                    result.source_id,
                    section.chunk_index,
                    section.heading,
                    section.content,
                    section.char_count,
                    section.search_text,
                    json.dumps(section.warnings),
                ),
            )

    def save_error(
        self,
        *,
        source_id: str,
        relative_path: str,
        filename: str,
        file_format: str,
        size_bytes: int,
        last_modified: datetime | None,
        error_message: str,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO web_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                source_id,
                relative_path,
                filename,
                file_format,
                size_bytes,
                last_modified,
                "error",
                error_message,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM web_sections WHERE source_id = %s",
            (source_id,),
        )

    def mark_missing(self, prefix: str, indexed_at: datetime) -> None:
        if prefix:
            self._connection.execute(
                """
                UPDATE web_files
                SET is_present = FALSE,
                    status = %s,
                    updated_at = NOW()
                WHERE (
                    relative_path = %s
                    OR starts_with(relative_path, %s)
                )
                  AND last_indexed_at < %s
                  AND is_present = TRUE
                """,
                (
                    FileStatus.MISSING.value,
                    prefix,
                    f"{prefix}/",
                    indexed_at,
                ),
            )
            return

        self._connection.execute(
            """
            UPDATE web_files
            SET is_present = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE last_indexed_at < %s
              AND is_present = TRUE
            """,
            (FileStatus.MISSING.value, indexed_at),
        )


class SqlScriptIndexRepository:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def find_file(self, relative_path: str) -> Mapping[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, size_bytes, last_modified, file_format, status
            FROM sql_script_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def save(
        self,
        result: SqlScriptIndexResult,
        *,
        size_bytes: int,
        last_modified: datetime | None,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO sql_script_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, parser_version, parse_warnings,
                file_summary, file_keywords, file_search_text, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb,
                %s, %s::jsonb, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                parser_version = EXCLUDED.parser_version,
                parse_warnings = EXCLUDED.parse_warnings,
                file_summary = EXCLUDED.file_summary,
                file_keywords = EXCLUDED.file_keywords,
                file_search_text = EXCLUDED.file_search_text,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                result.source_id,
                result.relative_path,
                result.filename,
                result.file_format,
                size_bytes,
                last_modified,
                result.parser_version,
                json.dumps(result.parse_warnings),
                result.file_summary,
                json.dumps(result.file_keywords),
                result.file_search_text,
                "indexed",
                None,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM sql_script_sections WHERE source_id = %s",
            (result.source_id,),
        )

        for section in result.sections:
            self._connection.execute(
                """
                INSERT INTO sql_script_sections (
                    section_id, source_id, chunk_index, heading,
                    content, char_count, search_text, warnings
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s::jsonb
                )
                """,
                (
                    section.section_id,
                    result.source_id,
                    section.chunk_index,
                    section.heading,
                    section.content,
                    section.char_count,
                    section.search_text,
                    json.dumps(section.warnings),
                ),
            )

    def save_error(
        self,
        *,
        source_id: str,
        relative_path: str,
        filename: str,
        file_format: str,
        size_bytes: int,
        last_modified: datetime | None,
        error_message: str,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO sql_script_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                source_id,
                relative_path,
                filename,
                file_format,
                size_bytes,
                last_modified,
                "error",
                error_message,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM sql_script_sections WHERE source_id = %s",
            (source_id,),
        )

    def mark_missing(self, prefix: str, indexed_at: datetime) -> None:
        if prefix:
            self._connection.execute(
                """
                UPDATE sql_script_files
                SET is_present = FALSE,
                    status = %s,
                    updated_at = NOW()
                WHERE (
                    relative_path = %s
                    OR starts_with(relative_path, %s)
                )
                  AND last_indexed_at < %s
                  AND is_present = TRUE
                """,
                (
                    FileStatus.MISSING.value,
                    prefix,
                    f"{prefix}/",
                    indexed_at,
                ),
            )
            return

        self._connection.execute(
            """
            UPDATE sql_script_files
            SET is_present = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE last_indexed_at < %s
              AND is_present = TRUE
            """,
            (FileStatus.MISSING.value, indexed_at),
        )


class DatabaseIndexRepository:
    def __init__(self, connection: Any) -> None:
        self._connection = connection

    def find_file(self, relative_path: str) -> Mapping[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, size_bytes, last_modified, file_format, status
            FROM database_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def save(
        self,
        result: DatabaseIndexResult,
        *,
        size_bytes: int,
        last_modified: datetime | None,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO database_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, parser_version, parse_warnings,
                file_summary, file_keywords, file_search_text, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s::jsonb,
                %s, %s::jsonb, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                parser_version = EXCLUDED.parser_version,
                parse_warnings = EXCLUDED.parse_warnings,
                file_summary = EXCLUDED.file_summary,
                file_keywords = EXCLUDED.file_keywords,
                file_search_text = EXCLUDED.file_search_text,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                result.source_id,
                result.relative_path,
                result.filename,
                result.file_format,
                size_bytes,
                last_modified,
                result.parser_version,
                json.dumps(result.parse_warnings),
                result.file_summary,
                json.dumps(result.file_keywords),
                result.file_search_text,
                "indexed",
                None,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM database_tables WHERE source_id = %s",
            (result.source_id,),
        )

        for table in result.tables:
            self._connection.execute(
                """
                INSERT INTO database_tables (
                    table_id, source_id, table_name,
                    row_count, column_count, columns_json, preview_rows,
                    summary, keywords, table_search_text, warnings
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s::jsonb, %s::jsonb,
                    %s, %s::jsonb, %s, %s::jsonb
                )
                """,
                (
                    table.table_id,
                    result.source_id,
                    table.table_name,
                    table.row_count,
                    table.column_count,
                    json.dumps([asdict(col) for col in table.columns]),
                    json.dumps(table.preview_rows),
                    table.summary,
                    json.dumps(table.keywords),
                    table.table_search_text,
                    json.dumps(table.warnings),
                ),
            )

    def save_error(
        self,
        *,
        source_id: str,
        relative_path: str,
        filename: str,
        file_format: str,
        size_bytes: int,
        last_modified: datetime | None,
        error_message: str,
        indexed_at: datetime,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO database_files (
                source_id, relative_path, filename, file_format,
                size_bytes, last_modified, status, error_message,
                first_indexed_at, last_indexed_at, is_present
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (relative_path) DO UPDATE SET
                source_id = EXCLUDED.source_id,
                filename = EXCLUDED.filename,
                file_format = EXCLUDED.file_format,
                size_bytes = EXCLUDED.size_bytes,
                last_modified = EXCLUDED.last_modified,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                last_indexed_at = EXCLUDED.last_indexed_at,
                is_present = EXCLUDED.is_present,
                updated_at = NOW()
            """,
            (
                source_id,
                relative_path,
                filename,
                file_format,
                size_bytes,
                last_modified,
                "error",
                error_message,
                indexed_at,
                indexed_at,
                True,
            ),
        )

        self._connection.execute(
            "DELETE FROM database_tables WHERE source_id = %s",
            (source_id,),
        )

    def mark_missing(self, prefix: str, indexed_at: datetime) -> None:
        if prefix:
            self._connection.execute(
                """
                UPDATE database_files
                SET is_present = FALSE,
                    status = %s,
                    updated_at = NOW()
                WHERE (
                    relative_path = %s
                    OR starts_with(relative_path, %s)
                )
                  AND last_indexed_at < %s
                  AND is_present = TRUE
                """,
                (
                    FileStatus.MISSING.value,
                    prefix,
                    f"{prefix}/",
                    indexed_at,
                ),
            )
            return

        self._connection.execute(
            """
            UPDATE database_files
            SET is_present = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE last_indexed_at < %s
              AND is_present = TRUE
            """,
            (FileStatus.MISSING.value, indexed_at),
        )
