from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field
from sqlalchemy.exc import ProgrammingError

from lake_agent.config import EmbeddingSettings, LocalSettings, PostgresSettings
from lake_agent.indexing.tabular.vector_store import build_pgvector_store
from lake_agent.persistence.database import PostgresDatabase

ModalityName = Literal[
    "tabular",
    "text",
    "web",
    "document",
    "slideshow",
    "image",
    "audio",
    "video",
]


@dataclass(frozen=True, slots=True)
class RetrievalTableNames:
    tabular: str = "tabular_index"
    text: str = "text_index"
    web: str = "web_index"
    document: str = "document_index"
    slideshow: str = "slideshow_index"
    image: str = "image_index"
    audio: str = "audio_index"
    video: str = "video_index"


class SearchArgs(BaseModel):
    query: str = Field(..., description="Natural-language query to search for.")
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of nearest results to return in this page.",
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="Zero-based offset into the similarity-ranked result list.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Backward-compatible alias for limit. Prefer `limit`.",
    )


class FileSummaryArgs(BaseModel):
    file_path: str = Field(
        ...,
        description="Indexed relative path, or an absolute path under the data lake.",
    )


class IndexedDataRetriever:
    def __init__(
        self,
        connection: Any,
        *,
        vector_stores: dict[ModalityName, Any],
        datalake_dir: str | None = None,
    ) -> None:
        self._connection = connection
        self._vector_stores = vector_stores
        self._datalake_dir = Path(datalake_dir).expanduser().resolve() if datalake_dir else None

    @classmethod
    def from_env(
        cls,
        *,
        table_names: RetrievalTableNames | None = None,
        postgres_settings: PostgresSettings | None = None,
        embedding_settings: EmbeddingSettings | None = None,
        local_settings: LocalSettings | None = None,
    ) -> "IndexedDataRetriever":
        table_names = table_names or RetrievalTableNames()
        postgres_settings = postgres_settings or PostgresSettings.from_env()
        local_settings = local_settings or LocalSettings.from_env()

        database = PostgresDatabase(postgres_settings.dsn)
        connection = database.connect()
        vector_stores: dict[ModalityName, Any] = {
            "tabular": build_pgvector_store(
                table_names.tabular,
                embedding_settings=embedding_settings,
                postgres_settings=postgres_settings,
            ),
            "text": build_pgvector_store(
                table_names.text,
                embedding_settings=embedding_settings,
                postgres_settings=postgres_settings,
            ),
            "web": build_pgvector_store(
                table_names.web,
                embedding_settings=embedding_settings,
                postgres_settings=postgres_settings,
            ),
            "document": build_pgvector_store(
                table_names.document,
                embedding_settings=embedding_settings,
                postgres_settings=postgres_settings,
            ),
            "slideshow": build_pgvector_store(
                table_names.slideshow,
                embedding_settings=embedding_settings,
                postgres_settings=postgres_settings,
            ),
            "image": build_pgvector_store(
                table_names.image,
                embedding_settings=embedding_settings,
                postgres_settings=postgres_settings,
            ),
            "audio": build_pgvector_store(
                table_names.audio,
                embedding_settings=embedding_settings,
                postgres_settings=postgres_settings,
            ),
            "video": build_pgvector_store(
                table_names.video,
                embedding_settings=embedding_settings,
                postgres_settings=postgres_settings,
            ),
        }
        return cls(
            connection,
            vector_stores=vector_stores,
            datalake_dir=local_settings.datalake_dir,
        )

    def close(self) -> None:
        close = getattr(self._connection, "close", None)
        if callable(close):
            close()

    def query_tabular(self, query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
        return self._query_modality("tabular", query, limit, offset)

    def query_text(self, query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
        return self._query_modality("text", query, limit, offset)

    def query_web(self, query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
        return self._query_modality("web", query, limit, offset)

    def query_document(self, query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
        return self._query_modality("document", query, limit, offset)

    def query_slideshow(self, query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
        return self._query_modality("slideshow", query, limit, offset)

    def query_image(self, query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
        return self._query_modality("image", query, limit, offset)

    def query_audio(self, query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
        return self._query_modality("audio", query, limit, offset)

    def query_video(self, query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
        return self._query_modality("video", query, limit, offset)

    def query_all(self, query: str, limit: int = 5, offset: int = 0) -> dict[str, Any]:
        combined: list[dict[str, Any]] = []
        skipped_modalities: list[dict[str, str]] = []
        for modality in (
            "tabular",
            "text",
            "web",
            "document",
            "slideshow",
            "image",
            "audio",
            "video",
        ):
            try:
                modality_results = self._query_modality(modality, query, limit + offset, 0)["results"]
            except Exception as exc:
                if not _is_missing_vector_table_error(exc):
                    raise
                skipped_modalities.append(
                    {
                        "modality": modality,
                        "reason": "vector_table_missing",
                    }
                )
                continue
            combined.extend(modality_results)
        combined.sort(key=lambda item: item["score"])
        paged_results = combined[offset : offset + limit]
        response = {
            "query": query,
            "limit": limit,
            "offset": offset,
            "returned_count": len(paged_results),
            "next_offset": offset + len(paged_results),
            "has_more": len(combined) > offset + len(paged_results),
            "results": paged_results,
        }
        if skipped_modalities:
            response["skipped_modalities"] = skipped_modalities
        return response

    def get_file_summary(self, file_path: str) -> dict[str, Any]:
        candidates = self._candidate_file_paths(file_path)
        for candidate in candidates:
            for modality in (
                "tabular",
                "text",
                "web",
                "document",
                "slideshow",
                "image",
                "audio",
                "video",
            ):
                loader = getattr(self, f"_load_{modality}_file_by_path")
                row = loader(candidate)
                if row is not None:
                    return self._format_file_summary(modality, row)
        raise ValueError(f"No indexed file found for path: {file_path}")

    def _query_modality(
        self,
        modality: ModalityName,
        query: str,
        limit: int,
        offset: int = 0,
    ) -> dict[str, Any]:
        vector_store = self._vector_stores[modality]
        requested = limit + offset + 1
        raw_hits = vector_store.similarity_search_with_score(query, k=requested)
        results = [
            self._hydrate_hit(modality, document.metadata, score)
            for document, score in raw_hits
        ]
        paged_results = results[offset : offset + limit]
        return {
            "modality": modality,
            "query": query,
            "limit": limit,
            "offset": offset,
            "returned_count": len(paged_results),
            "next_offset": offset + len(paged_results),
            "has_more": len(results) > offset + len(paged_results),
            "results": paged_results,
        }

    def _hydrate_hit(
        self,
        modality: ModalityName,
        metadata: dict[str, Any],
        score: float,
    ) -> dict[str, Any]:
        record_type = str(metadata.get("record_type") or "")
        if modality == "tabular":
            if record_type == "file":
                row = self._load_tabular_file(str(metadata["source_id"]))
                return self._format_tabular_file_hit(row, score)
            row = self._load_tabular_table(str(metadata["table_id"]))
            return self._format_tabular_table_hit(row, score)

        if modality == "text":
            if record_type == "file":
                row = self._load_text_file(str(metadata["source_id"]))
                return self._format_text_file_hit(row, score)
            row = self._load_text_section(str(metadata["section_id"]))
            return self._format_text_section_hit(row, score)

        if modality == "web":
            if record_type == "file":
                row = self._load_web_file(str(metadata["source_id"]))
                return self._format_web_file_hit(row, score)
            row = self._load_web_section(str(metadata["section_id"]))
            return self._format_web_section_hit(row, score)

        if modality == "document":
            if record_type == "file":
                row = self._load_document_file(str(metadata["source_id"]))
                return self._format_document_file_hit(row, score)
            row = self._load_document_section(str(metadata["section_id"]))
            return self._format_document_section_hit(row, score)

        if modality == "slideshow":
            if record_type == "file":
                row = self._load_slideshow_file(str(metadata["source_id"]))
                return self._format_slideshow_file_hit(row, score)
            row = self._load_slideshow_section(str(metadata["section_id"]))
            return self._format_slideshow_section_hit(row, score)

        if modality == "image":
            if record_type == "file":
                row = self._load_image_file(str(metadata["source_id"]))
                return self._format_image_file_hit(row, score)
            row = self._load_image_section(str(metadata["section_id"]))
            return self._format_image_section_hit(row, score)

        if modality == "audio":
            if record_type == "file":
                row = self._load_audio_file(str(metadata["source_id"]))
                return self._format_audio_file_hit(row, score)
            row = self._load_audio_section(str(metadata["section_id"]))
            return self._format_audio_section_hit(row, score)

        if modality == "video":
            if record_type == "file":
                row = self._load_video_file(str(metadata["source_id"]))
                return self._format_video_file_hit(row, score)
            row = self._load_video_section(str(metadata["section_id"]))
            return self._format_video_section_hit(row, score)

        raise ValueError(f"Unsupported modality: {modality}")

    def _load_tabular_file(self, source_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, file_summary,
                   file_keywords, parse_warnings, workbook_sheet_descriptions
            FROM tabular_files
            WHERE source_id = %s
            """,
            (source_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing tabular file row for source_id={source_id}")
        return row

    def _load_tabular_table(self, table_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT t.table_id, t.source_id, f.relative_path, f.filename, f.file_format,
                   t.table_name, t.sheet_name, t.sheet_description, t.header_row_index,
                   t.raw_header,
                   t.row_count, t.column_count, t.summary, t.keywords, t.columns_json,
                   t.preview_rows, t.warnings
            FROM tabular_tables AS t
            JOIN tabular_files AS f ON f.source_id = t.source_id
            WHERE t.table_id = %s
            """,
            (table_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing tabular table row for table_id={table_id}")
        return row

    def _load_text_file(self, source_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, file_summary,
                   file_keywords, parse_warnings
            FROM text_files
            WHERE source_id = %s
            """,
            (source_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing text file row for source_id={source_id}")
        return row

    def _load_text_section(self, section_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT s.section_id, s.source_id, f.relative_path, f.filename, f.file_format,
                   s.chunk_index, s.heading, s.content, s.search_text, s.line_start, s.line_end,
                   s.char_count, s.warnings
            FROM text_sections AS s
            JOIN text_files AS f ON f.source_id = s.source_id
            WHERE s.section_id = %s
            """,
            (section_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing text section row for section_id={section_id}")
        return row

    def _load_document_file(self, source_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, file_summary,
                   file_keywords, parse_warnings
            FROM document_files
            WHERE source_id = %s
            """,
            (source_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing document file row for source_id={source_id}")
        return row

    def _load_web_file(self, source_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, file_summary,
                   file_keywords, parse_warnings
            FROM web_files
            WHERE source_id = %s
            """,
            (source_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing web file row for source_id={source_id}")
        return row

    def _load_web_section(self, section_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT s.section_id, s.source_id, f.relative_path, f.filename, f.file_format,
                   s.chunk_index, s.heading, s.content, s.search_text, s.line_start, s.line_end,
                   s.char_count, s.warnings
            FROM web_sections AS s
            JOIN web_files AS f ON f.source_id = s.source_id
            WHERE s.section_id = %s
            """,
            (section_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing web section row for section_id={section_id}")
        return row

    def _load_document_section(self, section_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT s.section_id, s.source_id, f.relative_path, f.filename, f.file_format,
                   s.section_type, s.chunk_index, s.heading, s.content, s.search_text, s.page_start,
                   s.page_end, s.char_count, s.image_id, s.image_index, s.warnings
            FROM document_sections AS s
            JOIN document_files AS f ON f.source_id = s.source_id
            WHERE s.section_id = %s
            """,
            (section_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing document section row for section_id={section_id}")
        return row

    def _load_slideshow_file(self, source_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, file_summary,
                   file_keywords, parse_warnings
            FROM slideshow_files
            WHERE source_id = %s
            """,
            (source_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing slideshow file row for source_id={source_id}")
        return row

    def _load_slideshow_section(self, section_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT s.section_id, s.source_id, f.relative_path, f.filename, f.file_format,
                   s.section_type, s.chunk_index, s.heading, s.content, s.search_text, s.slide_start,
                   s.slide_end, s.char_count, s.image_id, s.image_index, s.warnings
            FROM slideshow_sections AS s
            JOIN slideshow_files AS f ON f.source_id = s.source_id
            WHERE s.section_id = %s
            """,
            (section_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing slideshow section row for section_id={section_id}")
        return row

    def _load_image_file(self, source_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, width, height,
                   color_mode, has_alpha, is_animated, frame_count, file_summary,
                   file_keywords, parse_warnings
            FROM image_files
            WHERE source_id = %s
            """,
            (source_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing image file row for source_id={source_id}")
        return row

    def _load_image_section(self, section_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT s.section_id, s.source_id, f.relative_path, f.filename, f.file_format,
                   s.section_type, s.chunk_index, s.heading, s.content, s.search_text, s.line_start,
                   s.line_end, s.char_count, s.warnings, f.width, f.height, f.color_mode
            FROM image_sections AS s
            JOIN image_files AS f ON f.source_id = s.source_id
            WHERE s.section_id = %s
            """,
            (section_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing image section row for section_id={section_id}")
        return row

    def _load_audio_file(self, source_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, duration_seconds,
                   codec_name, sample_rate, channels, asr_model_name, asr_cost_usd,
                   file_summary, file_keywords, parse_warnings
            FROM audio_files
            WHERE source_id = %s
            """,
            (source_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing audio file row for source_id={source_id}")
        return row

    def _load_audio_section(self, section_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT s.section_id, s.source_id, f.relative_path, f.filename, f.file_format,
                   s.chunk_index, s.start_seconds, s.end_seconds, s.content,
                   s.search_text, s.char_count, s.warnings, f.duration_seconds,
                   f.asr_model_name
            FROM audio_sections AS s
            JOIN audio_files AS f ON f.source_id = s.source_id
            WHERE s.section_id = %s
            """,
            (section_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing audio section row for section_id={section_id}")
        return row

    def _load_video_file(self, source_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, duration_seconds,
                   width, height, fps, video_codec, audio_codec, has_audio,
                   sampled_frame_count, asr_model_name, asr_cost_usd, vl_model_name,
                   file_summary, file_keywords, parse_warnings
            FROM video_files
            WHERE source_id = %s
            """,
            (source_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing video file row for source_id={source_id}")
        return row

    def _load_video_section(self, section_id: str) -> dict[str, Any]:
        row = self._connection.execute(
            """
            SELECT s.section_id, s.source_id, f.relative_path, f.filename, f.file_format,
                   s.section_type, s.chunk_index, s.timestamp_seconds, s.start_seconds,
                   s.end_seconds, s.frame_index, s.content, s.search_text,
                   s.char_count, s.warnings, f.duration_seconds, f.width, f.height,
                   f.asr_model_name, f.vl_model_name
            FROM video_sections AS s
            JOIN video_files AS f ON f.source_id = s.source_id
            WHERE s.section_id = %s
            """,
            (section_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Missing video section row for section_id={section_id}")
        return row

    def _load_tabular_file_by_path(self, relative_path: str) -> dict[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, file_summary,
                   file_keywords, parse_warnings, workbook_sheet_descriptions
            FROM tabular_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def _load_text_file_by_path(self, relative_path: str) -> dict[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, file_summary,
                   file_keywords, parse_warnings
            FROM text_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def _load_document_file_by_path(self, relative_path: str) -> dict[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, file_summary,
                   file_keywords, parse_warnings
            FROM document_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def _load_web_file_by_path(self, relative_path: str) -> dict[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, file_summary,
                   file_keywords, parse_warnings
            FROM web_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def _load_slideshow_file_by_path(self, relative_path: str) -> dict[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, file_summary,
                   file_keywords, parse_warnings
            FROM slideshow_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def _load_image_file_by_path(self, relative_path: str) -> dict[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, width, height,
                   color_mode, has_alpha, is_animated, frame_count, file_summary,
                   file_keywords, parse_warnings
            FROM image_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def _load_audio_file_by_path(self, relative_path: str) -> dict[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, duration_seconds,
                   codec_name, sample_rate, channels, asr_model_name, asr_cost_usd,
                   file_summary, file_keywords, parse_warnings
            FROM audio_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def _load_video_file_by_path(self, relative_path: str) -> dict[str, Any] | None:
        return self._connection.execute(
            """
            SELECT source_id, relative_path, filename, file_format, duration_seconds,
                   width, height, fps, video_codec, audio_codec, has_audio,
                   sampled_frame_count, asr_model_name, asr_cost_usd, vl_model_name,
                   file_summary, file_keywords, parse_warnings
            FROM video_files
            WHERE relative_path = %s
            """,
            (relative_path,),
        ).fetchone()

    def _format_tabular_file_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "tabular",
            "record_type": "file",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row["file_summary"],
            "sheet_descriptions": row["workbook_sheet_descriptions"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_tabular_table_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "tabular",
            "record_type": "table",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row["summary"],
            "table_name": row["table_name"],
            "sheet_name": row["sheet_name"],
            "sheet_description": row["sheet_description"],
            "header": row["raw_header"],
            "columns": row["columns_json"],
            "preview_rows": row["preview_rows"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_text_file_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "text",
            "record_type": "file",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row["file_summary"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_text_section_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "text",
            "record_type": "section",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row.get("search_text") or row["content"],
        }
        position = _build_position(row, start_key="line_start", end_key="line_end", unit="line")
        if position is not None:
            hit["position"] = position
        self._attach_absolute_file_path(hit)
        return hit

    def _format_web_file_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "web",
            "record_type": "file",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row["file_summary"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_web_section_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "web",
            "record_type": "section",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row.get("search_text") or row["content"],
        }
        position = _build_position(row, start_key="line_start", end_key="line_end", unit="line")
        if position is not None:
            hit["position"] = position
        self._attach_absolute_file_path(hit)
        return hit

    def _format_document_file_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "document",
            "record_type": "file",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row["file_summary"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_document_section_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "document",
            "record_type": "section",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row.get("search_text") or row["content"],
        }
        position = _build_position(
            row,
            start_key="page_start",
            end_key="page_end",
            unit="page",
            extra_keys=("image_index",),
        )
        if position is not None:
            hit["position"] = position
        self._attach_absolute_file_path(hit)
        return hit

    def _format_slideshow_file_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "slideshow",
            "record_type": "file",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row["file_summary"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_slideshow_section_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "slideshow",
            "record_type": "section",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row.get("search_text") or row["content"],
        }
        position = _build_position(
            row,
            start_key="slide_start",
            end_key="slide_end",
            unit="slide",
            extra_keys=("image_index",),
        )
        if position is not None:
            hit["position"] = position
        self._attach_absolute_file_path(hit)
        return hit

    def _format_image_file_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "image",
            "record_type": "file",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row["file_summary"],
            "width": row["width"],
            "height": row["height"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_image_section_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "image",
            "record_type": "section",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row.get("search_text") or row["content"],
            "width": row["width"],
            "height": row["height"],
        }
        position = _build_position(row, start_key="line_start", end_key="line_end", unit="line")
        if position is not None:
            hit["position"] = position
        self._attach_absolute_file_path(hit)
        return hit

    def _format_audio_file_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "audio",
            "record_type": "file",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row["file_summary"],
            "duration_seconds": row["duration_seconds"],
            "asr_model_name": row["asr_model_name"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_audio_section_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "audio",
            "record_type": "section",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row.get("search_text") or row["content"],
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "duration_seconds": row["duration_seconds"],
            "asr_model_name": row["asr_model_name"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_video_file_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "video",
            "record_type": "file",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row["file_summary"],
            "duration_seconds": row["duration_seconds"],
            "width": row["width"],
            "height": row["height"],
            "asr_model_name": row["asr_model_name"],
            "vl_model_name": row["vl_model_name"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_video_section_hit(self, row: dict[str, Any], score: float) -> dict[str, Any]:
        hit = {
            "modality": "video",
            "record_type": "section",
            "file_path": row["relative_path"],
            "score": float(score),
            "content": row.get("search_text") or row["content"],
            "section_type": row["section_type"],
            "timestamp_seconds": row["timestamp_seconds"],
            "start_seconds": row["start_seconds"],
            "end_seconds": row["end_seconds"],
            "frame_index": row["frame_index"],
            "duration_seconds": row["duration_seconds"],
            "width": row["width"],
            "height": row["height"],
            "asr_model_name": row["asr_model_name"],
            "vl_model_name": row["vl_model_name"],
        }
        self._attach_absolute_file_path(hit)
        return hit

    def _format_file_summary(self, modality: ModalityName, row: dict[str, Any]) -> dict[str, Any]:
        base = {
            "modality": modality,
            "file_path": row["relative_path"],
            "filename": row["filename"],
            "file_format": row["file_format"],
            "summary": row.get("file_summary"),
            "keywords": row.get("file_keywords", []),
            "warnings": row.get("parse_warnings", []),
        }
        if modality == "tabular":
            base["sheet_descriptions"] = row.get("workbook_sheet_descriptions", {})
        if modality == "image":
            base["width"] = row["width"]
            base["height"] = row["height"]
            base["color_mode"] = row["color_mode"]
            base["has_alpha"] = row["has_alpha"]
            base["is_animated"] = row["is_animated"]
            base["frame_count"] = row["frame_count"]
        if modality == "audio":
            base["duration_seconds"] = row["duration_seconds"]
            base["codec_name"] = row["codec_name"]
            base["sample_rate"] = row["sample_rate"]
            base["channels"] = row["channels"]
            base["asr_model_name"] = row["asr_model_name"]
            base["asr_cost_usd"] = row["asr_cost_usd"]
        if modality == "video":
            base["duration_seconds"] = row["duration_seconds"]
            base["width"] = row["width"]
            base["height"] = row["height"]
            base["fps"] = row["fps"]
            base["video_codec"] = row["video_codec"]
            base["audio_codec"] = row["audio_codec"]
            base["has_audio"] = row["has_audio"]
            base["sampled_frame_count"] = row["sampled_frame_count"]
            base["asr_model_name"] = row["asr_model_name"]
            base["asr_cost_usd"] = row["asr_cost_usd"]
            base["vl_model_name"] = row["vl_model_name"]
        self._attach_absolute_file_path(base)
        return base

    def _normalize_file_path(self, file_path: str) -> str:
        candidate = Path(file_path).expanduser()
        if candidate.is_absolute() and self._datalake_dir is not None:
            try:
                return candidate.resolve().relative_to(self._datalake_dir).as_posix()
            except ValueError:
                pass
        return file_path.replace("\\", "/")

    def _candidate_file_paths(self, file_path: str) -> list[str]:
        normalized = self._normalize_file_path(file_path)
        candidates: list[str] = []
        for candidate in (
            normalized,
            normalized.lstrip("/"),
        ):
            if candidate and candidate not in candidates:
                candidates.append(candidate)
        return candidates

    def _attach_absolute_file_path(self, payload: dict[str, Any]) -> None:
        relative_path = payload.get("file_path")
        if not relative_path:
            return
        payload["execution_file_path"] = f"./{str(relative_path).lstrip('./')}"
        if self._datalake_dir is None:
            return
        payload["absolute_file_path"] = str(
            (self._datalake_dir / str(relative_path)).resolve()
        )


def build_langchain_retrieval_tools(
    retriever: IndexedDataRetriever,
) -> list[Any]:
    from langchain_core.tools import StructuredTool

    def _search(
        fn: Callable[[str, int, int], dict[str, Any]],
    ) -> Callable[[str, int, int, int | None], dict[str, Any]]:
        def run(
            query: str,
            limit: int = 5,
            offset: int = 0,
            top_k: int | None = None,
        ) -> dict[str, Any]:
            effective_limit = top_k if top_k is not None else limit
            return fn(query, effective_limit, offset)

        return run

    tools = [
        StructuredTool.from_function(
            func=_search(retriever.query_tabular),
            name="search_tabular_data",
            description=(
                "Search indexed tabular data such as CSV/XLS/XLSX/TSV files and return the "
                "nearest file or table hits with true content, metadata, and score."
            ),
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            func=_search(retriever.query_text),
            name="search_text_data",
            description=(
                "Search indexed text files such as TXT/MD and return the nearest file or "
                "section hits with true content, metadata, and score."
            ),
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            func=_search(retriever.query_web),
            name="search_web_data",
            description=(
                "Search indexed web files such as HTML/HTM and return the nearest file or "
                "section hits with true content, metadata, and score."
            ),
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            func=_search(retriever.query_document),
            name="search_document_data",
            description=(
                "Search indexed document files such as PDF/DOCX/DOC/RTF and return the nearest "
                "file or section hits with true content, metadata, and score."
            ),
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            func=_search(retriever.query_slideshow),
            name="search_slideshow_data",
            description=(
                "Search indexed slideshow files such as PPT/PPTX and return the nearest file or "
                "section hits with true content, metadata, and score."
            ),
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            func=_search(retriever.query_image),
            name="search_image_data",
            description=(
                "Search indexed image files and return the nearest file or OCR/summary section hits "
                "with true content, metadata, and score."
            ),
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            func=_search(retriever.query_audio),
            name="search_audio_data",
            description=(
                "Search indexed audio transcripts such as MP3/WAV/M4A/FLAC/OGG files and return "
                "nearest file or transcript section hits with true content, timestamps, metadata, and score."
            ),
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            func=_search(retriever.query_video),
            name="search_video_data",
            description=(
                "Search indexed video transcripts and sampled frame captions, returning nearest "
                "file or section hits with true content, timestamps, metadata, and score."
            ),
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            func=_search(retriever.query_all),
            name="search_all_indexed_data",
            description=(
                "Search across all indexed modalities and return the globally nearest results with "
                "their modality, file path, true content, metadata, and score."
            ),
            args_schema=SearchArgs,
        ),
        StructuredTool.from_function(
            func=retriever.get_file_summary,
            name="get_indexed_file_summary",
            description=(
                "Get the stored summary and useful metadata for one indexed file path across all "
                "supported modalities."
            ),
            args_schema=FileSummaryArgs,
        ),
    ]
    return tools


def _build_position(
    row: dict[str, Any],
    *,
    start_key: str,
    end_key: str,
    unit: str,
    extra_keys: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    position: dict[str, Any] = {"unit": unit}
    start_value = row.get(start_key)
    end_value = row.get(end_key)
    if start_value is not None:
        position["start"] = start_value
    if end_value is not None:
        position["end"] = end_value
    for key in extra_keys:
        value = row.get(key)
        if value is not None:
            position[key] = value
    if len(position) == 1:
        return None
    return position


def _is_missing_vector_table_error(exc: Exception) -> bool:
    if isinstance(exc, ProgrammingError):
        message = str(exc).lower()
        return "does not exist" in message and "index" in message
    message = str(exc).lower()
    return "does not exist" in message and "index" in message
