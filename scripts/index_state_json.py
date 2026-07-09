from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lake_agent.config import EmbeddingSettings, LocalSettings, PostgresSettings
from lake_agent.indexing.tabular.vector_store import build_pgvector_store
from lake_agent.persistence.database import PostgresDatabase

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover
    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: dict[str, Any] = field(default_factory=dict)

JsonFormat = Literal["json", "jsonl", "ndjson"]
SUPPORTED_FORMATS = {"json", "jsonl", "ndjson"}
SCALAR_TYPES = (str, int, float, bool, type(None))
DEFAULT_RELATIVE_JSON_PATH = (
    "[iSE Summer Challenge 2026] Data Lake/wildfire/"
    "state_abbreviation_to_state.json"
)
DEFAULT_DATALAKE_DIR = Path(r"D:\data\Data-Lake")


@dataclass(slots=True)
class JsonSection:
    section_id: str
    chunk_index: int
    path_start: str | None
    path_end: str | None
    entry_count: int
    content: str
    char_count: int
    search_text: str
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class JsonIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: JsonFormat
    sections: list[JsonSection]
    parser_version: str = "direct_flattened_json_v1"
    parse_warnings: list[str] = field(default_factory=list)
    top_level_type: str | None = None
    entry_count: int = 0
    max_depth: int = 0
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None


@dataclass(frozen=True, slots=True)
class FlattenedEntry:
    path: str
    value: str
    depth: int

    @property
    def text(self) -> str:
        return f"{self.path}: {self.value}"


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)

    datalake_dir = _resolve_datalake_dir(args.datalake_dir).resolve()
    file_path = _resolve_json_file(args.file, datalake_dir).resolve()
    if not file_path.exists() or not file_path.is_file():
        print(f"JSON file not found: {file_path}", file=sys.stderr)
        return 1

    try:
        relative_path = file_path.relative_to(datalake_dir).as_posix()
    except ValueError:
        print(
            f"JSON file must be under datalake dir.\nFile: {file_path}\nDatalake: {datalake_dir}",
            file=sys.stderr,
        )
        return 1

    try:
        postgres_settings = PostgresSettings.from_env()
        embedding_settings = None if args.no_vector else EmbeddingSettings.from_env()
        result = parse_json_file(
            file_path,
            relative_path=relative_path,
            max_chars_per_section=args.max_chars_per_section,
        )

        database = PostgresDatabase(postgres_settings.dsn)
        with database.connect() as connection:
            database.initialize(connection)
            save_json_result(connection, result, file_path)

            vector_document_count = 0
            if embedding_settings is not None:
                _ensure_windows_selector_event_loop_policy()
                vector_store = build_pgvector_store(
                    args.table_name,
                    embedding_settings=embedding_settings,
                    postgres_settings=postgres_settings,
                )
                documents = build_documents(result)
                vector_store.add_documents(
                    documents=documents,
                    ids=[_document_id(document) for document in documents],
                )
                vector_document_count = len(documents)
    except Exception as exc:
        print(f"Direct JSON indexing failed: {exc}", file=sys.stderr)
        return 1

    print(f"Indexed JSON: {relative_path}")
    print(f"Absolute File: {file_path}")
    print(f"Top-level type: {result.top_level_type}")
    print(f"Flattened entries: {result.entry_count}")
    print(f"Sections: {len(result.sections)}")
    print(f"Vector Documents: {vector_document_count}")
    print(f"Vector Table: {args.table_name if not args.no_vector else '<skipped>'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Directly persist and embed one JSON file for QA retrieval."
    )
    parser.add_argument(
        "--file",
        default=None,
        help=(
            "JSON file path. Defaults to STATE_JSON_FILE from .env, then "
            "DATALAKE_DIR/[iSE Summer Challenge 2026] Data Lake/wildfire/"
            "state_abbreviation_to_state.json."
        ),
    )
    parser.add_argument(
        "--datalake-dir",
        default=None,
        help="Data lake root. Defaults to DATALAKE_DIR from .env, then D:\\data\\Data-Lake.",
    )
    parser.add_argument(
        "--table-name",
        default=os.getenv("STATE_JSON_VECTOR_TABLE", "json_index"),
        help="PGVector table name for JSON embeddings.",
    )
    parser.add_argument(
        "--max-chars-per-section",
        type=int,
        default=2400,
        help="Maximum flattened JSON characters per embedded section.",
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Persist JSON metadata/sections only; skip embedding.",
    )
    return parser


def parse_json_file(
    file_path: Path,
    *,
    relative_path: str,
    max_chars_per_section: int,
) -> JsonIndexResult:
    extension = file_path.suffix.lower().removeprefix(".")
    if extension not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported JSON format: {extension}")
    if max_chars_per_section <= 0:
        raise ValueError("max_chars_per_section must be positive")

    warnings: list[str] = []
    payload = _load_payload(file_path, extension, warnings)
    entries = list(_flatten_value(payload, path="$", depth=0))
    if not entries:
        warnings.append("The file did not contain any readable JSON values.")

    source_id = _stable_source_id(relative_path)
    sections = _build_sections(
        entries,
        source_id=source_id,
        max_chars_per_section=max_chars_per_section,
    )
    result = JsonIndexResult(
        source_id=source_id,
        relative_path=relative_path,
        filename=file_path.name,
        file_format=extension,  # type: ignore[arg-type]
        sections=sections,
        parse_warnings=warnings,
        top_level_type=_json_type_name(payload),
        entry_count=len(entries),
        max_depth=max((entry.depth for entry in entries), default=0),
    )
    result.file_search_text = _build_file_search_text(result)
    return result


def _load_payload(path: Path, extension: str, warnings: list[str]) -> Any:
    if extension == "json":
        return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))

    records: list[Any] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON record on line {line_number}: {exc.msg}") from exc
    if not records:
        warnings.append("JSON lines file did not contain any non-empty records.")
    return records


def _flatten_value(value: Any, *, path: str, depth: int) -> Iterable[FlattenedEntry]:
    if isinstance(value, dict):
        yield FlattenedEntry(path=path, value=f"object with {len(value)} key(s)", depth=depth)
        for key in sorted(value):
            yield from _flatten_value(
                value[key],
                path=f"{path}.{_escape_key(str(key))}",
                depth=depth + 1,
            )
        return

    if isinstance(value, list):
        yield FlattenedEntry(path=path, value=f"array with {len(value)} item(s)", depth=depth)
        for index, item in enumerate(value):
            yield from _flatten_value(item, path=f"{path}[{index}]", depth=depth + 1)
        return

    if not isinstance(value, SCALAR_TYPES):
        value = str(value)
    yield FlattenedEntry(path=path, value=_format_scalar(value), depth=depth)


def _build_sections(
    entries: list[FlattenedEntry],
    *,
    source_id: str,
    max_chars_per_section: int,
) -> list[JsonSection]:
    sections: list[JsonSection] = []
    current: list[FlattenedEntry] = []
    current_chars = 0

    for entry in entries:
        entry_chars = len(entry.text) + 1
        if current and current_chars + entry_chars > max_chars_per_section:
            sections.append(_build_section(source_id, len(sections) + 1, current))
            current = []
            current_chars = 0
        current.append(entry)
        current_chars += entry_chars

    if current:
        sections.append(_build_section(source_id, len(sections) + 1, current))
    return sections


def _build_section(
    source_id: str,
    chunk_index: int,
    entries: list[FlattenedEntry],
) -> JsonSection:
    content = "\n".join(entry.text for entry in entries).strip()
    path_start = entries[0].path if entries else None
    path_end = entries[-1].path if entries else None
    return JsonSection(
        section_id=_stable_id(f"{source_id}:{chunk_index}:{path_start or ''}:{path_end or ''}", "section"),
        chunk_index=chunk_index,
        path_start=path_start,
        path_end=path_end,
        entry_count=len(entries),
        content=content,
        char_count=len(content),
        search_text=content,
    )


def save_json_result(connection: Any, result: JsonIndexResult, file_path: Path) -> None:
    stat = file_path.stat()
    indexed_at = datetime.now(timezone.utc)
    last_modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    connection.execute(
        """
        INSERT INTO json_files (
            source_id, relative_path, filename, file_format,
            size_bytes, last_modified, parser_version, parse_warnings,
            top_level_type, entry_count, max_depth,
            file_summary, file_keywords, file_search_text, status, error_message,
            first_indexed_at, last_indexed_at, is_present
        ) VALUES (
            %s, %s, %s, %s,
            %s, %s, %s, %s::jsonb,
            %s, %s, %s,
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
            top_level_type = EXCLUDED.top_level_type,
            entry_count = EXCLUDED.entry_count,
            max_depth = EXCLUDED.max_depth,
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
            stat.st_size,
            last_modified,
            result.parser_version,
            json.dumps(result.parse_warnings),
            result.top_level_type,
            result.entry_count,
            result.max_depth,
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
    connection.execute("DELETE FROM json_sections WHERE source_id = %s", (result.source_id,))
    for section in result.sections:
        connection.execute(
            """
            INSERT INTO json_sections (
                section_id, source_id, chunk_index,
                path_start, path_end, entry_count,
                content, char_count, search_text, warnings
            ) VALUES (
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s::jsonb
            )
            """,
            (
                section.section_id,
                result.source_id,
                section.chunk_index,
                section.path_start,
                section.path_end,
                section.entry_count,
                section.content,
                section.char_count,
                section.search_text,
                json.dumps(section.warnings),
            ),
        )


def build_documents(result: JsonIndexResult) -> list[Document]:
    documents: list[Document] = []
    if result.file_search_text:
        documents.append(
            Document(
                page_content=result.file_search_text,
                metadata={
                    "record_type": "file",
                    "source_id": result.source_id,
                    "relative_path": result.relative_path,
                    "filename": result.filename,
                    "file_format": result.file_format,
                    "top_level_type": result.top_level_type,
                    "entry_count": result.entry_count,
                    "max_depth": result.max_depth,
                },
            )
        )

    for section in result.sections:
        if not section.search_text.strip():
            continue
        documents.append(
            Document(
                page_content=section.search_text,
                metadata={
                    "record_type": "section",
                    "source_id": result.source_id,
                    "relative_path": result.relative_path,
                    "filename": result.filename,
                    "file_format": result.file_format,
                    "section_id": section.section_id,
                    "chunk_index": section.chunk_index,
                    "path_start": section.path_start,
                    "path_end": section.path_end,
                    "entry_count": section.entry_count,
                },
            )
        )
    return documents


def _document_id(document: Document) -> str:
    metadata = document.metadata
    if metadata.get("record_type") == "file":
        return f"{metadata['source_id']}:file"
    return f"{metadata['source_id']}:section_{metadata['chunk_index']}"


def _resolve_json_file(value: str | None, datalake_dir: Path) -> Path:
    configured = value or os.getenv("STATE_JSON_FILE")
    if configured:
        return Path(configured).expanduser()
    return datalake_dir / DEFAULT_RELATIVE_JSON_PATH


def _resolve_datalake_dir(value: str | None) -> Path:
    if value:
        return Path(value).expanduser()
    try:
        return Path(LocalSettings.from_env().datalake_dir).expanduser()
    except Exception:
        return DEFAULT_DATALAKE_DIR


def _load_dotenv() -> None:
    dotenv_path = PROJECT_ROOT / ".env"
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        _load_dotenv_fallback(dotenv_path)
        return

    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(discovered, override=True)
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=True)


def _load_dotenv_fallback(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value


def _format_scalar(value: Any, max_chars: int = 1000) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip() if isinstance(value, str) else str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated]"


def _escape_key(key: str) -> str:
    if key.isidentifier():
        return key
    escaped = key.replace("\\", "\\\\").replace("'", "\\'")
    return f"['{escaped}']"


def _json_type_name(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _build_file_search_text(result: JsonIndexResult) -> str | None:
    parts = [
        result.filename,
        result.relative_path,
        f"top-level type: {result.top_level_type}" if result.top_level_type else None,
        f"flattened entries: {result.entry_count}",
        f"max depth: {result.max_depth}",
    ]
    for section in result.sections[:3]:
        if section.path_start and section.path_end:
            parts.append(f"{section.path_start} to {section.path_end}")
    return "\n".join(part for part in parts if part).strip() or None


def _stable_source_id(relative_path: str) -> str:
    return _stable_id(relative_path, "source")


def _stable_id(value: str, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _ensure_windows_selector_event_loop_policy() -> None:
    if sys.platform != "win32":
        return
    policy_factory = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if policy_factory is None:
        return
    if isinstance(asyncio.get_event_loop_policy(), policy_factory):
        return
    asyncio.set_event_loop_policy(policy_factory())


if __name__ == "__main__":
    raise SystemExit(main())
