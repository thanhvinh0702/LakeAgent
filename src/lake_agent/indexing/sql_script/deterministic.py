from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lake_agent.domain.indexing_models import (
    SqlScriptFormat,
    SqlScriptIndexResult,
    SqlScriptSection,
)
from lake_agent.indexing.text.chunking import build_basic_search_text, normalize_text


_SUPPORTED_FORMATS = {"sql"}


@dataclass(frozen=True, slots=True)
class SqlScriptParseOptions:
    max_chars_per_chunk: int = 2400
    min_chunk_chars: int = 400


class DeterministicSqlScriptParser:
    def __init__(self, options: SqlScriptParseOptions | None = None) -> None:
        self._options = options or SqlScriptParseOptions()

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> SqlScriptIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported deterministic sql script format: {extension}")

        normalized_relative_path = relative_path or path.name
        normalized_relative_path = PurePosixPath(
            normalized_relative_path.replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        sql_content = path.read_text(encoding="utf-8-sig", errors="replace")
        normalized_sql = normalize_text(sql_content)
        warnings: list[str] = []

        statements = _split_sql_statements(normalized_sql)
        sections = _build_sections(
            statements,
            source_id=normalized_source_id,
            options=self._options,
        )

        if not sections:
            warnings.append("The file did not contain any readable SQL script sections.")

        result = SqlScriptIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format=extension,
            sections=sections,
            parse_warnings=warnings,
        )

        for section in result.sections:
            section.search_text = build_basic_search_text(
                section.heading,
                section.content,
            )
        result.file_search_text = _build_file_search_text(result)
        return result


def _build_sections(
    statements: list[str],
    *,
    source_id: str,
    options: SqlScriptParseOptions,
) -> list[SqlScriptSection]:
    sections: list[SqlScriptSection] = []
    chunk_index = 1
    current_parts: list[str] = []
    current_char_count = 0

    def flush_chunk() -> None:
        nonlocal chunk_index, current_char_count
        if not current_parts:
            return
        content = "\n\n".join(current_parts).strip()
        if not content:
            current_parts.clear()
            current_char_count = 0
            return
        section_id = _stable_id(f"{source_id}:{chunk_index}", prefix="section")
        sections.append(
            SqlScriptSection(
                section_id=section_id,
                chunk_index=chunk_index,
                heading=None,
                content=content,
                char_count=len(content),
            )
        )
        chunk_index += 1
        current_parts.clear()
        current_char_count = 0

    for statement in statements:
        statement_length = len(statement)
        if statement_length > options.max_chars_per_chunk:
            flush_chunk()
            for start in range(0, statement_length, options.max_chars_per_chunk):
                part = statement[start : start + options.max_chars_per_chunk].strip()
                if not part:
                    continue
                current_parts.append(part)
                flush_chunk()
            continue

        separator_length = 2 if current_parts else 0
        if current_parts and current_char_count + separator_length + statement_length > options.max_chars_per_chunk:
            flush_chunk()
        current_parts.append(statement)
        current_char_count += separator_length + statement_length

    flush_chunk()

    return sections


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current_statement: list[str] = []
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False
    index = 0
    length = len(sql)

    while index < length:
        char = sql[index]

        if not in_single_quote and not in_double_quote:
            if in_line_comment:
                if char == "\n":
                    in_line_comment = False
            elif in_block_comment:
                if char == "*" and index + 1 < length and sql[index + 1] == "/":
                    current_statement.append("*/")
                    index += 2
                    in_block_comment = False
                    continue
            else:
                if char == "-" and index + 1 < length and sql[index + 1] == "-":
                    in_line_comment = True
                elif char == "/" and index + 1 < length and sql[index + 1] == "*":
                    in_block_comment = True

        if not in_line_comment and not in_block_comment:
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote

            if char == ";" and not in_single_quote and not in_double_quote:
                current_statement.append(char)
                statement_text = "".join(current_statement).strip()
                if statement_text:
                    statements.append(statement_text)
                current_statement = []
                index += 1
                continue

        current_statement.append(char)
        index += 1

    statement_text = "".join(current_statement).strip()
    if statement_text:
        statements.append(statement_text)

    return statements


def _build_file_search_text(result: SqlScriptIndexResult) -> str | None:
    parts = [result.filename, result.relative_path]
    for section in result.sections[:3]:
        if section.content:
            parts.append(section.content[:200])
    return "\n".join(part for part in parts if part).strip() or None


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
