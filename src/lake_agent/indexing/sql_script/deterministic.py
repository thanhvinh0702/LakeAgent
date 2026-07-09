from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lake_agent.domain.indexing_models import SqlScriptFormat, SqlScriptIndexResult, SqlScriptSection
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
        
        sections: list[SqlScriptSection] = []
        chunk_index = 1
        current_chunk_parts = []
        current_char_count = 0
        
        def flush_chunk():
            nonlocal chunk_index, current_char_count
            if not current_chunk_parts:
                return
            content = "\n\n".join(current_chunk_parts).strip()
            sec_id = _stable_id(f"{normalized_source_id}:{chunk_index}", prefix="section")
            sections.append(
                SqlScriptSection(
                    section_id=sec_id,
                    chunk_index=chunk_index,
                    heading=None,
                    content=content,
                    char_count=len(content),
                )
            )
            chunk_index += 1
            current_chunk_parts.clear()
            current_char_count = 0

        for stmt in statements:
            stmt_len = len(stmt)
            if stmt_len > self._options.max_chars_per_chunk:
                flush_chunk()
                for start in range(0, stmt_len, self._options.max_chars_per_chunk):
                    part = stmt[start:start + self._options.max_chars_per_chunk].strip()
                    if part:
                        current_chunk_parts.append(part)
                        flush_chunk()
            elif current_char_count + stmt_len + (2 if current_chunk_parts else 0) > self._options.max_chars_per_chunk:
                flush_chunk()
                current_chunk_parts.append(stmt)
                current_char_count = stmt_len
            else:
                current_chunk_parts.append(stmt)
                current_char_count += stmt_len + (2 if current_chunk_parts else 0)
                
        flush_chunk()

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


def _split_sql_statements(sql: str) -> list[str]:
    statements = []
    current_statement = []
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    in_block_comment = False
    i = 0
    n = len(sql)
    while i < n:
        char = sql[i]
        
        if not in_single_quote and not in_double_quote:
            if in_line_comment:
                if char == '\n':
                    in_line_comment = False
            elif in_block_comment:
                if char == '*' and i + 1 < n and sql[i+1] == '/':
                    current_statement.append('*/')
                    i += 2
                    in_block_comment = False
                    continue
            else:
                if char == '-' and i + 1 < n and sql[i+1] == '-':
                    in_line_comment = True
                elif char == '/' and i + 1 < n and sql[i+1] == '*':
                    in_block_comment = True
        
        if not in_line_comment and not in_block_comment:
            if char == "'" and not in_double_quote:
                in_single_quote = not in_single_quote
            elif char == '"' and not in_single_quote:
                in_double_quote = not in_double_quote
            
            if char == ';' and not in_single_quote and not in_double_quote:
                current_statement.append(char)
                stmt_text = "".join(current_statement).strip()
                if stmt_text:
                    statements.append(stmt_text)
                current_statement = []
                i += 1
                continue
                
        current_statement.append(char)
        i += 1
        
    stmt_text = "".join(current_statement).strip()
    if stmt_text:
        statements.append(stmt_text)
        
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
