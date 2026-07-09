from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lake_agent.domain.indexing_models import DatabaseFormat, DatabaseIndexResult, DbColumnProfile, DbTableProfile
from lake_agent.domain.indexing_models.tabular import ScalarType


_SUPPORTED_FORMATS = {"db", "sqlite", "sqlite3"}


@dataclass(frozen=True, slots=True)
class DatabaseParseOptions:
    max_preview_rows: int = 5
    max_sample_values: int = 5


class DeterministicDatabaseParser:
    def __init__(self, options: DatabaseParseOptions | None = None) -> None:
        self._options = options or DatabaseParseOptions()

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> DatabaseIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported deterministic database format: {extension}")

        canonical_format: DatabaseFormat = "sqlite" if extension in {"sqlite", "sqlite3"} else "db"

        normalized_relative_path = relative_path or path.name
        normalized_relative_path = PurePosixPath(
            normalized_relative_path.replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        tables: list[DbTableProfile] = []
        warnings: list[str] = []

        try:
            uri_path = f"file:{path.as_posix()}?mode=ro"
            connection = sqlite3.connect(uri_path, uri=True)
        except Exception as exc:
            try:
                connection = sqlite3.connect(str(path))
            except Exception as inner_exc:
                raise RuntimeError(f"Failed to open SQLite database: {inner_exc}") from exc

        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            table_names = [row[0] for row in cursor.fetchall()]

            for name in table_names:
                table_id = _stable_id(f"{normalized_source_id}:{name}", prefix="table")
                table_warnings = []
                
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM "{name}";')
                    row_count = cursor.fetchone()[0]
                except Exception as exc:
                    row_count = None
                    table_warnings.append(f"Failed to query row count: {exc}")

                columns: list[DbColumnProfile] = []
                try:
                    cursor.execute(f'PRAGMA table_info("{name}");')
                    pragma_info = cursor.fetchall()
                    for cid, col_name, col_type, notnull, _, _ in pragma_info:
                        nullable = not notnull
                        
                        col_type_lower = col_type.lower()
                        inf_type: ScalarType = "unknown"
                        if "int" in col_type_lower:
                            inf_type = "integer"
                        elif "char" in col_type_lower or "text" in col_type_lower or "clob" in col_type_lower:
                            inf_type = "string"
                        elif "real" in col_type_lower or "floa" in col_type_lower or "doub" in col_type_lower:
                            inf_type = "float"
                        elif "bool" in col_type_lower:
                            inf_type = "boolean"
                        elif "date" in col_type_lower:
                            inf_type = "date"
                        elif "time" in col_type_lower:
                            inf_type = "datetime"
                            
                        null_count = None
                        distinct_ratio = None
                        sample_values = []
                        
                        if row_count is not None and row_count > 0:
                            try:
                                cursor.execute(f'SELECT COUNT(*) FROM "{name}" WHERE "{col_name}" IS NULL;')
                                null_count = cursor.fetchone()[0]
                            except Exception as exc:
                                table_warnings.append(f"Failed to query null count for column {col_name}: {exc}")
                                
                            try:
                                cursor.execute(f'SELECT COUNT(DISTINCT "{col_name}") FROM "{name}";')
                                distinct_count = cursor.fetchone()[0]
                                distinct_ratio = distinct_count / row_count if row_count > 0 else 0.0
                            except Exception as exc:
                                table_warnings.append(f"Failed to query distinct count for column {col_name}: {exc}")

                            try:
                                cursor.execute(
                                    f'SELECT "{col_name}" FROM "{name}" WHERE "{col_name}" IS NOT NULL LIMIT {self._options.max_sample_values};'
                                )
                                sample_values = [str(r[0]) for r in cursor.fetchall() if r[0] is not None]
                            except Exception as exc:
                                table_warnings.append(f"Failed to query sample values for column {col_name}: {exc}")

                        columns.append(
                            DbColumnProfile(
                                name=col_name,
                                ordinal=cid,
                                inferred_type=inf_type,
                                nullable=nullable,
                                distinct_ratio=distinct_ratio,
                                sample_values=sample_values,
                                null_count=null_count,
                            )
                        )
                except Exception as exc:
                    table_warnings.append(f"Failed to query columns info: {exc}")

                preview_rows = []
                if row_count is not None and row_count > 0:
                    try:
                        cursor.execute(f'SELECT * FROM "{name}" LIMIT {self._options.max_preview_rows};')
                        preview_rows = [[str(cell) if cell is not None else "" for cell in row] for row in cursor.fetchall()]
                    except Exception as exc:
                        table_warnings.append(f"Failed to query preview rows: {exc}")

                col_headers = [col.name for col in columns]
                preview_lines = []
                if col_headers:
                    preview_lines.append("| " + " | ".join(col_headers) + " |")
                    preview_lines.append("| " + " | ".join(["---"] * len(col_headers)) + " |")
                for row in preview_rows:
                    preview_lines.append("| " + " | ".join(row) + " |")
                    
                table_search_text = (
                    f"Table: {name}\n"
                    f"Columns: {', '.join(col_headers)}\n"
                    f"Row count: {row_count or 0}\n"
                    f"Preview:\n" + "\n".join(preview_lines)
                )

                tables.append(
                    DbTableProfile(
                        table_id=table_id,
                        table_name=name,
                        row_count=row_count,
                        column_count=len(columns),
                        columns=columns,
                        preview_rows=preview_rows,
                        table_search_text=table_search_text,
                        warnings=table_warnings,
                    )
                )

        except Exception as exc:
            warnings.append(f"Database scan failed: {exc}")
        finally:
            connection.close()

        if not tables:
            warnings.append("The database did not contain any readable user tables.")

        result = DatabaseIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format=canonical_format,
            tables=tables,
            parse_warnings=warnings,
        )
        result.file_search_text = _build_file_search_text(result)
        return result


def _build_file_search_text(result: DatabaseIndexResult) -> str | None:
    parts = [result.filename, result.relative_path]
    for table in result.tables[:3]:
        parts.append(f"Table: {table.table_name}")
        cols = [col.name for col in table.columns]
        if cols:
            parts.append(f"Columns: {', '.join(cols)}")
    return "\n".join(part for part in parts if part).strip() or None


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
