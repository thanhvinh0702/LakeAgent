from __future__ import annotations

from pathlib import Path
from typing import Any


class PostgresDatabase:
    def __init__(self, dsn: str, *, connect_timeout: int = 10) -> None:
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")
        self._dsn = dsn
        self._connect_timeout = connect_timeout

    def connect(self) -> Any:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - integration guard
            raise RuntimeError(
                "PostgreSQL support requires 'psycopg[binary]'. "
                "Install the project dependencies first."
            ) from exc
        return psycopg.connect(
            self._dsn,
            autocommit=True,
            row_factory=dict_row,
            connect_timeout=self._connect_timeout,
        )

    def initialize(self, connection: Any) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        schema = schema_path.read_text(encoding="utf-8")
        connection.execute(schema, prepare=False)
