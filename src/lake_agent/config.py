from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True, slots=True)
class LocalSettings:
    datalake_dir: str

    @classmethod
    def from_env(cls) -> "LocalSettings":
        return cls(datalake_dir=_required_env("DATALAKE_DIR"))


@dataclass(frozen=True, slots=True)
class PostgresSettings:
    dsn: str

    @classmethod
    def from_env(cls) -> "PostgresSettings":
        explicit_dsn = os.getenv("POSTGRES_DSN")
        if explicit_dsn:
            return cls(dsn=explicit_dsn)
        host = os.getenv("POSTGRES_DB_HOST", "localhost")
        port = os.getenv("POSTGRES_DB_PORT", "5432")
        database = os.getenv("POSTGRES_DB", "lakeagent_db")
        user = os.getenv("POSTGRES_DB_USER", "lakeagent")
        password = _required_env("POSTGRES_DB_PASSWORD")
        return cls(
            dsn=(
                f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
                f"@{host}:{port}/{quote(database, safe='')}"
            )
        )
