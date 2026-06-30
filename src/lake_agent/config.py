from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class MinioSettings:
    endpoint: str
    access_key: str
    secret_key: str
    secure: bool
    bucket: str

    @classmethod
    def from_env(cls) -> "MinioSettings":
        endpoint = os.getenv("MINIO_ENDPOINT")
        if not endpoint:
            host = os.getenv("MINIO_HOST", "localhost")
            port = os.getenv("MINIO_PORT", "9000")
            endpoint = f"{host}:{port}"
        return cls(
            endpoint=endpoint,
            access_key=_required_env("MINIO_USER"),
            secret_key=_required_env("MINIO_PASSWORD"),
            secure=_env_bool("MINIO_SECURE", False),
            bucket=os.getenv("MINIO_BUCKET", "datalake"),
        )


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
