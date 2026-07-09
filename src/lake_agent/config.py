from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


@dataclass(frozen=True, slots=True)
class LocalSettings:
    datalake_dir: str

    @classmethod
    def from_env(cls) -> "LocalSettings":
        return cls(datalake_dir=_required_env("DATALAKE_DIR"))


@dataclass(frozen=True, slots=True)
class PostgresSettings:
    dsn: str
    dsn_vector: str

    @classmethod
    def from_env(cls) -> "PostgresSettings":
        explicit_dsn = os.getenv("POSTGRES_DSN")
        if explicit_dsn:
            return cls(
                dsn=explicit_dsn,
                dsn_vector=explicit_dsn.replace("postgresql://", "postgresql+psycopg://", 1),
            )
        host = os.getenv("POSTGRES_DB_HOST", "localhost")
        port = os.getenv("POSTGRES_DB_PORT", "5432")
        database = os.getenv("POSTGRES_DB", "lakeagent_db")
        user = os.getenv("POSTGRES_DB_USER", "lakeagent")
        password = _required_env("POSTGRES_DB_PASSWORD")
        return cls(
            dsn=(
                f"postgresql://{quote(user, safe='')}:{quote(password, safe='')}"
                f"@{host}:{port}/{quote(database, safe='')}"
            ),
            dsn_vector=(
                f"postgresql+psycopg://{quote(user, safe='')}:{quote(password, safe='')}"
                f"@{host}:{port}/{quote(database, safe='')}"
            )
        )


@dataclass(frozen=True, slots=True)
class LLMSettings:
    api_key: str
    model_name: str
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "LLMSettings":
        api_key = _first_env("OPENAI_API_KEY", "API_KEY")
        if not api_key:
            raise ValueError(
                "Missing required environment variable: OPENAI_API_KEY or API_KEY"
            )

        model_name = _first_env("OPENAI_MODEL_NAME", "MODEL_NAME")
        if not model_name:
            raise ValueError(
                "Missing required environment variable: OPENAI_MODEL_NAME or MODEL_NAME"
            )

        return cls(
            api_key=api_key,
            model_name=model_name,
            base_url=_first_env("OPENAI_BASE_URL", "BASE_URL"),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    api_key: str
    model_name: str
    base_url: str | None = None
    dimensions: int | None = None

    @classmethod
    def from_env(cls) -> "EmbeddingSettings":
        api_key = _first_env("OPENAI_API_KEY", "API_KEY")
        if not api_key:
            raise ValueError(
                "Missing required environment variable: OPENAI_API_KEY or API_KEY"
            )

        model_name = _first_env(
            "OPENAI_EMBEDDING_MODEL_NAME",
            "EMBEDDING_MODEL_NAME",
        )
        if not model_name:
            raise ValueError(
                "Missing required environment variable: "
                "OPENAI_EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_NAME"
            )

        return cls(
            api_key=api_key,
            model_name=model_name,
            base_url=_first_env("OPENAI_BASE_URL", "BASE_URL"),
            dimensions=_optional_int_env(
                "OPENAI_EMBEDDING_DIMENSIONS",
                "EMBEDDING_DIMENSIONS",
            ),
        )


@dataclass(frozen=True, slots=True)
class OCRSettings:
    model_url: str

    @classmethod
    def from_env(cls) -> "OCRSettings":
        return cls(model_url=_required_env("OCR_MODEL_URL"))


@dataclass(frozen=True, slots=True)
class VLSettings:
    api_key: str
    model_name: str
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "VLSettings":
        api_key = _first_env("OPENAI_API_KEY", "API_KEY")
        if not api_key:
            raise ValueError(
                "Missing required environment variable: OPENAI_API_KEY or API_KEY"
            )

        model_name = _first_env("VL_MODEL_NAME")
        if not model_name:
            raise ValueError("Missing required environment variable: VL_MODEL_NAME")

        return cls(
            api_key=api_key,
            model_name=model_name,
            base_url=_first_env("VL_BASE_URL", "OPENAI_BASE_URL", "BASE_URL"),
        )


@dataclass(frozen=True, slots=True)
class ASRSettings:
    api_key: str
    model_name: str
    base_url: str
    fallback_model_name: str | None = None

    @classmethod
    def from_env(cls) -> "ASRSettings":
        api_key = _first_env("ASR_API_KEY", "OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing required environment variable: ASR_API_KEY or OPENROUTER_API_KEY"
            )

        model_name = _first_env("ASR_MODEL_NAME")
        if not model_name:
            raise ValueError("Missing required environment variable: ASR_MODEL_NAME")

        return cls(
            api_key=api_key,
            model_name=model_name,
            base_url=_first_env("ASR_BASE_URL", "OPENROUTER_BASE_URL")
            or "https://openrouter.ai/api/v1",
            fallback_model_name=_first_env("ASR_FALLBACK_MODEL_NAME"),
        )


def _optional_int_env(*names: str) -> int | None:
    value = _first_env(*names)
    if value is None:
        return None
    return int(value)
