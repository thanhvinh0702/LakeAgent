from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from lake_agent.config import EmbeddingSettings, LocalSettings, PostgresSettings
from lake_agent.indexing.json import (
    DeterministicJsonParser,
    JsonIndexingProgress,
    JsonIndexingService,
    build_pgvector_store,
)
from lake_agent.persistence.database import PostgresDatabase
from lake_agent.persistence.repositories import JsonIndexRepository


def _load_dotenv() -> None:
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return

    discovered = find_dotenv(usecwd=True)
    if discovered:
        load_dotenv(discovered)

    for parent in Path(__file__).resolve().parents:
        dotenv_path = parent / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path, override=False)
            return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse, persist, and vector-index JSON files."
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional subfolder inside DATALAKE_DIR",
    )
    parser.add_argument(
        "--table-name",
        default="json_index",
        help="PGVectorStore table name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of files to flush to the vector store at once",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Skip embedding/vector indexing and only persist deterministic JSON results",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    progress_callback = None

    try:
        local_settings = LocalSettings.from_env()
        postgres_settings = PostgresSettings.from_env()
        embedding_settings = None if args.no_vector else EmbeddingSettings.from_env()

        database = PostgresDatabase(postgres_settings.dsn)
        with database.connect() as connection:
            database.initialize(connection)
            repository = JsonIndexRepository(connection)
            vector_store = None
            if embedding_settings is not None:
                vector_store = build_pgvector_store(
                    args.table_name,
                    embedding_settings=embedding_settings,
                    postgres_settings=postgres_settings,
                )
            progress_callback = None if args.no_progress else _build_progress_reporter()
            service = JsonIndexingService(
                local_settings.datalake_dir,
                DeterministicJsonParser(),
                repository,
                vector_store=vector_store,
                vector_batch_size=args.batch_size,
                progress_callback=progress_callback,
            )
            result = service.run(args.prefix)
    except Exception as exc:
        if progress_callback is not None:
            _close_progress_reporter(progress_callback)
        print(f"JSON indexing failed: {exc}", file=sys.stderr)
        return 1
    if progress_callback is not None:
        _close_progress_reporter(progress_callback)

    print(f"Datalake Dir: {local_settings.datalake_dir}")
    print(f"Prefix: {result['prefix'] or '<root>'}")
    print(f"Discovered: {result['discovered_count']}")
    print(f"Indexed: {result['indexed_count']}")
    print(f"Unchanged: {result['unchanged_count']}")
    print(f"Errors: {result['error_count']}")
    print(f"Vector Documents: {result['vector_document_count']}")
    if result["errors"]:
        print("Error Details:")
        for error in result["errors"]:
            print(f"- {error.relative_path}: {error.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _build_progress_reporter() -> Callable[[JsonIndexingProgress], None]:
    try:
        from tqdm import tqdm
    except ImportError:
        return _plain_progress_reporter()

    progress_bar: dict[str, object] = {"bar": None}

    def report(progress: JsonIndexingProgress) -> None:
        if progress.event == "start":
            progress_bar["bar"] = tqdm(
                total=progress.total_count,
                unit="file",
                desc="Indexing JSON files",
            )
            return

        bar = progress_bar.get("bar")
        if bar is None:
            return

        postfix = {
            "indexed": progress.indexed_count,
            "unchanged": progress.unchanged_count,
            "errors": progress.error_count,
            "vectors": progress.vector_document_count,
        }
        if progress.event == "error" and progress.relative_path and progress.message:
            tqdm.write(f"ERROR {progress.relative_path}: {progress.message}", file=sys.stderr)
        bar.set_postfix(postfix)
        if progress.relative_path:
            bar.set_description(f"Indexing {progress.relative_path}")

        if progress.event in {"indexed", "unchanged", "error"}:
            bar.update(1)
        if progress.event == "done":
            remaining = progress.total_count - bar.n
            if remaining > 0:
                bar.update(remaining)
            bar.set_description("Indexing complete")

    setattr(report, "_close", lambda: progress_bar.get("bar") and progress_bar["bar"].close())
    return report


def _plain_progress_reporter() -> Callable[[JsonIndexingProgress], None]:
    def report(progress: JsonIndexingProgress) -> None:
        if progress.event == "start":
            print(
                f"Indexing JSON files: 0/{progress.total_count}",
                file=sys.stderr,
            )
            return

        if progress.event in {"indexed", "unchanged", "error"}:
            status = progress.event.upper()
            extra = f" error={progress.message}" if progress.event == "error" and progress.message else ""
            print(
                f"[{progress.processed_count}/{progress.total_count}] "
                f"{status} {progress.relative_path} "
                f"(indexed={progress.indexed_count}, unchanged={progress.unchanged_count}, "
                f"errors={progress.error_count}, vectors={progress.vector_document_count})"
                f"{extra}",
                file=sys.stderr,
            )
            return

        if progress.event == "done":
            print(
                f"Indexing complete: {progress.processed_count}/{progress.total_count} "
                f"(indexed={progress.indexed_count}, unchanged={progress.unchanged_count}, "
                f"errors={progress.error_count}, vectors={progress.vector_document_count})",
                file=sys.stderr,
            )

    setattr(report, "_close", lambda: None)
    return report


def _close_progress_reporter(
    reporter: Callable[[JsonIndexingProgress], None],
) -> None:
    close = getattr(reporter, "_close", None)
    if callable(close):
        close()
