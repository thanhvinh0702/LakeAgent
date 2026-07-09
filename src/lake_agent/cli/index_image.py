from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

from lake_agent.config import LocalSettings, PostgresSettings
from lake_agent.indexing.image import (
    DeterministicImageParser,
    ImageEnrichmentOptions,
    ImageIndexingProgress,
    ImageIndexingService,
    OCRExtractionOptions,
    OCRMarkdownExtractor,
    ImageVLMEnricher,
    build_pgvector_store,
)
from lake_agent.persistence.database import PostgresDatabase
from lake_agent.persistence.repositories import ImageIndexRepository


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe and persist image metadata for deterministic image indexing."
    )
    parser.add_argument("--prefix", default="", help="Optional subfolder inside DATALAKE_DIR")
    parser.add_argument(
        "--table-name",
        default="image_index",
        help="PGVectorStore table name",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Skip OCR extraction and only store deterministic image metadata",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of image documents to flush to the vector store at once",
    )
    parser.add_argument(
        "--ocr-batch-size",
        type=int,
        default=10,
        help="Number of images to send in each OCR batch",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip VLM image summary enrichment",
    )
    parser.add_argument(
        "--vl-batch-size",
        type=int,
        default=10,
        help="Number of images to send in each VLM enrichment batch",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if args.ocr_batch_size <= 0:
        raise SystemExit("--ocr-batch-size must be greater than 0")
    if args.vl_batch_size <= 0:
        raise SystemExit("--vl-batch-size must be greater than 0")
    progress_callback = None

    try:
        local_settings = LocalSettings.from_env()
        postgres_settings = PostgresSettings.from_env()

        database = PostgresDatabase(postgres_settings.dsn)
        with database.connect() as connection:
            database.initialize(connection)
            repository = ImageIndexRepository(connection)
            ocr_extractor = None
            vlm_enricher = None
            if not args.no_ocr and os.getenv("OCR_MODEL_NAME"):
                ocr_extractor = OCRMarkdownExtractor.from_env(
                    options=OCRExtractionOptions(batch_size=args.ocr_batch_size)
                )
            if not args.no_vlm and os.getenv("VL_MODEL_NAME"):
                vlm_enricher = ImageVLMEnricher.from_env(
                    options=ImageEnrichmentOptions()
                )
            vector_store = build_pgvector_store(
                args.table_name,
                postgres_settings=postgres_settings,
            )
            progress_callback = None if args.no_progress else _build_progress_reporter()
            service = ImageIndexingService(
                local_settings.datalake_dir,
                DeterministicImageParser(),
                repository,
                ocr_extractor=ocr_extractor,
                vlm_enricher=vlm_enricher,
                vector_store=vector_store,
                vl_batch_size=args.vl_batch_size,
                vector_batch_size=args.batch_size,
                progress_callback=progress_callback,
            )
            result = service.run(args.prefix)
    except Exception as exc:
        if progress_callback is not None:
            _close_progress_reporter(progress_callback)
        print(f"Image indexing failed: {exc}", file=sys.stderr)
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


def _build_progress_reporter() -> Callable[[ImageIndexingProgress], None]:
    try:
        from tqdm import tqdm
    except ImportError:
        return _plain_progress_reporter()

    progress_bar: dict[str, object] = {"bar": None}

    def report(progress: ImageIndexingProgress) -> None:
        if progress.event == "start":
            progress_bar["bar"] = tqdm(
                total=progress.total_count,
                unit="file",
                desc="Indexing image files",
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
        if progress.event in {"parsing", "parsed", "flush_pending", "ocr_batch", "vlm_batch", "saving", "vector_flush"} and progress.message:
            tqdm.write(
                f"INFO {progress.relative_path or '<batch>'}: {progress.message}",
                file=sys.stderr,
            )
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


def _plain_progress_reporter() -> Callable[[ImageIndexingProgress], None]:
    def report(progress: ImageIndexingProgress) -> None:
        if progress.event == "start":
            print(
                f"Indexing image files: 0/{progress.total_count}",
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
                f"errors={progress.error_count}, vectors={progress.vector_document_count}){extra}",
                file=sys.stderr,
            )
            return

        if progress.event in {"parsing", "parsed", "flush_pending", "ocr_batch", "vlm_batch", "saving", "vector_flush"}:
            if progress.message:
                print(
                    f"INFO {progress.relative_path or '<batch>'}: {progress.message}",
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


def _close_progress_reporter(reporter: Callable[[ImageIndexingProgress], None]) -> None:
    close = getattr(reporter, "_close", None)
    if callable(close):
        close()
