from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

from lake_agent.config import LocalSettings, PostgresSettings
from lake_agent.indexing.epub import (
    DeterministicEpubParser,
    EpubEnrichmentOptions,
    EpubImageVLMCaptioner,
    EpubIndexingProgress,
    EpubIndexingService,
    EpubParseOptions,
    EpubLLMEnricher,
    EpubVLMOptions,
    build_pgvector_store,
)
from lake_agent.persistence.database import PostgresDatabase
from lake_agent.persistence.repositories import EpubIndexRepository


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse EPUB text/images, persist, and vector-index EPUB files."
    )
    parser.add_argument("--prefix", default="", help="Optional subfolder inside DATALAKE_DIR")
    parser.add_argument("--table-name", default="epub_index", help="PGVectorStore table name")
    parser.add_argument("--batch-size", type=int, default=25, help="Vector flush batch size")
    parser.add_argument("--force", action="store_true", help="Re-index unchanged files")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output")
    parser.add_argument("--no-vector", action="store_true", help="Skip vector indexing")
    parser.add_argument("--no-enrich", action="store_true", help="Skip file-level LLM summary/keywords")
    parser.add_argument("--no-vlm", action="store_true", help="Skip EPUB image VLM captioning")
    parser.add_argument(
        "--vlm-long-edge",
        type=int,
        default=int(os.getenv("EPUB_VL_LONG_EDGE", "768")),
        help="Maximum long edge for EPUB images sent to the VLM",
    )
    parser.add_argument(
        "--max-images-per-file",
        type=int,
        default=int(os.getenv("EPUB_MAX_IMAGES_PER_FILE", "20")),
        help="Maximum embedded EPUB images to caption per file",
    )
    parser.add_argument(
        "--max-chars-per-chunk",
        type=int,
        default=2400,
        help="Maximum characters per text chunk",
    )
    parser.add_argument(
        "--enrich-section-count",
        type=int,
        default=int(os.getenv("EPUB_ENRICH_SECTION_COUNT", "12")),
        help="Number of evenly sampled text chunks used for file summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_utf8_stdio()
    _load_dotenv()
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if args.vlm_long_edge <= 0:
        raise SystemExit("--vlm-long-edge must be greater than 0")
    if args.max_images_per_file < 0:
        raise SystemExit("--max-images-per-file must not be negative")
    if args.max_chars_per_chunk <= 0:
        raise SystemExit("--max-chars-per-chunk must be greater than 0")
    if args.enrich_section_count < 0:
        raise SystemExit("--enrich-section-count must not be negative")

    progress_callback = None
    try:
        local_settings = LocalSettings.from_env()
        postgres_settings = PostgresSettings.from_env()

        vlm_captioner = None
        if not args.no_vlm:
            vlm_captioner = EpubImageVLMCaptioner.from_env(
                options=EpubVLMOptions(max_long_edge=args.vlm_long_edge)
            )
        enricher = None
        if not args.no_enrich:
            enricher = EpubLLMEnricher.from_env(
                options=EpubEnrichmentOptions(
                    section_count_limit=args.enrich_section_count,
                )
            )

        database = PostgresDatabase(postgres_settings.dsn)
        with database.connect() as connection:
            database.initialize(connection)
            repository = EpubIndexRepository(connection)
            vector_store = None
            if not args.no_vector:
                vector_store = build_pgvector_store(
                    args.table_name,
                    postgres_settings=postgres_settings,
                )
            progress_callback = None if args.no_progress else _build_progress_reporter()
            service = EpubIndexingService(
                local_settings.datalake_dir,
                DeterministicEpubParser(
                    EpubParseOptions(
                        max_chars_per_chunk=args.max_chars_per_chunk,
                        extract_images=vlm_captioner is not None,
                        max_images_per_file=args.max_images_per_file,
                    )
                ),
                repository,
                enricher=enricher,
                vlm_captioner=vlm_captioner,
                vector_store=vector_store,
                vector_batch_size=args.batch_size,
                progress_callback=progress_callback,
            )
            result = service.run(args.prefix, force=args.force)
    except Exception as exc:
        if progress_callback is not None:
            _close_progress_reporter(progress_callback)
        print(f"EPUB indexing failed: {exc}", file=sys.stderr)
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


def _build_progress_reporter() -> Callable[[EpubIndexingProgress], None]:
    try:
        from tqdm import tqdm
    except ImportError:
        return _plain_progress_reporter()

    progress_bar: dict[str, object] = {"bar": None}

    def report(progress: EpubIndexingProgress) -> None:
        if progress.event == "start":
            progress_bar["bar"] = tqdm(
                total=progress.total_count,
                unit="file",
                desc="Indexing EPUB files",
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
            bar.set_description("EPUB indexing complete")

    setattr(report, "_close", lambda: progress_bar.get("bar") and progress_bar["bar"].close())
    return report


def _plain_progress_reporter() -> Callable[[EpubIndexingProgress], None]:
    def report(progress: EpubIndexingProgress) -> None:
        if progress.event == "start":
            print(f"Indexing EPUB files: 0/{progress.total_count}", file=sys.stderr)
            return
        if progress.event in {"indexed", "unchanged", "error"}:
            extra = f" error={progress.message}" if progress.event == "error" and progress.message else ""
            print(
                f"[{progress.processed_count}/{progress.total_count}] "
                f"{progress.event.upper()} {progress.relative_path} "
                f"(indexed={progress.indexed_count}, unchanged={progress.unchanged_count}, "
                f"errors={progress.error_count}, vectors={progress.vector_document_count})"
                f"{extra}",
                file=sys.stderr,
            )

    setattr(report, "_close", lambda: None)
    return report


def _close_progress_reporter(reporter: Callable[[EpubIndexingProgress], None]) -> None:
    close = getattr(reporter, "_close", None)
    if callable(close):
        close()


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    raise SystemExit(main())
