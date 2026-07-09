from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

from lake_agent.config import LocalSettings, PostgresSettings
from lake_agent.indexing.audio import (
    AudioIndexingProgress,
    AudioIndexingService,
    AudioParseOptions,
    AudioTranscriptParser,
    OpenRouterAudioTranscriber,
    build_pgvector_store,
)
from lake_agent.persistence.database import PostgresDatabase
from lake_agent.persistence.repositories import AudioIndexRepository


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe, persist, and vector-index audio files."
    )
    parser.add_argument("--prefix", default="", help="Optional subfolder inside DATALAKE_DIR")
    parser.add_argument(
        "--table-name",
        default="audio_index",
        help="PGVectorStore table name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=25,
        help="Number of audio documents to flush to the vector store at once",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index unchanged files and call ASR/import transcript again",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress output",
    )
    parser.add_argument(
        "--no-vector",
        action="store_true",
        help="Skip embedding/vector indexing and only persist audio transcripts",
    )
    parser.add_argument(
        "--transcript-dir",
        default="",
        help="Optional directory containing imported transcript JSON files",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional ASR prompt/context, e.g. speaker names or domain terms",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional ASR language hint if the provider supports it, e.g. en or vi",
    )
    parser.add_argument(
        "--max-chunk-seconds",
        type=int,
        default=int(os.getenv("ASR_MAX_CHUNK_SECONDS", "600")),
        help="Maximum audio seconds per ASR request",
    )
    parser.add_argument(
        "--chunk-overlap-seconds",
        type=int,
        default=int(os.getenv("ASR_CHUNK_OVERLAP_SECONDS", "8")),
        help="Overlap seconds between long-audio ASR chunks",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")
    if args.max_chunk_seconds <= 0:
        raise SystemExit("--max-chunk-seconds must be greater than 0")
    if args.chunk_overlap_seconds < 0:
        raise SystemExit("--chunk-overlap-seconds must not be negative")
    if args.chunk_overlap_seconds >= args.max_chunk_seconds:
        raise SystemExit("--chunk-overlap-seconds must be smaller than --max-chunk-seconds")

    progress_callback = None
    try:
        local_settings = LocalSettings.from_env()
        postgres_settings = PostgresSettings.from_env()

        transcript_dir = args.transcript_dir or None
        transcriber = None
        if transcript_dir is None or _asr_settings_present():
            transcriber = OpenRouterAudioTranscriber.from_env(
                prompt=args.prompt,
                language=args.language,
            )

        database = PostgresDatabase(postgres_settings.dsn)
        with database.connect() as connection:
            database.initialize(connection)
            repository = AudioIndexRepository(connection)
            vector_store = None
            if not args.no_vector:
                vector_store = build_pgvector_store(
                    args.table_name,
                    postgres_settings=postgres_settings,
                )
            progress_callback = None if args.no_progress else _build_progress_reporter()
            service = AudioIndexingService(
                local_settings.datalake_dir,
                AudioTranscriptParser(
                    transcriber,
                    options=AudioParseOptions(
                        max_chunk_seconds=args.max_chunk_seconds,
                        chunk_overlap_seconds=args.chunk_overlap_seconds,
                    ),
                    transcript_dir=transcript_dir,
                ),
                repository,
                vector_store=vector_store,
                vector_batch_size=args.batch_size,
                progress_callback=progress_callback,
            )
            result = service.run(args.prefix, force=args.force)
    except Exception as exc:
        if progress_callback is not None:
            _close_progress_reporter(progress_callback)
        print(f"Audio indexing failed: {exc}", file=sys.stderr)
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


def _asr_settings_present() -> bool:
    return bool(
        os.getenv("ASR_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    ) and bool(os.getenv("ASR_MODEL_NAME"))


def _build_progress_reporter() -> Callable[[AudioIndexingProgress], None]:
    try:
        from tqdm import tqdm
    except ImportError:
        return _plain_progress_reporter()

    progress_bar: dict[str, object] = {"bar": None}

    def report(progress: AudioIndexingProgress) -> None:
        if progress.event == "start":
            progress_bar["bar"] = tqdm(
                total=progress.total_count,
                unit="file",
                desc="Indexing audio files",
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
            bar.set_description("Audio indexing complete")

    setattr(report, "_close", lambda: progress_bar.get("bar") and progress_bar["bar"].close())
    return report


def _plain_progress_reporter() -> Callable[[AudioIndexingProgress], None]:
    def report(progress: AudioIndexingProgress) -> None:
        if progress.event == "start":
            print(
                f"Indexing audio files: 0/{progress.total_count}",
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
                f"Audio indexing complete: {progress.processed_count}/{progress.total_count} "
                f"(indexed={progress.indexed_count}, unchanged={progress.unchanged_count}, "
                f"errors={progress.error_count}, vectors={progress.vector_document_count})",
                file=sys.stderr,
            )

    setattr(report, "_close", lambda: None)
    return report


def _close_progress_reporter(
    reporter: Callable[[AudioIndexingProgress], None],
) -> None:
    close = getattr(reporter, "_close", None)
    if callable(close):
        close()


if __name__ == "__main__":
    raise SystemExit(main())
