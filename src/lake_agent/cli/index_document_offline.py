from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from lake_agent.config import LocalSettings
from lake_agent.indexing.document import (
    DeterministicDocumentParser,
    DocumentEmbeddedImageProcessingOptions,
    DocumentEmbeddedImageProcessor,
    DocumentLLMEnricher,
)
from lake_agent.indexing.document.exchange import (
    serialize_document_error,
    serialize_document_result,
)
from lake_agent.indexing.image import (
    DoclingOCRExtractionOptions,
    DoclingOCRMarkdownExtractor,
    ImageEnrichmentOptions,
    ImageVLMEnricher,
)

_SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc", ".rtf"}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse and enrich document files, then export results to JSONL."
    )
    parser.add_argument("--prefix", default="", help="Optional subfolder inside DATALAKE_DIR")
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to the JSONL file that will store exported document indexing results",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress output")
    parser.add_argument("--no-enrich", action="store_true", help="Skip whole-document LLM enrichment")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR for embedded images")
    parser.add_argument("--ocr-batch-size", type=int, default=10, help="OCR batch size for embedded images")
    parser.add_argument("--no-vlm", action="store_true", help="Skip VLM summaries for embedded images")
    parser.add_argument("--vlm-batch-size", type=int, default=10, help="VLM batch size for embedded images")
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    if args.ocr_batch_size <= 0:
        raise SystemExit("--ocr-batch-size must be greater than 0")
    if args.vlm_batch_size <= 0:
        raise SystemExit("--vlm-batch-size must be greater than 0")

    progress = None if args.no_progress else _build_progress_reporter()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        local_settings = LocalSettings.from_env()
        root_dir = Path(local_settings.datalake_dir).expanduser().resolve()
        parser = DeterministicDocumentParser()
        enricher = None if args.no_enrich else DocumentLLMEnricher.from_env()
        ocr_extractor = None
        if not args.no_ocr:
            ocr_extractor = DoclingOCRMarkdownExtractor.from_default(
                DoclingOCRExtractionOptions(batch_size=args.ocr_batch_size)
            )
        vlm_enricher = None
        if not args.no_vlm and os.getenv("VL_MODEL_NAME"):
            vlm_enricher = ImageVLMEnricher.from_env(ImageEnrichmentOptions())
        image_processor = None
        if ocr_extractor is not None or vlm_enricher is not None:
            image_processor = DocumentEmbeddedImageProcessor(
                ocr_extractor=ocr_extractor,
                vlm_enricher=vlm_enricher,
                options=DocumentEmbeddedImageProcessingOptions(
                    ocr_batch_size=args.ocr_batch_size,
                    vlm_batch_size=args.vlm_batch_size,
                ),
                log_callback=lambda message: progress and progress(f"INFO {message}"),
            )

        files = _scan_files(root_dir, args.prefix)
        if progress is not None:
            progress(f"Indexing document files offline: 0/{len(files)}")

        indexed_count = 0
        error_count = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for index, file_path in enumerate(files, start=1):
                relative_path = file_path.relative_to(root_dir).as_posix()
                stat = file_path.stat()
                indexed_at = datetime.now(timezone.utc)
                source_id = _stable_source_id(relative_path)
                try:
                    result = parser.parse_file(
                        file_path,
                        relative_path=relative_path,
                        source_id=source_id,
                    )
                    if image_processor is not None:
                        image_processor.enrich_batch([result])
                    if enricher is not None:
                        enricher.enrich_batch([result])
                    payload = serialize_document_result(
                        result,
                        size_bytes=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                        indexed_at=indexed_at,
                    )
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    indexed_count += 1
                    if progress is not None:
                        progress(f"[{index}/{len(files)}] INDEXED {relative_path}")
                except Exception as exc:
                    payload = serialize_document_error(
                        source_id=source_id,
                        relative_path=relative_path,
                        filename=file_path.name,
                        file_format=file_path.suffix.lower().removeprefix("."),
                        size_bytes=stat.st_size,
                        last_modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                        indexed_at=indexed_at,
                        error_message=str(exc),
                    )
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    error_count += 1
                    if progress is not None:
                        progress(f"[{index}/{len(files)}] ERROR {relative_path}: {exc}")
                finally:
                    if "result" in locals():
                        _cleanup_artifacts(result)
                        del result
    except Exception as exc:
        print(f"Offline document indexing failed: {exc}", file=sys.stderr)
        return 1

    print(f"Datalake Dir: {local_settings.datalake_dir}")
    print(f"Output Path: {output_path}")
    print(f"Prefix: {args.prefix or '<root>'}")
    print(f"Discovered: {len(files)}")
    print(f"Indexed: {indexed_count}")
    print(f"Errors: {error_count}")
    return 0


def _scan_files(root_dir: Path, prefix: str) -> list[Path]:
    normalized_prefix = prefix.strip().strip("/")
    base_dir = root_dir if not normalized_prefix else (root_dir / normalized_prefix).resolve()
    if not base_dir.exists():
        return []
    if base_dir.is_file():
        return [base_dir] if base_dir.suffix.lower() in _SUPPORTED_SUFFIXES else []
    return [
        path
        for path in sorted(base_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
    ]


def _stable_source_id(relative_path: str) -> str:
    import hashlib

    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"source_{digest}"


def _cleanup_artifacts(result: object) -> None:
    artifact_dir = getattr(result, "artifact_dir", None)
    if artifact_dir:
        shutil.rmtree(artifact_dir, ignore_errors=True)
        setattr(result, "artifact_dir", None)


def _build_progress_reporter() -> Callable[[str], None]:
    def report(message: str) -> None:
        print(message, file=sys.stderr)

    return report


if __name__ == "__main__":
    raise SystemExit(main())
