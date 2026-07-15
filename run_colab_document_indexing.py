from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1nk8Z_DBbqQNJljHv6WzNV-v1WSzIfmdG?usp=drive_link"
)
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the shared Data Lake folder in Colab and export document "
            "indexing results to JSONL without PostgreSQL."
        )
    )
    parser.add_argument("--drive-url", default=DEFAULT_DRIVE_FOLDER_URL)
    parser.add_argument("--data-dir", default="/content/lakeagent-data")
    parser.add_argument("--output-path", default="/content/documents.jsonl")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-enrich", action="store_true")
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-vlm", action="store_true")
    parser.add_argument("--ocr-batch-size", type=int, default=1)
    parser.add_argument("--vlm-batch-size", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()

    _configure_openrouter()
    _download_drive_folder(
        args.drive_url,
        data_dir,
        force=args.force_download,
    )
    os.environ["DATALAKE_DIR"] = str(data_dir)

    from lake_agent.cli.index_document_offline import main as offline_main

    offline_args = [
        "--output-path",
        str(output_path),
        "--ocr-batch-size",
        str(args.ocr_batch_size),
        "--vlm-batch-size",
        str(args.vlm_batch_size),
    ]
    if args.prefix:
        offline_args.extend(["--prefix", args.prefix])
    if args.no_enrich:
        offline_args.append("--no-enrich")
    if args.no_ocr:
        offline_args.append("--no-ocr")
    if args.no_vlm:
        offline_args.append("--no-vlm")

    print(f"Data Lake: {data_dir}")
    print(f"Output JSONL: {output_path}")
    return offline_main(offline_args)


def _configure_openrouter() -> None:
    key = os.getenv("OPENROUTER_API_KEY") or _read_colab_secret(
        "OPENROUTER_API_KEY"
    )
    if not key:
        raise SystemExit(
            "OpenRouter key not found. Add OPENROUTER_API_KEY to Colab Secrets "
            "or set os.environ['OPENROUTER_API_KEY'] before running this script."
        )

    os.environ["OPENROUTER_API_KEY"] = key
    os.environ["OPENAI_API_KEY"] = key
    os.environ.setdefault("OPENAI_BASE_URL", OPENROUTER_BASE_URL)
    os.environ.setdefault("OPENAI_MODEL_NAME", "google/gemini-2.5-flash")
    os.environ.setdefault("VL_BASE_URL", OPENROUTER_BASE_URL)
    os.environ.setdefault("VL_MODEL_NAME", "qwen/qwen3-vl-32b-instruct")


def _read_colab_secret(name: str) -> str | None:
    try:
        from google.colab import userdata
    except ImportError:
        return None
    try:
        value = userdata.get(name)
    except Exception:
        return None
    return str(value).strip() if value else None


def _download_drive_folder(url: str, destination: Path, *, force: bool) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit(
            "Missing gdown. In Colab run: %pip install -q gdown"
        ) from exc

    if force and destination.exists():
        shutil.rmtree(destination)
    if destination.exists() and any(destination.iterdir()):
        print(f"Using existing downloaded data: {destination}")
        return

    destination.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download_folder(
        url=url,
        output=str(destination),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    if not downloaded:
        raise SystemExit(
            "Google Drive folder download returned no files. Ensure the folder is "
            "shared as 'Anyone with the link' and rerun with --force-download."
        )


if __name__ == "__main__":
    raise SystemExit(main())
