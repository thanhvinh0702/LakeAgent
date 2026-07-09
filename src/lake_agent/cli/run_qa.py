from __future__ import annotations

import argparse
import sys
import logging

from lake_agent.qa.pipeline import QAPipeline

def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Multi-Agent Question Answering Pipeline over the Multi-Modal Data Lake."
    )
    parser.add_argument(
        "--questions",
        default="question.xlsx",
        help="Path to questions XLSX file (default: question.xlsx)",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Path to output CSV file (default: submission.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    _load_dotenv()
    args = build_parser().parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    try:
        pipeline = QAPipeline(xlsx_path=args.questions, output_csv_path=args.output)
        pipeline.run()
    except Exception as exc:
        print(f"QA Pipeline failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
