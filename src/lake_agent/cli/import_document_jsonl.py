from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lake_agent.config import PostgresSettings
from lake_agent.indexing.document import build_pgvector_store
from lake_agent.indexing.document.exchange import deserialize_document_result
from lake_agent.indexing.document.vector_store import add_document_results
from lake_agent.persistence.database import PostgresDatabase
from lake_agent.persistence.repositories import DocumentIndexRepository


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import offline-exported document indexing results from JSONL into Postgres."
    )
    parser.add_argument("--input-path", required=True, help="Path to exported document JSONL")
    parser.add_argument("--table-name", default="document_index", help="PGVectorStore table name")
    parser.add_argument("--batch-size", type=int, default=25, help="Vector insert batch size")
    parser.add_argument("--skip-vectors", action="store_true", help="Import DB rows only, skip vector store")
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than 0")

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    try:
        postgres_settings = PostgresSettings.from_env()
        database = PostgresDatabase(postgres_settings.dsn)
        with database.connect() as connection:
            database.initialize(connection)
            repository = DocumentIndexRepository(connection)
            vector_store = None
            if not args.skip_vectors:
                vector_store = build_pgvector_store(
                    args.table_name,
                    postgres_settings=postgres_settings,
                )

            imported_count = 0
            skipped_error_count = 0
            vector_batch = []
            with input_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    payload = json.loads(raw)
                    record_type = payload.get("record_type")
                    if record_type == "document_error":
                        skipped_error_count += 1
                        continue
                    if record_type != "document_result":
                        raise ValueError(
                            f"Unsupported record_type at line {line_number}: {record_type!r}"
                        )
                    result, size_bytes, last_modified, indexed_at = deserialize_document_result(payload)
                    repository.save(
                        result,
                        size_bytes=size_bytes,
                        last_modified=last_modified,
                        indexed_at=indexed_at,
                    )
                    vector_batch.append(result)
                    imported_count += 1
                    if vector_store is not None and len(vector_batch) >= args.batch_size:
                        add_document_results(vector_store, vector_batch)
                        vector_batch.clear()

            if vector_store is not None and vector_batch:
                add_document_results(vector_store, vector_batch)

    except Exception as exc:
        print(f"Document JSONL import failed: {exc}", file=sys.stderr)
        return 1

    print(f"Input Path: {input_path}")
    print(f"Imported: {imported_count}")
    print(f"Skipped Errors: {skipped_error_count}")
    print(f"Vectors: {'disabled' if args.skip_vectors else 'enabled'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
