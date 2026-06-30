from __future__ import annotations

import argparse
import sys

from lake_agent.config import MinioSettings, PostgresSettings
from lake_agent.inventory.hasher import ObjectHasher
from lake_agent.inventory.identifier import ObjectIdentifier
from lake_agent.inventory.scanner import ObjectScanner
from lake_agent.inventory.service import InventoryService
from lake_agent.persistence.database import PostgresDatabase
from lake_agent.persistence.repositories import InventoryRepository
from lake_agent.storage.minio_store import MinioObjectStore


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory and identify objects stored in MinIO."
    )
    parser.add_argument("--bucket", help="MinIO bucket; defaults to MINIO_BUCKET")
    parser.add_argument("--prefix", default="", help="Optional object-key prefix")
    parser.add_argument(
        "--hash-content",
        action="store_true",
        help="Stream every new/changed object and calculate SHA-256",
    )
    parser.add_argument(
        "--no-stat",
        action="store_true",
        help="Skip stat_object for new/changed objects",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)

    try:
        minio_settings = MinioSettings.from_env()
        postgres_settings = PostgresSettings.from_env()
        bucket = args.bucket or minio_settings.bucket

        store = MinioObjectStore(
            minio_settings.endpoint,
            minio_settings.access_key,
            minio_settings.secret_key,
            secure=minio_settings.secure,
        )
        database = PostgresDatabase(postgres_settings.dsn)
        with database.connect() as connection:
            database.initialize(connection)
            repository = InventoryRepository(connection)
            scanner = ObjectScanner(store)
            service = InventoryService(
                scanner,
                ObjectIdentifier(store),
                repository,
                hasher=ObjectHasher(store),
                hash_content=args.hash_content,
                stat_new_or_changed=not args.no_stat,
            )
            result = service.run(bucket, args.prefix)
    except Exception as exc:
        print(f"Inventory failed: {exc}", file=sys.stderr)
        return 1

    print(f"Inventory run: {result.run_id}")
    print(f"Bucket: {result.bucket}")
    print(f"Prefix: {result.prefix or '<root>'}")
    print(f"Discovered: {result.discovered_count}")
    print(f"Identified: {result.identified_count}")
    print(f"Unchanged: {result.unchanged_count}")
    print(f"Errors: {result.error_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
