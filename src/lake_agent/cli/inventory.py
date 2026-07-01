from __future__ import annotations

import argparse
import sys

from lake_agent.config import LocalSettings, PostgresSettings
from lake_agent.inventory.identifier import ObjectIdentifier
from lake_agent.inventory.scanner import ObjectScanner
from lake_agent.inventory.service import InventoryService
from lake_agent.persistence.database import PostgresDatabase
from lake_agent.persistence.repositories import InventoryRepository
from lake_agent.storage.local_store import LocalFileStore


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inventory and identify files stored in a local folder."
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional subfolder inside DATALAKE_DIR",
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
        local_settings = LocalSettings.from_env()
        postgres_settings = PostgresSettings.from_env()

        store = LocalFileStore(local_settings.datalake_dir)
        database = PostgresDatabase(postgres_settings.dsn)
        with database.connect() as connection:
            database.initialize(connection)
            repository = InventoryRepository(connection)
            scanner = ObjectScanner(store)
            service = InventoryService(
                scanner,
                ObjectIdentifier(store),
                repository,
                stat_new_or_changed=not args.no_stat,
            )
            result = service.run(args.prefix)
    except Exception as exc:
        print(f"Inventory failed: {exc}", file=sys.stderr)
        return 1

    print(f"Datalake Dir: {local_settings.datalake_dir}")
    print(f"Prefix: {result['prefix'] or '<root>'}")
    print(f"Discovered: {result['discovered_count']}")
    print(f"Identified: {result['identified_count']}")
    print(f"Unchanged: {result['unchanged_count']}")
    print(f"Errors: {result['error_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
