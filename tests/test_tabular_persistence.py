from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.indexing.tabular import DeterministicTabularParser, TabularIndexingService
from lake_agent.persistence.repositories import TabularIndexRepository


class RecordingResult:
    def __init__(self, row=None) -> None:
        self._row = row

    def fetchone(self):
        return self._row


class PlaceholderCheckingConnection:
    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def execute(self, query: str, params=None):
        if params is not None:
            expected = query.count("%s")
            if expected != len(params):
                raise AssertionError(
                    f"SQL expects {expected} parameters, received {len(params)}"
                )

        compact_query = " ".join(query.split())
        if "FROM tabular_files WHERE relative_path = %s" in compact_query:
            return RecordingResult(self.rows.get(params[0]))
        return RecordingResult()


class FakeTabularRepository:
    def __init__(self) -> None:
        self.files: dict[str, dict] = {}
        self.saved_results = []
        self.saved_errors = []
        self.marked_missing = []

    def find_file(self, relative_path: str):
        return self.files.get(relative_path)

    def save(self, result, *, size_bytes: int, last_modified: datetime, indexed_at: datetime):
        self.files[result.relative_path] = {
            "source_id": result.source_id,
            "size_bytes": size_bytes,
            "last_modified": last_modified,
            "status": "indexed",
        }
        self.saved_results.append(result)

    def save_error(self, **kwargs):
        self.files[kwargs["relative_path"]] = {
            "source_id": kwargs["source_id"],
            "size_bytes": kwargs["size_bytes"],
            "last_modified": kwargs["last_modified"],
            "status": "error",
        }
        self.saved_errors.append(kwargs)

    def mark_missing(self, prefix: str, indexed_at: datetime):
        self.marked_missing.append((prefix, indexed_at))


class FakeVectorStore:
    def __init__(self) -> None:
        self.batches = []

    def add_documents(self, *, documents, ids):
        self.batches.append((documents, ids))


class FakeEnricher:
    def enrich(self, result):
        result = replace(
            result,
            file_summary=f"Summary for {result.filename}",
            file_keywords=["tabular"],
            file_search_text=(
                f"{result.filename}\n"
                f"{result.relative_path}\n"
                f"Summary for {result.filename}\n"
                "tabular"
            ),
        )
        for table in result.tables:
            table.summary = f"Summary for {table.table_name}"
            table.keywords = ["table"]
            table.table_search_text = f"{table.table_name}\n{table.summary}"
        return result


class TabularIndexRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = TabularIndexRepository(connection)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sales.csv"
            file_path.write_text("customer_id,amount\n101,12.5\n", encoding="utf-8")
            result = DeterministicTabularParser().parse_file(
                file_path,
                relative_path="tables/sales.csv",
            )

        repository.find_file("tables/sales.csv")
        repository.save(
            result,
            size_bytes=18,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.save_error(
            source_id=result.source_id,
            relative_path=result.relative_path,
            filename=result.filename,
            file_format=result.file_format,
            size_bytes=18,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            error_message="failed",
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.mark_missing("", datetime(2026, 7, 1, tzinfo=UTC))
        repository.mark_missing("tables", datetime(2026, 7, 1, tzinfo=UTC))


class TabularIndexingServiceTest(unittest.TestCase):
    def test_service_saves_results_and_flushes_vector_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tables").mkdir()
            (root / "tables" / "sales.csv").write_text(
                "customer_id,amount\n101,12.5\n",
                encoding="utf-8",
            )
            (root / "tables" / "users.tsv").write_text(
                "user_id\tname\n1\talice\n",
                encoding="utf-8",
            )
            (root / "tables" / "notes.txt").write_text(
                "ignore me",
                encoding="utf-8",
            )

            repository = FakeTabularRepository()
            vector_store = FakeVectorStore()
            service = TabularIndexingService(
                root,
                DeterministicTabularParser(),
                repository,
                enricher=FakeEnricher(),
                vector_store=vector_store,
                vector_batch_size=2,
            )

            first = service.run("tables")
            second = service.run("tables")

        self.assertEqual(2, first["discovered_count"])
        self.assertEqual(2, first["indexed_count"])
        self.assertEqual(0, first["unchanged_count"])
        self.assertEqual(0, first["error_count"])
        self.assertEqual(4, first["vector_document_count"])
        self.assertEqual(1, len(vector_store.batches))
        self.assertEqual(2, len(repository.saved_results))

        self.assertEqual(2, second["discovered_count"])
        self.assertEqual(0, second["indexed_count"])
        self.assertEqual(2, second["unchanged_count"])
        self.assertEqual(0, second["error_count"])

    def test_service_emits_progress_events(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "tables").mkdir()
            (root / "tables" / "sales.csv").write_text(
                "customer_id,amount\n101,12.5\n",
                encoding="utf-8",
            )

            repository = FakeTabularRepository()
            events = []
            service = TabularIndexingService(
                root,
                DeterministicTabularParser(),
                repository,
                progress_callback=events.append,
            )

            service.run("tables")

        self.assertEqual("start", events[0].event)
        self.assertEqual("indexed", events[1].event)
        self.assertEqual("tables/sales.csv", events[1].relative_path)
        self.assertEqual(1, events[1].processed_count)
        self.assertEqual(1, events[1].total_count)
        self.assertEqual("done", events[-1].event)


if __name__ == "__main__":
    unittest.main()
