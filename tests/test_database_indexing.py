from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.indexing.database import (
    DeterministicDatabaseParser,
    DatabaseIndexingService,
    build_database_documents,
)
from lake_agent.persistence.repositories import DatabaseIndexRepository


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
        if "FROM database_files WHERE relative_path = %s" in compact_query:
            return RecordingResult(self.rows.get(params[0]))
        return RecordingResult()


class FakeDatabaseRepository:
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


class FakeDatabaseEnricher:
    def __init__(self) -> None:
        self.batch_calls = []

    def enrich(self, result):
        result.file_summary = f"Summary for {result.filename}"
        result.file_keywords = ["database", "sqlite"]
        for table in result.tables:
            table.summary = f"Summary for {table.table_name}"
            table.keywords = ["data"]
            table.table_search_text = f"{table.table_name} {table.summary}"
        result.file_search_text = f"{result.filename} {result.file_summary}"
        return result

    def enrich_batch(self, results):
        self.batch_calls.append([result.source_id for result in results])
        return [self.enrich(result) for result in results]


class DeterministicDatabaseParserTest(unittest.TestCase):
    def test_database_parser_extracts_schema_and_previews(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE users (id INT, name TEXT);")
            cursor.execute("INSERT INTO users VALUES (1, 'Alice');")
            cursor.execute("INSERT INTO users VALUES (2, 'Bob');")
            cursor.execute("INSERT INTO users VALUES (3, NULL);")
            conn.commit()
            conn.close()

            result = DeterministicDatabaseParser().parse_file(
                db_path,
                relative_path="data/test.db",
            )

        self.assertEqual("db", result.file_format)
        self.assertEqual(1, len(result.tables))
        table = result.tables[0]
        self.assertEqual("users", table.table_name)
        self.assertEqual(3, table.row_count)
        self.assertEqual(2, table.column_count)
        
        # Test columns
        self.assertEqual("id", table.columns[0].name)
        self.assertEqual("integer", table.columns[0].inferred_type)
        self.assertEqual(0, table.columns[0].null_count)
        
        self.assertEqual("name", table.columns[1].name)
        self.assertEqual("string", table.columns[1].inferred_type)
        self.assertEqual(1, table.columns[1].null_count)
        self.assertEqual(2/3, table.columns[1].distinct_ratio)
        self.assertEqual(["Alice", "Bob"], table.columns[1].sample_values)
        
        # Test preview rows
        self.assertEqual([["1", "Alice"], ["2", "Bob"], ["3", ""]], table.preview_rows)
        self.assertIsNone(result.file_summary)


class DatabaseIndexRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = DatabaseIndexRepository(connection)

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE users (id INT);")
            conn.commit()
            conn.close()
            
            result = DeterministicDatabaseParser().parse_file(
                db_path,
                relative_path="test.db",
            )

        repository.find_file("test.db")
        repository.save(
            result,
            size_bytes=8192,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.save_error(
            source_id=result.source_id,
            relative_path=result.relative_path,
            filename=result.filename,
            file_format=result.file_format,
            size_bytes=8192,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            error_message="failed",
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.mark_missing("", datetime(2026, 7, 1, tzinfo=UTC))
        repository.mark_missing("data", datetime(2026, 7, 1, tzinfo=UTC))


class DatabaseIndexingServiceTest(unittest.TestCase):
    def test_service_saves_results_and_flushes_vector_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "db").mkdir()
            
            # Create two sqlite files
            for name in ["a.sqlite", "b.db"]:
                db_path = root / "db" / name
                conn = sqlite3.connect(str(db_path))
                conn.execute("CREATE TABLE val (num INT);")
                conn.execute("INSERT INTO val VALUES (42);")
                conn.commit()
                conn.close()

            repository = FakeDatabaseRepository()
            vector_store = FakeVectorStore()
            enricher = FakeDatabaseEnricher()
            service = DatabaseIndexingService(
                root,
                DeterministicDatabaseParser(),
                repository,
                enricher=enricher,
                vector_store=vector_store,
                enrich_batch_size=2,
                vector_batch_size=2,
            )

            first = service.run("db")
            second = service.run("db")

        self.assertEqual(2, first["discovered_count"])
        self.assertEqual(2, first["indexed_count"])
        self.assertEqual(0, first["error_count"])
        self.assertGreaterEqual(first["vector_document_count"], 4)
        self.assertEqual(1, len(vector_store.batches))
        self.assertEqual(1, len(enricher.batch_calls))
        self.assertEqual(2, len(enricher.batch_calls[0]))
        self.assertEqual(2, second["unchanged_count"])
        self.assertEqual("Summary for a.sqlite", repository.saved_results[0].file_summary)

    def test_build_database_documents_emits_file_and_table_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE x (y INT);")
            conn.commit()
            conn.close()
            
            result = DeterministicDatabaseParser().parse_file(
                db_path,
                relative_path="db/test.db",
            )

        documents = build_database_documents(result)

        self.assertEqual("file", documents[0].metadata["record_type"])
        self.assertEqual("table", documents[1].metadata["record_type"])


if __name__ == "__main__":
    unittest.main()
