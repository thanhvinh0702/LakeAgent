from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.indexing.sql_script import (
    DeterministicSqlScriptParser,
    SqlScriptParseOptions,
    SqlScriptIndexingService,
    build_sql_script_documents,
)
from lake_agent.persistence.repositories import SqlScriptIndexRepository


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
        if "FROM sql_script_files WHERE relative_path = %s" in compact_query:
            return RecordingResult(self.rows.get(params[0]))
        return RecordingResult()


class FakeSqlScriptRepository:
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


class FakeSqlScriptEnricher:
    def __init__(self) -> None:
        self.batch_calls = []

    def enrich(self, result):
        result.file_summary = f"Summary for {result.filename}"
        result.file_keywords = ["sql", "script"]
        for section in result.sections:
            parts = []
            if section.heading:
                parts.append(section.heading)
            parts.append(result.file_summary)
            parts.append(section.content)
            section.search_text = "\n".join(part for part in parts if part)
        result.file_search_text = "\n".join(
            part
            for part in [
                result.filename,
                result.relative_path,
                result.file_summary,
                ", ".join(result.file_keywords),
            ]
            if part
        )
        return result

    def enrich_batch(self, results):
        self.batch_calls.append([result.source_id for result in results])
        return [self.enrich(result) for result in results]


class DeterministicSqlScriptParserTest(unittest.TestCase):
    def test_sql_parser_splits_statements_correctly(self) -> None:
        sql_content = (
            "CREATE TABLE users (\n"
            "  id INTEGER PRIMARY KEY,\n"
            "  name TEXT\n"
            ");\n"
            "\n"
            "-- Comment in SQL\n"
            "INSERT INTO users (name) VALUES ('Alice');\n"
            "INSERT INTO users (name) VALUES ('Bob');\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "init.sql"
            file_path.write_text(sql_content, encoding="utf-8")

            options = SqlScriptParseOptions(max_chars_per_chunk=80)
            result = DeterministicSqlScriptParser(options).parse_file(
                file_path,
                relative_path="scripts/init.sql",
            )

        self.assertEqual("sql", result.file_format)
        self.assertEqual(3, len(result.sections))
        self.assertIn("CREATE TABLE users", result.sections[0].content)
        self.assertIn("Alice", result.sections[1].content)
        self.assertIn("Bob", result.sections[2].content)
        self.assertIsNone(result.file_summary)


class SqlScriptIndexRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = SqlScriptIndexRepository(connection)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "query.sql"
            file_path.write_text("SELECT * FROM x;", encoding="utf-8")
            result = DeterministicSqlScriptParser().parse_file(
                file_path,
                relative_path="query.sql",
            )

        repository.find_file("query.sql")
        repository.save(
            result,
            size_bytes=16,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.save_error(
            source_id=result.source_id,
            relative_path=result.relative_path,
            filename=result.filename,
            file_format=result.file_format,
            size_bytes=16,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            error_message="failed",
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.mark_missing("", datetime(2026, 7, 1, tzinfo=UTC))
        repository.mark_missing("scripts", datetime(2026, 7, 1, tzinfo=UTC))


class SqlScriptIndexingServiceTest(unittest.TestCase):
    def test_service_saves_results_and_flushes_vector_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "sql").mkdir()
            (root / "sql" / "a.sql").write_text("SELECT 1;", encoding="utf-8")
            (root / "sql" / "b.sql").write_text("SELECT 2;", encoding="utf-8")

            repository = FakeSqlScriptRepository()
            vector_store = FakeVectorStore()
            enricher = FakeSqlScriptEnricher()
            service = SqlScriptIndexingService(
                root,
                DeterministicSqlScriptParser(),
                repository,
                enricher=enricher,
                vector_store=vector_store,
                enrich_batch_size=2,
                vector_batch_size=2,
            )

            first = service.run("sql")
            second = service.run("sql")

        self.assertEqual(2, first["discovered_count"])
        self.assertEqual(2, first["indexed_count"])
        self.assertEqual(0, first["error_count"])
        self.assertGreaterEqual(first["vector_document_count"], 4)
        self.assertEqual(1, len(vector_store.batches))
        self.assertEqual(1, len(enricher.batch_calls))
        self.assertEqual(2, len(enricher.batch_calls[0]))
        self.assertEqual(2, second["unchanged_count"])
        self.assertEqual("Summary for a.sql", repository.saved_results[0].file_summary)

    def test_build_sql_script_documents_emits_file_and_section_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "query.sql"
            file_path.write_text("SELECT 1;", encoding="utf-8")
            result = DeterministicSqlScriptParser().parse_file(
                file_path,
                relative_path="sql/query.sql",
            )

        documents = build_sql_script_documents(result)

        self.assertEqual("file", documents[0].metadata["record_type"])
        self.assertEqual("section", documents[1].metadata["record_type"])


if __name__ == "__main__":
    unittest.main()
