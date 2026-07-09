from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.domain.indexing_models import JsonIndexResult, JsonSection
from lake_agent.indexing.json.deterministic import (
    DeterministicJsonParser,
    JsonParseOptions,
)
from lake_agent.indexing.json.service import JsonIndexingService
from lake_agent.indexing.json.vector_store import build_json_documents
from lake_agent.persistence.repositories import JsonIndexRepository


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
        if "FROM json_files WHERE relative_path = %s" in compact_query:
            return RecordingResult(self.rows.get(params[0]))
        return RecordingResult()


class FakeJsonRepository:
    def __init__(self) -> None:
        self.files: dict[str, dict] = {}
        self.saved_results = []
        self.saved_errors = []
        self.marked_missing = []

    def find_file(self, relative_path: str):
        return self.files.get(relative_path)

    def save(
        self,
        result,
        *,
        size_bytes: int,
        last_modified: datetime,
        indexed_at: datetime,
    ):
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


class FailingVectorStore:
    def add_documents(self, *, documents, ids):
        raise RuntimeError("embedding failed")


class DeterministicJsonParserTest(unittest.TestCase):
    def test_parser_flattens_nested_json_into_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "jobs.json"
            file_path.write_text(
                json.dumps(
                    {
                        "source": "LinkedIn",
                        "jobs": [
                            {
                                "title": "Data Engineer",
                                "skills": ["python", "sql"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = DeterministicJsonParser(
                JsonParseOptions(max_chars_per_section=90)
            ).parse_file(
                file_path,
                relative_path="test_json/jobs.json",
                source_id="source_jobs",
            )

        self.assertEqual("json", result.file_format)
        self.assertEqual("object", result.top_level_type)
        self.assertGreater(result.entry_count, 0)
        self.assertGreaterEqual(len(result.sections), 2)
        content = "\n".join(section.content for section in result.sections)
        self.assertIn("$.jobs[0].title: Data Engineer", content)
        self.assertIn("$.jobs[0].skills[0]: python", content)
        self.assertIn("$.source: LinkedIn", content)
        self.assertTrue(result.file_search_text)

    def test_parser_reads_json_lines_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "events.jsonl"
            file_path.write_text(
                '{"id": 1, "name": "created"}\n{"id": 2, "name": "updated"}\n',
                encoding="utf-8",
            )

            result = DeterministicJsonParser().parse_file(file_path)

        self.assertEqual("jsonl", result.file_format)
        self.assertEqual("array", result.top_level_type)
        content = "\n".join(section.content for section in result.sections)
        self.assertIn("$[0].name: created", content)
        self.assertIn("$[1].id: 2", content)


class JsonIndexRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = JsonIndexRepository(connection)
        result = JsonIndexResult(
            source_id="source_test",
            relative_path="test_json/sample.json",
            filename="sample.json",
            file_format="json",
            sections=[
                JsonSection(
                    section_id="section_test",
                    chunk_index=1,
                    path_start="$.source",
                    path_end="$.source",
                    entry_count=1,
                    content="$.source: LinkedIn",
                    char_count=18,
                    search_text="$.source: LinkedIn",
                )
            ],
            top_level_type="object",
            entry_count=1,
            max_depth=1,
        )

        now = datetime(2026, 7, 1, tzinfo=UTC)
        repository.find_file("test_json/sample.json")
        repository.save(
            result,
            size_bytes=42,
            last_modified=now,
            indexed_at=now,
        )
        repository.save_error(
            source_id=result.source_id,
            relative_path=result.relative_path,
            filename=result.filename,
            file_format=result.file_format,
            size_bytes=42,
            last_modified=now,
            error_message="failed",
            indexed_at=now,
        )
        repository.mark_missing("", now)
        repository.mark_missing("test_json", now)


class JsonIndexingServiceTest(unittest.TestCase):
    def test_service_saves_results_and_flushes_vector_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "test_json").mkdir()
            (root / "test_json" / "a.json").write_text(
                json.dumps({"source": "LinkedIn", "jobs": [{"title": "Engineer"}]}),
                encoding="utf-8",
            )
            (root / "test_json" / "b.ndjson").write_text(
                '{"id": 1}\n{"id": 2}\n',
                encoding="utf-8",
            )
            (root / "test_json" / "ignore.txt").write_text("ignore", encoding="utf-8")

            repository = FakeJsonRepository()
            vector_store = FakeVectorStore()
            service = JsonIndexingService(
                root,
                DeterministicJsonParser(),
                repository,
                vector_store=vector_store,
                vector_batch_size=2,
            )

            first = service.run("test_json")
            second = service.run("test_json")

        self.assertEqual(2, first["discovered_count"])
        self.assertEqual(2, first["indexed_count"])
        self.assertEqual(0, first["error_count"])
        self.assertGreaterEqual(first["vector_document_count"], 2)
        self.assertEqual(1, len(vector_store.batches))
        self.assertEqual(2, len(repository.saved_results))
        self.assertEqual(2, second["unchanged_count"])

    def test_vector_failure_marks_pending_file_as_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "test_json").mkdir()
            (root / "test_json" / "a.json").write_text(
                json.dumps({"source": "LinkedIn"}),
                encoding="utf-8",
            )

            repository = FakeJsonRepository()
            service = JsonIndexingService(
                root,
                DeterministicJsonParser(),
                repository,
                vector_store=FailingVectorStore(),
            )

            result = service.run("test_json")

        self.assertEqual(0, result["indexed_count"])
        self.assertEqual(1, result["error_count"])
        self.assertEqual(1, len(repository.saved_errors))
        self.assertEqual("error", repository.files["test_json/a.json"]["status"])
        self.assertIn("embedding failed", result["errors"][0].message)

    def test_service_can_index_without_vector_store(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "test_json").mkdir()
            (root / "test_json" / "a.json").write_text(
                json.dumps({"source": "LinkedIn"}),
                encoding="utf-8",
            )

            repository = FakeJsonRepository()
            service = JsonIndexingService(
                root,
                DeterministicJsonParser(),
                repository,
            )

            result = service.run("test_json")

        self.assertEqual(1, result["indexed_count"])
        self.assertEqual(0, result["vector_document_count"])
        self.assertEqual(1, len(repository.saved_results))

    def test_build_json_documents_emits_file_and_section_docs(self) -> None:
        result = JsonIndexResult(
            source_id="source_test",
            relative_path="test_json/sample.json",
            filename="sample.json",
            file_format="json",
            sections=[
                JsonSection(
                    section_id="section_test",
                    chunk_index=1,
                    path_start="$.source",
                    path_end="$.source",
                    entry_count=1,
                    content="$.source: LinkedIn",
                    char_count=18,
                    search_text="$.source: LinkedIn",
                )
            ],
            top_level_type="object",
            entry_count=1,
            max_depth=1,
            file_search_text="sample.json\ntest_json/sample.json",
        )

        documents = build_json_documents(result)

        self.assertEqual("file", documents[0].metadata["record_type"])
        self.assertEqual("section", documents[1].metadata["record_type"])
        self.assertEqual("$.source", documents[1].metadata["path_start"])
        self.assertEqual("$.source: LinkedIn", documents[1].page_content)


if __name__ == "__main__":
    unittest.main()
