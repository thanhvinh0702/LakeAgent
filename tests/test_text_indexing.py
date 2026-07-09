from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.indexing.text import (
    DeterministicTextParser,
    TextIndexingService,
    build_text_documents,
)
from lake_agent.indexing.text.chunking import build_basic_search_text
from lake_agent.persistence.repositories import TextIndexRepository


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
        if "FROM text_files WHERE relative_path = %s" in compact_query:
            return RecordingResult(self.rows.get(params[0]))
        return RecordingResult()


class FakeTextRepository:
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


class FakeTextEnricher:
    def __init__(self) -> None:
        self.batch_calls = []

    def enrich(self, result):
        result.file_summary = f"Summary for {result.filename}"
        result.file_keywords = ["notes", "text"]
        for section in result.sections:
            section.search_text = build_basic_search_text(
                section.heading,
                section.content,
            )
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


class DeterministicTextParserTest(unittest.TestCase):
    def test_markdown_parser_splits_sections_by_heading(self) -> None:
        content = (
            "# Overview\n"
            "Lake overview paragraph.\n\n"
            "## Measurements\n"
            "Temperature values and notes.\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "report.md"
            file_path.write_text(content, encoding="utf-8")

            result = DeterministicTextParser().parse_file(
                file_path,
                relative_path="docs/report.md",
            )

        self.assertEqual("md", result.file_format)
        self.assertEqual(2, len(result.sections))
        self.assertEqual("Overview", result.sections[0].heading)
        self.assertEqual("Measurements", result.sections[1].heading)
        self.assertIn("Lake overview paragraph.", result.sections[0].content)
        self.assertEqual(
            "Overview\nLake overview paragraph.",
            result.sections[0].search_text,
        )
        self.assertIsNone(result.file_summary)

    def test_text_parser_chunks_large_plain_text(self) -> None:
        paragraph = "This is a sentence about field notes. " * 80

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "notes.txt"
            file_path.write_text(paragraph, encoding="utf-8")

            result = DeterministicTextParser().parse_file(file_path)

        self.assertEqual("txt", result.file_format)
        self.assertGreaterEqual(len(result.sections), 2)
        self.assertTrue(all(section.content for section in result.sections))
        self.assertTrue(all(section.search_text == section.content for section in result.sections))
        self.assertIsNone(result.file_summary)

    def test_text_parser_uses_recursive_chunking_for_plain_text(self) -> None:
        paragraphs = [
            "Paragraph one discusses the library renovation timeline and community use. " * 8,
            "Paragraph two covers digital resources, study rooms, and mobile app access. " * 8,
            "Paragraph three explains preserved furniture, lighting upgrades, and layout changes. "
            * 8,
            "Paragraph four summarizes outcomes, visitor growth, and continued print borrowing. "
            * 8,
        ]
        content = "\n\n".join(paragraphs)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "story.txt"
            file_path.write_text(content, encoding="utf-8")

            result = DeterministicTextParser().parse_file(file_path)

        self.assertGreaterEqual(len(result.sections), 2)
        self.assertTrue(all(section.char_count <= 1000 for section in result.sections))
        self.assertIn("Paragraph one discusses", result.sections[0].content)
        self.assertIn("Paragraph four summarizes", result.sections[-1].content)


class TextIndexRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = TextIndexRepository(connection)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "notes.txt"
            file_path.write_text("alpha\n\nbeta\n", encoding="utf-8")
            result = DeterministicTextParser().parse_file(
                file_path,
                relative_path="notes.txt",
            )

        repository.find_file("notes.txt")
        repository.save(
            result,
            size_bytes=11,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.save_error(
            source_id=result.source_id,
            relative_path=result.relative_path,
            filename=result.filename,
            file_format=result.file_format,
            size_bytes=11,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            error_message="failed",
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.mark_missing("", datetime(2026, 7, 1, tzinfo=UTC))
        repository.mark_missing("docs", datetime(2026, 7, 1, tzinfo=UTC))


class TextIndexingServiceTest(unittest.TestCase):
    def test_service_saves_results_and_flushes_vector_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "docs" / "a.txt").write_text("alpha\n\nbeta\n", encoding="utf-8")
            (root / "docs" / "b.md").write_text("# Title\n\nbody\n", encoding="utf-8")
            (root / "docs" / "ignore.csv").write_text("a,b\n1,2\n", encoding="utf-8")

            repository = FakeTextRepository()
            vector_store = FakeVectorStore()
            enricher = FakeTextEnricher()
            service = TextIndexingService(
                root,
                DeterministicTextParser(),
                repository,
                enricher=enricher,
                vector_store=vector_store,
                enrich_batch_size=2,
                vector_batch_size=2,
            )

            first = service.run("docs")
            second = service.run("docs")

        self.assertEqual(2, first["discovered_count"])
        self.assertEqual(2, first["indexed_count"])
        self.assertEqual(0, first["error_count"])
        self.assertGreaterEqual(first["vector_document_count"], 4)
        self.assertEqual(1, len(vector_store.batches))
        self.assertEqual(1, len(enricher.batch_calls))
        self.assertEqual(2, len(enricher.batch_calls[0]))
        self.assertEqual(2, second["unchanged_count"])
        self.assertEqual("Summary for a.txt", repository.saved_results[0].file_summary)
        self.assertEqual(
            repository.saved_results[0].sections[0].content,
            repository.saved_results[0].sections[0].search_text,
        )
        self.assertEqual(
            "Title\nbody",
            repository.saved_results[1].sections[0].search_text,
        )

    def test_build_text_documents_emits_file_and_section_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "guide.md"
            file_path.write_text("# Intro\n\nhello world\n", encoding="utf-8")
            result = DeterministicTextParser().parse_file(
                file_path,
                relative_path="docs/guide.md",
            )

        documents = build_text_documents(result)

        self.assertEqual("file", documents[0].metadata["record_type"])
        self.assertEqual("section", documents[1].metadata["record_type"])
        self.assertEqual("Intro", documents[1].metadata["heading"])


if __name__ == "__main__":
    unittest.main()
