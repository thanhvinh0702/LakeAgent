from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.indexing.web import (
    DeterministicWebParser,
    WebIndexingService,
    build_web_documents,
)
from lake_agent.persistence.repositories import WebIndexRepository


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
        if "FROM web_files WHERE relative_path = %s" in compact_query:
            return RecordingResult(self.rows.get(params[0]))
        return RecordingResult()


class FakeWebRepository:
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


class FakeWebEnricher:
    def __init__(self) -> None:
        self.batch_calls = []

    def enrich(self, result):
        result.file_summary = f"Summary for {result.filename}"
        result.file_keywords = ["web", "html"]
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


class DeterministicWebParserTest(unittest.TestCase):
    def test_html_parser_extracts_headings_paragraphs_and_tables(self) -> None:
        html_content = (
            "<html>\n"
            "<body>\n"
            "<h1>Page Title</h1>\n"
            "<p>Introduction paragraph with <strong>bold</strong> text.</p>\n"
            "<h2>Data Table</h2>\n"
            "<table>\n"
            "  <tr><th>Month</th><th>Sales</th></tr>\n"
            "  <tr><td>Jan</td><td>100</td></tr>\n"
            "  <tr><td>Feb</td><td>200</td></tr>\n"
            "</table>\n"
            "</body>\n"
            "</html>\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "page.html"
            file_path.write_text(html_content, encoding="utf-8")

            result = DeterministicWebParser().parse_file(
                file_path,
                relative_path="web/page.html",
            )

        self.assertEqual("html", result.file_format)
        self.assertEqual(2, len(result.sections))
        self.assertEqual("Page Title", result.sections[0].heading)
        self.assertEqual("Data Table", result.sections[1].heading)
        self.assertIn("Introduction paragraph with bold text.", result.sections[0].content)
        self.assertIn("| Jan | 100 |", result.sections[1].content)
        self.assertIsNone(result.file_summary)


class WebIndexRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = WebIndexRepository(connection)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "page.html"
            file_path.write_text("<h1>Title</h1><p>hello</p>", encoding="utf-8")
            result = DeterministicWebParser().parse_file(
                file_path,
                relative_path="page.html",
            )

        repository.find_file("page.html")
        repository.save(
            result,
            size_bytes=26,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.save_error(
            source_id=result.source_id,
            relative_path=result.relative_path,
            filename=result.filename,
            file_format=result.file_format,
            size_bytes=26,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            error_message="failed",
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.mark_missing("", datetime(2026, 7, 1, tzinfo=UTC))
        repository.mark_missing("web", datetime(2026, 7, 1, tzinfo=UTC))


class WebIndexingServiceTest(unittest.TestCase):
    def test_service_saves_results_and_flushes_vector_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "web").mkdir()
            (root / "web" / "a.html").write_text("<h1>A</h1><p>content a</p>", encoding="utf-8")
            (root / "web" / "b.htm").write_text("<h1>B</h1><p>content b</p>", encoding="utf-8")

            repository = FakeWebRepository()
            vector_store = FakeVectorStore()
            enricher = FakeWebEnricher()
            service = WebIndexingService(
                root,
                DeterministicWebParser(),
                repository,
                enricher=enricher,
                vector_store=vector_store,
                enrich_batch_size=2,
                vector_batch_size=2,
            )

            first = service.run("web")
            second = service.run("web")

        self.assertEqual(2, first["discovered_count"])
        self.assertEqual(2, first["indexed_count"])
        self.assertEqual(0, first["error_count"])
        self.assertGreaterEqual(first["vector_document_count"], 4)
        self.assertEqual(1, len(vector_store.batches))
        self.assertEqual(1, len(enricher.batch_calls))
        self.assertEqual(2, len(enricher.batch_calls[0]))
        self.assertEqual(2, second["unchanged_count"])
        self.assertEqual("Summary for a.html", repository.saved_results[0].file_summary)

    def test_build_web_documents_emits_file_and_section_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "page.html"
            file_path.write_text("<h1>Title</h1><p>content</p>", encoding="utf-8")
            result = DeterministicWebParser().parse_file(
                file_path,
                relative_path="web/page.html",
            )

        documents = build_web_documents(result)

        self.assertEqual("file", documents[0].metadata["record_type"])
        self.assertEqual("section", documents[1].metadata["record_type"])
        self.assertEqual("Title", documents[1].metadata["heading"])


if __name__ == "__main__":
    unittest.main()
