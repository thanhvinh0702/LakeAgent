from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.indexing.document import (
    DeterministicDocumentParser,
    DocumentIndexingService,
    build_document_documents,
)
from lake_agent.indexing.document.vector_store import (
    _ensure_approx_index,
    _ensure_hnsw_index,
)
from lake_agent.persistence.repositories import DocumentIndexRepository


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
        if "FROM document_files WHERE relative_path = %s" in compact_query:
            return RecordingResult(self.rows.get(params[0]))
        return RecordingResult()


class FakeDocumentRepository:
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


class FakeIndexableVectorStore:
    def __init__(self, valid: bool = False) -> None:
        self.valid = valid
        self.checked_names = []
        self.applied = []

    def is_valid_index(self, index_name=None):
        self.checked_names.append(index_name)
        return self.valid

    def apply_vector_index(self, index, name=None, concurrently=False):
        self.applied.append((index, name, concurrently))


class FakeDocumentEnricher:
    def __init__(self) -> None:
        self.batch_calls = []

    def enrich(self, result):
        result.file_summary = f"Summary for {result.filename}"
        result.file_keywords = ["document", result.file_format]
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


class FakeProv:
    def __init__(self, page_no: int) -> None:
        self.page_no = page_no


class FakeDocItem:
    def __init__(self, page_no: int) -> None:
        self.prov = [FakeProv(page_no)]


class FakeMeta:
    def __init__(self, headings: list[str], page_no: int) -> None:
        self.headings = headings
        self.doc_items = [FakeDocItem(page_no)]


class FakeChunk:
    def __init__(self, text: str, headings: list[str], page_no: int) -> None:
        self.text = text
        self.meta = FakeMeta(headings, page_no)


class FakeChunker:
    def chunk(self, dl_doc=None):
        return [
            FakeChunk("Overview paragraph.", ["Intro"], 1),
            FakeChunk("Detailed findings.", ["Intro", "Findings"], 2),
        ]

    def contextualize(self, chunk):
        heading = " > ".join(chunk.meta.headings)
        return f"{heading}\n{chunk.text}".strip()


def _fake_prepare_source(path: Path):
    from lake_agent.indexing.document.deterministic import _PreparedSource

    return _PreparedSource(path=path, warnings=[])


class DeterministicDocumentParserTest(unittest.TestCase):
    def test_parser_builds_sections_from_docling_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "report.pdf"
            file_path.write_bytes(b"%PDF-1.4 fake")

            result = DeterministicDocumentParser(
                load_document=lambda path: object(),
                build_chunker=lambda: FakeChunker(),
                prepare_source=_fake_prepare_source,
            ).parse_file(
                file_path,
                relative_path="docs/report.pdf",
            )

        self.assertEqual("pdf", result.file_format)
        self.assertEqual(2, len(result.sections))
        self.assertEqual("Intro", result.sections[0].heading)
        self.assertEqual("Intro > Findings", result.sections[1].heading)
        self.assertEqual("Intro\nOverview paragraph.", result.sections[0].search_text)
        self.assertEqual(1, result.sections[0].page_start)
        self.assertEqual(2, result.sections[1].page_start)


class DocumentIndexRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = DocumentIndexRepository(connection)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "report.pdf"
            file_path.write_bytes(b"%PDF-1.4 fake")
            result = DeterministicDocumentParser(
                load_document=lambda path: object(),
                build_chunker=lambda: FakeChunker(),
                prepare_source=_fake_prepare_source,
            ).parse_file(
                file_path,
                relative_path="docs/report.pdf",
            )

        repository.find_file("docs/report.pdf")
        repository.save(
            result,
            size_bytes=12,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.save_error(
            source_id=result.source_id,
            relative_path=result.relative_path,
            filename=result.filename,
            file_format=result.file_format,
            size_bytes=12,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            error_message="failed",
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.mark_missing("", datetime(2026, 7, 1, tzinfo=UTC))
        repository.mark_missing("docs", datetime(2026, 7, 1, tzinfo=UTC))


class DocumentIndexingServiceTest(unittest.TestCase):
    def test_service_saves_results_and_flushes_vector_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "docs").mkdir()
            (root / "docs" / "a.pdf").write_bytes(b"%PDF-1.4 fake a")
            (root / "docs" / "b.docx").write_bytes(b"fake docx")
            (root / "docs" / "ignore.md").write_text("# ignore", encoding="utf-8")

            repository = FakeDocumentRepository()
            vector_store = FakeVectorStore()
            enricher = FakeDocumentEnricher()
            service = DocumentIndexingService(
                root,
                DeterministicDocumentParser(
                    load_document=lambda path: object(),
                    build_chunker=lambda: FakeChunker(),
                    prepare_source=_fake_prepare_source,
                ),
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
        self.assertEqual("Summary for a.pdf", repository.saved_results[0].file_summary)
        self.assertEqual("Intro\nOverview paragraph.", repository.saved_results[0].sections[0].search_text)

    def test_build_document_documents_emits_file_and_section_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "report.pdf"
            file_path.write_bytes(b"%PDF-1.4 fake")
            result = DeterministicDocumentParser(
                load_document=lambda path: object(),
                build_chunker=lambda: FakeChunker(),
                prepare_source=_fake_prepare_source,
            ).parse_file(
                file_path,
                relative_path="docs/report.pdf",
            )

        documents = build_document_documents(result)

        self.assertEqual("file", documents[0].metadata["record_type"])
        self.assertEqual("section", documents[1].metadata["record_type"])
        self.assertEqual("Intro", documents[1].metadata["heading"])
        self.assertEqual(1, documents[1].metadata["page_start"])

    def test_ensure_hnsw_index_only_creates_missing_index(self) -> None:
        missing_store = FakeIndexableVectorStore(valid=False)
        existing_store = FakeIndexableVectorStore(valid=True)

        _ensure_hnsw_index(missing_store, table_name="document_index")
        _ensure_hnsw_index(existing_store, table_name="document_index")

        self.assertEqual(["document_index_hnsw_idx"], missing_store.checked_names)
        self.assertEqual(1, len(missing_store.applied))
        self.assertEqual("document_index_hnsw_idx", missing_store.applied[0][1])
        self.assertEqual(["document_index_hnsw_idx"], existing_store.checked_names)
        self.assertEqual([], existing_store.applied)

    def test_ensure_approx_index_falls_back_to_ivfflat_for_high_dimensions(self) -> None:
        store = FakeIndexableVectorStore(valid=False)

        _ensure_approx_index(
            store,
            table_name="document_index",
            embedding_dimensions=3072,
        )

        self.assertEqual(["document_index_ivfflat_idx"], store.checked_names)
        self.assertEqual(1, len(store.applied))
        self.assertEqual("document_index_ivfflat_idx", store.applied[0][1])


if __name__ == "__main__":
    unittest.main()
