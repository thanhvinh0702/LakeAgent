from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.indexing.slideshow import (
    DeterministicSlideshowParser,
    SlideshowIndexingService,
    build_slideshow_documents,
)
from lake_agent.persistence.repositories import SlideshowIndexRepository


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
        if "FROM slideshow_files WHERE relative_path = %s" in compact_query:
            return RecordingResult(self.rows.get(params[0]))
        return RecordingResult()


class FakeSlideshowRepository:
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


class FakeSlideshowEnricher:
    def __init__(self) -> None:
        self.batch_calls = []

    def enrich(self, result):
        result.file_summary = f"Summary for {result.filename}"
        result.file_keywords = ["slideshow", result.file_format]
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


class FakeSlideshowImageProcessor:
    def enrich_batch(self, results):
        for result in results:
            result.sections.append(
                type(result.sections[0])(
                    section_id=f"{result.source_id}:image-summary",
                    section_type="image_summary",
                    chunk_index=len(result.sections) + 1,
                    heading="Embedded slide image",
                    content="A summary of the embedded slide image.",
                    slide_start=1,
                    slide_end=1,
                    char_count=38,
                    search_text="Embedded slide image\nA summary of the embedded slide image.",
                    image_id=f"{result.source_id}:image-1",
                    image_index=1,
                )
            )
        return results


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
            FakeChunk("Opening slide.", ["Launch Plan"], 1),
            FakeChunk("Metrics and timeline.", ["Launch Plan", "Timeline"], 2),
        ]

    def contextualize(self, chunk):
        heading = " > ".join(chunk.meta.headings)
        return f"{heading}\n{chunk.text}".strip()


def _fake_prepare_source(path: Path):
    from lake_agent.indexing.slideshow.deterministic import _PreparedSource

    return _PreparedSource(path=path, warnings=[])


class DeterministicSlideshowParserTest(unittest.TestCase):
    def test_parser_builds_sections_from_docling_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "deck.pptx"
            file_path.write_bytes(b"fake pptx")

            result = DeterministicSlideshowParser(
                load_document=lambda path: object(),
                build_chunker=lambda: FakeChunker(),
                prepare_source=_fake_prepare_source,
            ).parse_file(
                file_path,
                relative_path="slides/deck.pptx",
            )

        self.assertEqual("pptx", result.file_format)
        self.assertEqual(2, len(result.sections))
        self.assertEqual("Launch Plan", result.sections[0].heading)
        self.assertEqual("slide_chunk", result.sections[0].section_type)
        self.assertEqual("Launch Plan > Timeline", result.sections[1].heading)
        self.assertEqual("Launch Plan\nOpening slide.", result.sections[0].search_text)
        self.assertEqual(1, result.sections[0].slide_start)
        self.assertEqual(2, result.sections[1].slide_start)

    def test_parser_keeps_original_ppt_format_after_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "legacy.ppt"
            file_path.write_bytes(b"fake ppt")

            result = DeterministicSlideshowParser(
                load_document=lambda path: object(),
                build_chunker=lambda: FakeChunker(),
                prepare_source=_fake_prepare_source,
            ).parse_file(
                file_path,
                relative_path="slides/legacy.ppt",
            )

        self.assertEqual("ppt", result.file_format)

    def test_parser_merges_small_chunks_on_same_slide(self) -> None:
        class TinyChunker:
            def chunk(self, dl_doc=None):
                return [
                    FakeChunk("Alpha", ["Topic"], 1),
                    FakeChunk("Beta", ["Topic"], 1),
                    FakeChunk("Gamma", ["Topic"], 1),
                ]

            def contextualize(self, chunk):
                heading = " > ".join(chunk.meta.headings)
                return f"{heading}\n{chunk.text}".strip()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "deck.pptx"
            file_path.write_bytes(b"fake pptx")

            result = DeterministicSlideshowParser(
                load_document=lambda path: object(),
                build_chunker=lambda: TinyChunker(),
                prepare_source=_fake_prepare_source,
            ).parse_file(
                file_path,
                relative_path="slides/deck.pptx",
            )

        self.assertEqual(1, len(result.sections))
        self.assertEqual("Alpha\nBeta\nGamma", result.sections[0].content)

    def test_parser_does_not_merge_across_slides(self) -> None:
        class MultiSlideChunker:
            def chunk(self, dl_doc=None):
                return [
                    FakeChunk("Alpha", ["Topic"], 1),
                    FakeChunk("Beta", ["Topic"], 1),
                    FakeChunk("Gamma", ["Topic"], 2),
                    FakeChunk("Delta", ["Topic"], 2),
                ]

            def contextualize(self, chunk):
                heading = " > ".join(chunk.meta.headings)
                return f"{heading}\n{chunk.text}".strip()

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "deck.pptx"
            file_path.write_bytes(b"fake pptx")

            result = DeterministicSlideshowParser(
                load_document=lambda path: object(),
                build_chunker=lambda: MultiSlideChunker(),
                prepare_source=_fake_prepare_source,
            ).parse_file(
                file_path,
                relative_path="slides/deck.pptx",
            )

        self.assertEqual(2, len(result.sections))
        self.assertEqual(1, result.sections[0].slide_start)
        self.assertEqual(2, result.sections[1].slide_start)

    def test_parser_fails_fast_for_snap_libreoffice_on_ppt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "legacy.ppt"
            file_path.write_bytes(b"fake ppt")

            parser = DeterministicSlideshowParser()

            from unittest.mock import patch

            with patch(
                "lake_agent.indexing.slideshow.deterministic._resolve_office_binary",
                return_value="/snap/bin/libreoffice",
            ):
                with self.assertRaisesRegex(RuntimeError, "native LibreOffice/soffice binary"):
                    parser.parse_file(
                        file_path,
                        relative_path="slides/legacy.ppt",
                    )


class SlideshowIndexRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = SlideshowIndexRepository(connection)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "deck.pptx"
            file_path.write_bytes(b"fake pptx")
            result = DeterministicSlideshowParser(
                load_document=lambda path: object(),
                build_chunker=lambda: FakeChunker(),
                prepare_source=_fake_prepare_source,
            ).parse_file(
                file_path,
                relative_path="slides/deck.pptx",
            )

        repository.find_file("slides/deck.pptx")
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
        repository.mark_missing("slides", datetime(2026, 7, 1, tzinfo=UTC))


class SlideshowIndexingServiceTest(unittest.TestCase):
    def test_service_saves_results_and_flushes_vector_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "slides").mkdir()
            (root / "slides" / "a.pptx").write_bytes(b"fake pptx a")
            (root / "slides" / "b.ppt").write_bytes(b"fake ppt")
            (root / "slides" / "ignore.md").write_text("# ignore", encoding="utf-8")

            repository = FakeSlideshowRepository()
            vector_store = FakeVectorStore()
            enricher = FakeSlideshowEnricher()
            service = SlideshowIndexingService(
                root,
                DeterministicSlideshowParser(
                    load_document=lambda path: object(),
                    build_chunker=lambda: FakeChunker(),
                    prepare_source=_fake_prepare_source,
                ),
                repository,
                image_processor=FakeSlideshowImageProcessor(),
                enricher=enricher,
                vector_store=vector_store,
                enrich_batch_size=2,
                vector_batch_size=2,
            )

            first = service.run("slides")
            second = service.run("slides")

        self.assertEqual(2, first["discovered_count"])
        self.assertEqual(2, first["indexed_count"])
        self.assertEqual(0, first["error_count"])
        self.assertGreaterEqual(first["vector_document_count"], 4)
        self.assertEqual(1, len(vector_store.batches))
        self.assertEqual(1, len(enricher.batch_calls))
        self.assertEqual(2, len(enricher.batch_calls[0]))
        self.assertEqual(2, second["unchanged_count"])
        self.assertEqual("Summary for a.pptx", repository.saved_results[0].file_summary)
        self.assertEqual("Launch Plan\nOpening slide.", repository.saved_results[0].sections[0].search_text)
        self.assertEqual("image_summary", repository.saved_results[0].sections[-1].section_type)

    def test_build_slideshow_documents_emits_file_and_section_docs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "deck.pptx"
            file_path.write_bytes(b"fake pptx")
            result = DeterministicSlideshowParser(
                load_document=lambda path: object(),
                build_chunker=lambda: FakeChunker(),
                prepare_source=_fake_prepare_source,
            ).parse_file(
                file_path,
                relative_path="slides/deck.pptx",
            )

        documents = build_slideshow_documents(result)

        self.assertEqual("file", documents[0].metadata["record_type"])
        self.assertEqual("section", documents[1].metadata["record_type"])
        self.assertEqual("slide_chunk", documents[1].metadata["section_type"])
        self.assertEqual("Launch Plan", documents[1].metadata["heading"])
        self.assertEqual(1, documents[1].metadata["slide_start"])


if __name__ == "__main__":
    unittest.main()
