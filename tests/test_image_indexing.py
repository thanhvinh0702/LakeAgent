from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from lake_agent.indexing.image import (
    DeterministicImageParser,
    ImageIndexingService,
    build_image_documents,
)
from lake_agent.indexing.image.ocr import OCRMarkdownExtractor
from lake_agent.persistence.repositories import ImageIndexRepository


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
        if "FROM image_files WHERE relative_path = %s" in compact_query:
            return RecordingResult(self.rows.get(params[0]))
        return RecordingResult()


class FakeImageRepository:
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


class FakeOCRExtractor:
    batch_size = 2

    def extract_sections(self, image_path, *, source_id: str):
        return self.extract_sections_batch([image_path], source_ids=[source_id])[source_id]

    def extract_sections_batch(self, image_paths, *, source_ids):
        extractor = OCRMarkdownExtractor(
            client=FakeOCRClient(),
            handle_file=lambda path: path,
        )
        return {
            source_id: extractor.extract_sections(str(image_path), source_id=source_id)
            for image_path, source_id in zip(image_paths, source_ids, strict=True)
        }


class FakeOCRClient:
    def predict(self, files, api_name):
        return ["# Title\n\nFirst paragraph.\n\n## Detail\n\nSecond paragraph." for _ in files]


class FakeVLMEnricher:
    batch_size = 2

    def enrich_batch(self, image_paths, results):
        for image_path, result in zip(image_paths, results, strict=True):
            stem = Path(image_path).stem.replace("_", " ")
            result.file_summary = f"Image of {stem}"
            result.file_keywords = ["image", stem]
            result.file_search_text = "\n".join(
                part
                for part in [
                    result.filename,
                    result.relative_path,
                    result.file_format,
                    f"{result.width}x{result.height}",
                    result.color_mode,
                    result.file_summary,
                    ", ".join(result.file_keywords),
                ]
                if part
            )
        return results


def _require_pillow() -> None:
    try:
        import PIL  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise unittest.SkipTest("Pillow is not installed") from exc


class DeterministicImageParserTest(unittest.TestCase):
    def test_png_parser_extracts_minimal_metadata(self) -> None:
        _require_pillow()
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sample.png"
            Image.new("RGBA", (12, 8), (255, 0, 0, 128)).save(file_path)

            result = DeterministicImageParser().parse_file(
                file_path,
                relative_path="images/sample.png",
            )

        self.assertEqual("png", result.file_format)
        self.assertEqual(12, result.width)
        self.assertEqual(8, result.height)
        self.assertEqual("RGBA", result.color_mode)
        self.assertTrue(result.has_alpha)
        self.assertFalse(result.is_animated)
        self.assertEqual(1, result.frame_count)


class ImageIndexRepositoryTest(unittest.TestCase):
    def test_repository_statements_have_matching_parameter_counts(self) -> None:
        connection = PlaceholderCheckingConnection()
        repository = ImageIndexRepository(connection)

        result = _build_fake_result()

        repository.find_file("images/sample.png")
        repository.save(
            result,
            size_bytes=10,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.save_error(
            source_id=result.source_id,
            relative_path=result.relative_path,
            filename=result.filename,
            file_format=result.file_format,
            size_bytes=10,
            last_modified=datetime(2026, 7, 1, tzinfo=UTC),
            error_message="failed",
            indexed_at=datetime(2026, 7, 1, tzinfo=UTC),
        )
        repository.mark_missing("", datetime(2026, 7, 1, tzinfo=UTC))
        repository.mark_missing("images", datetime(2026, 7, 1, tzinfo=UTC))


class ImageIndexingServiceTest(unittest.TestCase):
    def test_service_saves_results_and_skips_unchanged_files(self) -> None:
        _require_pillow()
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "images").mkdir()
            Image.new("RGB", (16, 9), (0, 0, 255)).save(root / "images" / "a.jpg")
            Image.new("RGB", (20, 20), (0, 255, 0)).save(root / "images" / "b.png")

            repository = FakeImageRepository()
            vector_store = FakeVectorStore()
            service = ImageIndexingService(
                root,
                DeterministicImageParser(),
                repository,
                ocr_extractor=FakeOCRExtractor(),
                vlm_enricher=FakeVLMEnricher(),
                vector_store=vector_store,
                vector_batch_size=2,
            )

            first = service.run("images")
            second = service.run("images")

        self.assertEqual(2, first["discovered_count"])
        self.assertEqual(2, first["indexed_count"])
        self.assertEqual(0, first["error_count"])
        self.assertGreaterEqual(first["vector_document_count"], 4)
        self.assertEqual(1, len(vector_store.batches))
        self.assertEqual(2, second["unchanged_count"])
        self.assertEqual(2, len(repository.saved_results[0].sections))
        self.assertEqual("Title", repository.saved_results[0].sections[0].heading)
        self.assertEqual("Image of a", repository.saved_results[0].file_summary)
        self.assertIn("Image of a", repository.saved_results[0].file_search_text)

    def test_build_image_documents_emits_file_and_section_docs(self) -> None:
        result = _build_fake_result()
        result.sections = [
            _build_fake_section(
                section_id="imgsec_1",
                content="Visible text",
                heading="Heading",
            )
        ]

        documents = build_image_documents(result)

        self.assertEqual("file", documents[0].metadata["record_type"])
        self.assertEqual("section", documents[1].metadata["record_type"])
        self.assertEqual("Heading", documents[1].metadata["heading"])
        self.assertEqual("ocr_chunk", documents[1].metadata["section_type"])

def _build_fake_result():
    from lake_agent.domain.indexing_models import ImageIndexResult

    return ImageIndexResult(
        source_id="source_test",
        relative_path="images/sample.png",
        filename="sample.png",
        file_format="png",
        width=12,
        height=8,
        color_mode="RGBA",
        has_alpha=True,
        is_animated=False,
        frame_count=1,
        file_search_text="sample.png\nimages/sample.png\npng\n12x8\nRGBA\ntransparent",
    )


def _build_fake_section(*, section_id: str, content: str, heading: str | None):
    from lake_agent.domain.indexing_models import ImageSection

    return ImageSection(
        section_id=section_id,
        section_type="ocr_chunk",
        chunk_index=1,
        heading=heading,
        content=content,
        char_count=len(content),
        search_text=content,
    )


if __name__ == "__main__":
    unittest.main()
