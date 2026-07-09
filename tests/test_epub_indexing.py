from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from lake_agent.indexing.epub import (
    DeterministicEpubParser,
    EpubLLMEnricher,
    EpubParseOptions,
    build_epub_documents,
)
from lake_agent.domain.indexing_models import EnrichedEpubResult


class EpubIndexingTests(unittest.TestCase):
    def test_epub_parser_extracts_spine_text_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            epub_path = Path(tmp) / "sample.epub"
            _write_sample_epub(epub_path)

            result = DeterministicEpubParser(
                EpubParseOptions(extract_images=False)
            ).parse_file(
                epub_path,
                relative_path="books/sample.epub",
                source_id="source_epub",
            )

            self.assertEqual("Sample Book", result.title)
            self.assertEqual(["Author A"], result.creators)
            self.assertEqual("en", result.language)
            self.assertEqual(1, result.chapter_count)
            self.assertGreaterEqual(len(result.sections), 1)
            self.assertEqual("chapter_text", result.sections[0].section_type)
            self.assertEqual(1, result.sections[0].chapter_index)
            self.assertIn("Hello EPUB world", result.sections[0].search_text or "")

    def test_epub_parser_extracts_embedded_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            epub_path = Path(tmp) / "sample.epub"
            _write_sample_epub(epub_path)

            result = DeterministicEpubParser(
                EpubParseOptions(extract_images=True, max_images_per_file=1)
            ).parse_file(
                epub_path,
                relative_path="books/sample.epub",
                source_id="source_epub",
            )

            try:
                self.assertEqual(1, len(result.embedded_images))
                image = result.embedded_images[0]
                self.assertEqual("OPS/images/cover.jpg", image.href)
                self.assertEqual(12, image.width)
                self.assertEqual(8, image.height)
                self.assertTrue(Path(image.path).exists())
            finally:
                if result.artifact_dir:
                    import shutil

                    shutil.rmtree(result.artifact_dir, ignore_errors=True)

    def test_build_epub_documents_emits_file_and_section_docs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            epub_path = Path(tmp) / "sample.epub"
            _write_sample_epub(epub_path)
            result = DeterministicEpubParser(
                EpubParseOptions(extract_images=False)
            ).parse_file(
                epub_path,
                relative_path="books/sample.epub",
                source_id="source_epub",
            )

            documents = build_epub_documents(result)

            self.assertEqual("file", documents[0].metadata["record_type"])
            self.assertEqual("section", documents[1].metadata["record_type"])
            self.assertEqual("epub", documents[1].metadata["file_format"])
            self.assertEqual("chapter_text", documents[1].metadata["section_type"])
            self.assertEqual(1, documents[1].metadata["chapter_index"])

    def test_epub_enricher_sets_summary_keywords_and_file_search_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            epub_path = Path(tmp) / "sample.epub"
            _write_sample_epub(epub_path)
            result = DeterministicEpubParser(
                EpubParseOptions(extract_images=False)
            ).parse_file(
                epub_path,
                relative_path="books/sample.epub",
                source_id="source_epub",
            )
            enricher = EpubLLMEnricher(
                invoke_enrichment=lambda _system, _user: EnrichedEpubResult(
                    file_summary="A short summary of the sample EPUB.",
                    file_keywords=["sample", "epub"],
                )
            )

            enriched = enricher.enrich(result)

            self.assertEqual("A short summary of the sample EPUB.", enriched.file_summary)
            self.assertEqual(["sample", "epub"], enriched.file_keywords)
            self.assertIn("A short summary", enriched.file_search_text or "")


def _write_sample_epub(path: Path) -> None:
    image_path = path.with_suffix(".jpg")
    Image.new("RGB", (12, 8), (20, 40, 80)).save(image_path)
    image_bytes = image_path.read_bytes()

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr(
            "META-INF/container.xml",
            """<?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OPS/content.opf" media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>
            """,
        )
        archive.writestr(
            "OPS/content.opf",
            """<?xml version="1.0"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:title>Sample Book</dc:title>
                <dc:creator>Author A</dc:creator>
                <dc:language>en</dc:language>
                <dc:identifier>sample-id</dc:identifier>
              </metadata>
              <manifest>
                <item id="chap1" href="chap1.xhtml" media-type="application/xhtml+xml"/>
                <item id="cover" href="images/cover.jpg" media-type="image/jpeg" properties="cover-image"/>
              </manifest>
              <spine>
                <itemref idref="chap1"/>
              </spine>
            </package>
            """,
        )
        archive.writestr(
            "OPS/chap1.xhtml",
            """<html xmlns="http://www.w3.org/1999/xhtml">
              <head><title>Chapter One</title></head>
              <body><h1>Chapter One</h1><p>Hello EPUB world.</p><p>Second paragraph.</p></body>
            </html>
            """,
        )
        archive.writestr("OPS/images/cover.jpg", image_bytes)


if __name__ == "__main__":
    unittest.main()
