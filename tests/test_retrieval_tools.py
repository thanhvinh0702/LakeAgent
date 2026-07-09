from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from lake_agent.tools.retrieval import (
    IndexedDataRetriever,
    build_langchain_retrieval_tools,
)


class FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class FakeConnection:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], dict] = {}

    def add_row(self, marker: str, key: str, row: dict) -> None:
        self._rows[(marker, key)] = row

    def execute(self, query: str, params=None):
        params = params or ()
        compact = " ".join(query.split())
        for marker, key in self._rows:
            if marker in compact and params and params[0] == key:
                return FakeResult(self._rows[(marker, key)])
        return FakeResult(None)


class FakeVectorStore:
    def __init__(self, hits):
        self._hits = hits

    def similarity_search_with_score(self, query: str, k: int = 5):
        return self._hits[:k]


class FailingVectorStore:
    def __init__(self, message: str):
        self._message = message

    def similarity_search_with_score(self, query: str, k: int = 5):
        raise RuntimeError(self._message)


class RetrievalToolsTest(unittest.TestCase):
    def test_query_tabular_returns_true_table_content(self) -> None:
        connection = FakeConnection()
        connection.add_row(
            "FROM tabular_tables AS t JOIN tabular_files AS f ON f.source_id = t.source_id WHERE t.table_id = %s",
            "table_1",
            {
                "table_id": "table_1",
                "source_id": "source_1",
                "relative_path": "data/report.xlsx",
                "filename": "report.xlsx",
                "file_format": "xlsx",
                "table_name": "Sheet1",
                "sheet_name": "Sheet1",
                "sheet_description": "Revenue sheet",
                "header_row_index": 1,
                "raw_header": ["month", "revenue"],
                "row_count": 10,
                "column_count": 3,
                "summary": "Revenue by month.",
                "keywords": ["revenue"],
                "columns_json": [{"name": "month"}],
                "preview_rows": [["Jan", "10"]],
                "warnings": [],
            },
        )
        retriever = IndexedDataRetriever(
            connection,
            vector_stores={
                "tabular": FakeVectorStore(
                    [
                        (
                            Document(
                                page_content="ignored",
                                metadata={"record_type": "table", "table_id": "table_1"},
                            ),
                            0.12,
                        )
                    ]
                ),
                "text": FakeVectorStore([]),
                "document": FakeVectorStore([]),
                "slideshow": FakeVectorStore([]),
                "image": FakeVectorStore([]),
            },
            datalake_dir="/lake",
        )

        result = retriever.query_tabular("revenue", 3, 0)

        self.assertEqual("tabular", result["modality"])
        self.assertEqual("Revenue by month.", result["results"][0]["content"])
        self.assertEqual(
            "./data/report.xlsx",
            result["results"][0]["execution_file_path"],
        )
        self.assertEqual(
            "/lake/data/report.xlsx",
            result["results"][0]["absolute_file_path"],
        )
        self.assertEqual("Sheet1", result["results"][0]["sheet_name"])
        self.assertEqual(["month", "revenue"], result["results"][0]["header"])
        self.assertEqual(3, result["limit"])
        self.assertEqual(0, result["offset"])
        self.assertNotIn("table_id", result["results"][0])
        self.assertNotIn("row_count", result["results"][0])

    def test_query_all_merges_modalities_by_score(self) -> None:
        connection = FakeConnection()
        connection.add_row(
            "FROM text_sections AS s JOIN text_files AS f ON f.source_id = s.source_id WHERE s.section_id = %s",
            "text_section_1",
            {
                "section_id": "text_section_1",
                "source_id": "source_text",
                "relative_path": "notes/a.md",
                "filename": "a.md",
                "file_format": "md",
                "chunk_index": 1,
                "heading": "Intro",
                "content": "Alpha content",
                "search_text": "Intro\nAlpha content\nSummary context",
                "line_start": 1,
                "line_end": 4,
                "char_count": 13,
                "warnings": [],
            },
        )
        connection.add_row(
            "FROM image_files WHERE source_id = %s",
            "source_img",
            {
                "source_id": "source_img",
                "relative_path": "images/a.png",
                "filename": "a.png",
                "file_format": "png",
                "width": 100,
                "height": 100,
                "color_mode": "RGB",
                "has_alpha": False,
                "is_animated": False,
                "frame_count": 1,
                "file_summary": "Image of a chart",
                "file_keywords": ["chart"],
                "parse_warnings": [],
            },
        )
        retriever = IndexedDataRetriever(
            connection,
            vector_stores={
                "tabular": FakeVectorStore([]),
                "text": FakeVectorStore(
                    [
                        (
                            Document(
                                page_content="ignored",
                                metadata={"record_type": "section", "section_id": "text_section_1"},
                            ),
                            0.22,
                        )
                    ]
                ),
                "document": FakeVectorStore([]),
                "slideshow": FakeVectorStore([]),
                "image": FakeVectorStore(
                    [
                        (
                            Document(
                                page_content="ignored",
                                metadata={"record_type": "file", "source_id": "source_img"},
                            ),
                            0.08,
                        )
                    ]
                ),
            },
            datalake_dir="/lake",
        )

        result = retriever.query_all("chart", 2, 0)

        self.assertEqual(2, len(result["results"]))
        self.assertEqual("image", result["results"][0]["modality"])
        self.assertEqual(
            "./images/a.png",
            result["results"][0]["execution_file_path"],
        )
        self.assertEqual(
            "/lake/images/a.png",
            result["results"][0]["absolute_file_path"],
        )
        self.assertEqual("text", result["results"][1]["modality"])
        self.assertEqual("Intro\nAlpha content\nSummary context", result["results"][1]["content"])
        self.assertEqual(
            "./notes/a.md",
            result["results"][1]["execution_file_path"],
        )
        self.assertEqual(
            "/lake/notes/a.md",
            result["results"][1]["absolute_file_path"],
        )
        self.assertEqual(
            {"unit": "line", "start": 1, "end": 4},
            result["results"][1]["position"],
        )
        self.assertNotIn("section_id", result["results"][1])
        self.assertNotIn("chunk_index", result["results"][1])
        self.assertNotIn("heading", result["results"][1])

    def test_query_all_skips_missing_vector_table_modalities(self) -> None:
        connection = FakeConnection()
        connection.add_row(
            "FROM text_sections AS s JOIN text_files AS f ON f.source_id = s.source_id WHERE s.section_id = %s",
            "text_section_1",
            {
                "section_id": "text_section_1",
                "source_id": "source_text",
                "relative_path": "notes/a.md",
                "filename": "a.md",
                "file_format": "md",
                "chunk_index": 1,
                "heading": "Intro",
                "content": "Alpha content",
                "search_text": "Intro\nAlpha content\nSummary context",
                "line_start": 1,
                "line_end": 4,
                "char_count": 13,
                "warnings": [],
            },
        )
        retriever = IndexedDataRetriever(
            connection,
            vector_stores={
                "tabular": FakeVectorStore([]),
                "text": FakeVectorStore(
                    [
                        (
                            Document(
                                page_content="ignored",
                                metadata={"record_type": "section", "section_id": "text_section_1"},
                            ),
                            0.22,
                        )
                    ]
                ),
                "document": FakeVectorStore([]),
                "slideshow": FakeVectorStore([]),
                "image": FailingVectorStore('relation "public.image_index" does not exist'),
            },
        )

        result = retriever.query_all("chart", 3, 0)

        self.assertEqual(1, len(result["results"]))
        self.assertEqual("text", result["results"][0]["modality"])
        self.assertEqual(
            [{"modality": "image", "reason": "vector_table_missing"}],
            result["skipped_modalities"],
        )

    def test_query_slideshow_omits_noisy_section_metadata(self) -> None:
        connection = FakeConnection()
        connection.add_row(
            "FROM slideshow_sections AS s JOIN slideshow_files AS f ON f.source_id = s.source_id WHERE s.section_id = %s",
            "slide_section_1",
            {
                "section_id": "slide_section_1",
                "source_id": "source_slide",
                "relative_path": "slides/a.pptx",
                "filename": "a.pptx",
                "file_format": "pptx",
                "section_type": "slide_chunk",
                "chunk_index": 3,
                "heading": None,
                "content": "Useful slide content",
                "search_text": "Useful slide content\nSlide context",
                "slide_start": 3,
                "slide_end": 3,
                "char_count": 491,
                "image_id": None,
                "image_index": None,
                "warnings": [],
            },
        )
        retriever = IndexedDataRetriever(
            connection,
            vector_stores={
                "tabular": FakeVectorStore([]),
                "text": FakeVectorStore([]),
                "document": FakeVectorStore([]),
                "slideshow": FakeVectorStore(
                    [
                        (
                            Document(
                                page_content="ignored",
                                metadata={"record_type": "section", "section_id": "slide_section_1"},
                            ),
                            0.19,
                        )
                    ]
                ),
                "image": FakeVectorStore([]),
            },
            datalake_dir="/lake",
        )

        result = retriever.query_slideshow("topic", 3, 0)
        hit = result["results"][0]

        self.assertEqual("Useful slide content\nSlide context", hit["content"])
        self.assertEqual("./slides/a.pptx", hit["execution_file_path"])
        self.assertEqual("/lake/slides/a.pptx", hit["absolute_file_path"])
        self.assertEqual({"unit": "slide", "start": 3, "end": 3}, hit["position"])
        for key in (
            "section_id",
            "section_type",
            "chunk_index",
            "heading",
            "slide_start",
            "slide_end",
            "char_count",
            "image_id",
            "image_index",
        ):
            self.assertNotIn(key, hit)

    def test_query_slideshow_uses_search_text_for_section_content(self) -> None:
        connection = FakeConnection()
        connection.add_row(
            "FROM slideshow_sections AS s JOIN slideshow_files AS f ON f.source_id = s.source_id WHERE s.section_id = %s",
            "slide_section_2",
            {
                "section_id": "slide_section_2",
                "source_id": "source_slide",
                "relative_path": "slides/b.pptx",
                "filename": "b.pptx",
                "file_format": "pptx",
                "section_type": "slide_chunk",
                "chunk_index": 4,
                "heading": "Market Overview",
                "content": "Demand is rising across regions.",
                "search_text": "Market Overview\nDemand is rising across regions.\nSlide 4 context",
                "slide_start": 4,
                "slide_end": 4,
                "char_count": 31,
                "image_id": None,
                "image_index": None,
                "warnings": [],
            },
        )
        retriever = IndexedDataRetriever(
            connection,
            vector_stores={
                "tabular": FakeVectorStore([]),
                "text": FakeVectorStore([]),
                "document": FakeVectorStore([]),
                "slideshow": FakeVectorStore(
                    [
                        (
                            Document(
                                page_content="ignored",
                                metadata={"record_type": "section", "section_id": "slide_section_2"},
                            ),
                            0.11,
                        )
                    ]
                ),
                "image": FakeVectorStore([]),
            },
            datalake_dir="/lake",
        )

        result = retriever.query_slideshow("market", 3, 0)

        self.assertEqual(
            "Market Overview\nDemand is rising across regions.\nSlide 4 context",
            result["results"][0]["content"],
        )

    def test_query_text_supports_limit_and_offset_paging(self) -> None:
        connection = FakeConnection()
        for index, score in enumerate((0.11, 0.22, 0.33), start=1):
            connection.add_row(
                "FROM text_sections AS s JOIN text_files AS f ON f.source_id = s.source_id WHERE s.section_id = %s",
                f"text_section_{index}",
                {
                    "section_id": f"text_section_{index}",
                    "source_id": "source_text",
                    "relative_path": "notes/a.md",
                    "filename": "a.md",
                    "file_format": "md",
                    "chunk_index": index,
                    "heading": f"H{index}",
                    "content": f"C{index}",
                    "search_text": f"S{index}",
                    "line_start": index,
                    "line_end": index,
                    "char_count": 2,
                    "warnings": [],
                },
            )
        retriever = IndexedDataRetriever(
            connection,
            vector_stores={
                "tabular": FakeVectorStore([]),
                "text": FakeVectorStore(
                    [
                        (
                            Document(
                                page_content="ignored",
                                metadata={"record_type": "section", "section_id": "text_section_1"},
                            ),
                            0.11,
                        ),
                        (
                            Document(
                                page_content="ignored",
                                metadata={"record_type": "section", "section_id": "text_section_2"},
                            ),
                            0.22,
                        ),
                        (
                            Document(
                                page_content="ignored",
                                metadata={"record_type": "section", "section_id": "text_section_3"},
                            ),
                            0.33,
                        ),
                    ]
                ),
                "document": FakeVectorStore([]),
                "slideshow": FakeVectorStore([]),
                "image": FakeVectorStore([]),
            },
            datalake_dir="/lake",
        )

        result = retriever.query_text("query", 1, 1)

        self.assertEqual(1, result["limit"])
        self.assertEqual(1, result["offset"])
        self.assertEqual(1, result["returned_count"])
        self.assertEqual(2, result["next_offset"])
        self.assertTrue(result["has_more"])
        self.assertEqual("S2", result["results"][0]["content"])
        self.assertEqual("./notes/a.md", result["results"][0]["execution_file_path"])
        self.assertEqual("/lake/notes/a.md", result["results"][0]["absolute_file_path"])
        self.assertEqual(
            {"unit": "line", "start": 2, "end": 2},
            result["results"][0]["position"],
        )

    def test_get_file_summary_supports_absolute_path_under_datalake(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            datalake_dir = Path(temp_dir) / "data"
            datalake_dir.mkdir()
            absolute_path = datalake_dir / "docs" / "report.pdf"
            absolute_path.parent.mkdir()

            connection = FakeConnection()
            connection.add_row(
                "FROM document_files WHERE relative_path = %s",
                "docs/report.pdf",
                {
                    "source_id": "source_doc",
                    "relative_path": "docs/report.pdf",
                    "filename": "report.pdf",
                    "file_format": "pdf",
                    "file_summary": "Quarterly report summary.",
                    "file_keywords": ["quarterly"],
                    "parse_warnings": [],
                },
            )
            retriever = IndexedDataRetriever(
                connection,
                vector_stores={
                    "tabular": FakeVectorStore([]),
                    "text": FakeVectorStore([]),
                    "document": FakeVectorStore([]),
                    "slideshow": FakeVectorStore([]),
                    "image": FakeVectorStore([]),
                },
                datalake_dir=str(datalake_dir),
            )

            summary = retriever.get_file_summary(str(absolute_path))

            self.assertEqual("document", summary["modality"])
            self.assertEqual("Quarterly report summary.", summary["summary"])
            self.assertEqual("./docs/report.pdf", summary["execution_file_path"])
            self.assertEqual(str(absolute_path), summary["absolute_file_path"])

    def test_get_file_summary_accepts_leading_slash_relative_path(self) -> None:
        connection = FakeConnection()
        connection.add_row(
            "FROM image_files WHERE relative_path = %s",
            "number_image/2-cach-viet-chu-so-5.jpg",
            {
                "source_id": "source_img",
                "relative_path": "number_image/2-cach-viet-chu-so-5.jpg",
                "filename": "2-cach-viet-chu-so-5.jpg",
                "file_format": "jpeg",
                "width": 100,
                "height": 100,
                "color_mode": "RGB",
                "has_alpha": False,
                "is_animated": False,
                "frame_count": 1,
                "file_summary": "An image showing how to write the number 5.",
                "file_keywords": ["number 5"],
                "parse_warnings": [],
            },
        )
        retriever = IndexedDataRetriever(
            connection,
            vector_stores={
                "tabular": FakeVectorStore([]),
                "text": FakeVectorStore([]),
                "document": FakeVectorStore([]),
                "slideshow": FakeVectorStore([]),
                "image": FakeVectorStore([]),
            },
            datalake_dir="/lake",
        )

        summary = retriever.get_file_summary("/number_image/2-cach-viet-chu-so-5.jpg")

        self.assertEqual("image", summary["modality"])
        self.assertEqual(
            "An image showing how to write the number 5.",
            summary["summary"],
        )
        self.assertEqual(
            "./number_image/2-cach-viet-chu-so-5.jpg",
            summary["execution_file_path"],
        )
        self.assertEqual(
            "/lake/number_image/2-cach-viet-chu-so-5.jpg",
            summary["absolute_file_path"],
        )

    def test_build_langchain_tools_exposes_expected_names(self) -> None:
        retriever = IndexedDataRetriever(
            FakeConnection(),
            vector_stores={
                "tabular": FakeVectorStore([]),
                "text": FakeVectorStore([]),
                "document": FakeVectorStore([]),
                "slideshow": FakeVectorStore([]),
                "image": FakeVectorStore([]),
            },
        )

        tools = build_langchain_retrieval_tools(retriever)

        self.assertEqual(
            [
                "search_tabular_data",
                "search_text_data",
                "search_document_data",
                "search_slideshow_data",
                "search_image_data",
                "search_all_indexed_data",
                "get_indexed_file_summary",
            ],
            [tool.name for tool in tools],
        )


if __name__ == "__main__":
    unittest.main()
