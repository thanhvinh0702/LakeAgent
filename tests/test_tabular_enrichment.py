from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from lake_agent.config import EmbeddingSettings, LLMSettings
from lake_agent.domain.indexing_models import EnrichedTabularResult
from lake_agent.indexing.tabular import (
    DeterministicTabularParser,
    TabularLLMEnricher,
    build_openai_embeddings,
    build_tabular_documents,
)


class TabularLLMEnricherTest(unittest.TestCase):
    def test_enricher_merges_llm_fields_into_parse_result(self) -> None:
        csv_content = (
            "customer_id,amount,order_date\n"
            "101,12.5,2026-07-01\n"
            "102,15.0,2026-07-02\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sales.csv"
            file_path.write_text(csv_content, encoding="utf-8")
            result = DeterministicTabularParser().parse_file(
                file_path,
                relative_path="tables/sales.csv",
            )

        table = result.tables[0]

        def fake_invoke_json(
            system_prompt: str,
            user_prompt: str,
        ) -> EnrichedTabularResult:
            self.assertIn("data lake index", system_prompt)
            self.assertIn("retrieval and dataset understanding", system_prompt)
            prompt_payload = json.loads(user_prompt)
            self.assertEqual("tables/sales.csv", prompt_payload["input"]["relative_path"])
            self.assertEqual(table.table_id, prompt_payload["input"]["tables"][0]["table_id"])
            self.assertNotIn("columns", prompt_payload["input"]["tables"][0])
            self.assertEqual(
                [["101", "12.5", "2026-07-01"], ["102", "15.0", "2026-07-02"]],
                prompt_payload["input"]["tables"][0]["preview_rows"],
            )
            self.assertNotIn("context_before_header", prompt_payload["input"]["tables"][0])
            self.assertNotIn("workbook_context_text", prompt_payload["input"])
            self.assertIn(
                "Make the file summary specific and retrieval-oriented, not generic.",
                prompt_payload["rules"],
            )
            self.assertIn(
                "Describe what a user can find in this file.",
                prompt_payload["rules"],
            )
            return EnrichedTabularResult.model_validate(
                {
                    "file_summary": "Sales transactions with customer and date fields.",
                    "file_keywords": ["sales", "transactions", "customers"],
                    "tables": [
                        {
                            "table_id": table.table_id,
                            "summary": "One table of sales records.",
                            "keywords": ["orders", "revenue"],
                        }
                    ],
                }
            )

        enriched = TabularLLMEnricher(fake_invoke_json).enrich(result)

        self.assertEqual(
            "Sales transactions with customer and date fields.",
            enriched.file_summary,
        )
        self.assertEqual(["sales", "transactions", "customers"], enriched.file_keywords)
        self.assertEqual("One table of sales records.", table.summary)
        self.assertEqual(
            "customer_id | amount | order_date\nOne table of sales records.\norders, revenue",
            table.table_search_text,
        )
        self.assertIn("File summary:", enriched.semantic_text or "")
        self.assertIn("Table summary:", enriched.semantic_text or "")

    def test_llm_settings_reads_expected_env_names(self) -> None:
        previous_values = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "OPENAI_MODEL_NAME": os.getenv("OPENAI_MODEL_NAME"),
        }
        try:
            os.environ["OPENAI_API_KEY"] = "test-key"
            os.environ["OPENAI_BASE_URL"] = "https://example.test/v1"
            os.environ["OPENAI_MODEL_NAME"] = "gpt-test"

            settings = LLMSettings.from_env()
        finally:
            for key, value in previous_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual("test-key", settings.api_key)
        self.assertEqual("https://example.test/v1", settings.base_url)
        self.assertEqual("gpt-test", settings.model_name)

    def test_embedding_settings_reads_expected_env_names(self) -> None:
        previous_values = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
            "OPENAI_EMBEDDING_MODEL_NAME": os.getenv("OPENAI_EMBEDDING_MODEL_NAME"),
        }
        try:
            os.environ["OPENAI_API_KEY"] = "test-key"
            os.environ["OPENAI_BASE_URL"] = "https://example.test/v1"
            os.environ["OPENAI_EMBEDDING_MODEL_NAME"] = "text-embedding-test"

            settings = EmbeddingSettings.from_env()
        finally:
            for key, value in previous_values.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        self.assertEqual("test-key", settings.api_key)
        self.assertEqual("https://example.test/v1", settings.base_url)
        self.assertEqual("text-embedding-test", settings.model_name)

    def test_build_openai_embeddings_uses_embedding_settings(self) -> None:
        embeddings = build_openai_embeddings(
            EmbeddingSettings(
                api_key="test-key",
                base_url="https://example.test/v1",
                model_name="text-embedding-test",
            )
        )

        self.assertEqual("text-embedding-test", embeddings.model)

    def test_enricher_raises_clear_error_when_structured_output_is_missing(self) -> None:
        csv_content = "customer_id,amount\n101,12.5\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sales.csv"
            file_path.write_text(csv_content, encoding="utf-8")
            result = DeterministicTabularParser().parse_file(file_path)

        def missing_response(_: str, __: str) -> EnrichedTabularResult:
            raise RuntimeError(
                "LLM returned None for structured output. "
                "model='test-model', base_url='https://example.test/v1'."
            )

        with self.assertRaises(RuntimeError) as context:
            TabularLLMEnricher(missing_response).enrich(result)

        self.assertIn("structured output", str(context.exception))

    def test_enricher_includes_context_before_header_when_header_is_skipped(self) -> None:
        csv_content = (
            "JAMES LOGAN,,,,,,,,\n"
            "Wins,Name,Code,School,,,STANDARD TEAM NAME,NUM ROUNDS,WEIGHTING\n"
            "5,Gatlin & Ramarao,Archbishop Mitty GR,Archbishop Mitty,,,Mitty GR,5,0.8\n"
            "5,Lahiri & Ponnuswamy,Archbishop Mitty LP,Archbishop Mitty,,,Mitty PL,,\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "tournament.csv"
            file_path.write_text(csv_content, encoding="utf-8")
            result = DeterministicTabularParser().parse_file(file_path)

        table = result.tables[0]

        def fake_invoke_json(
            _: str,
            user_prompt: str,
        ) -> EnrichedTabularResult:
            prompt_payload = json.loads(user_prompt)
            table_payload = prompt_payload["input"]["tables"][0]
            self.assertEqual(
                [["JAMES LOGAN", "", "", "", "", "", "", "", ""]],
                table_payload["context_before_header"],
            )
            self.assertEqual(
                [
                    [
                        "5",
                        "Gatlin & Ramarao",
                        "Archbishop Mitty GR",
                        "Archbishop Mitty",
                        "",
                        "",
                        "Mitty GR",
                        "5",
                        "0.8",
                    ],
                    [
                        "5",
                        "Lahiri & Ponnuswamy",
                        "Archbishop Mitty LP",
                        "Archbishop Mitty",
                        "",
                        "",
                        "Mitty PL",
                        "",
                        "",
                    ],
                ],
                table_payload["preview_rows"],
            )
            return EnrichedTabularResult.model_validate({"tables": []})

        TabularLLMEnricher(fake_invoke_json).enrich(result)

    def test_enricher_uses_sheet_descriptions_and_skips_context_sheet_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "supplement.xlsx"
            _write_multi_sheet_xlsx(
                file_path,
                sheets=[
                    (
                        "Index",
                        [
                            ["Sheet", "Description"],
                            ["A-MSI", "MSI-H determination"],
                            [
                                "B-SE-proteomics",
                                "Significant genes by global proteomics in Figure 1",
                            ],
                        ],
                    ),
                    (
                        "A-MSI",
                        [
                            ["Sample", "Status"],
                            ["S1", "MSI-H"],
                        ],
                    ),
                    (
                        "B-SE-proteomics",
                        [
                            ["Gene", "FoldChange"],
                            ["TP53", "2.1"],
                        ],
                    ),
                ],
            )
            result = DeterministicTabularParser().parse_file(file_path)

        def fake_invoke_json(
            _: str,
            user_prompt: str,
        ) -> EnrichedTabularResult:
            prompt_payload = json.loads(user_prompt)
            self.assertNotIn("workbook_context_text", prompt_payload["input"])
            self.assertIn("workbook_sheet_descriptions", prompt_payload["input"])
            table_names = [table["table_name"] for table in prompt_payload["input"]["tables"]]
            self.assertEqual(["A-MSI", "B-SE-proteomics"], table_names)
            self.assertEqual(
                "MSI-H determination",
                prompt_payload["input"]["tables"][0]["sheet_description"],
            )
            self.assertEqual(
                "Significant genes by global proteomics in Figure 1",
                prompt_payload["input"]["tables"][1]["sheet_description"],
            )
            return EnrichedTabularResult.model_validate({"tables": []})

        enriched = TabularLLMEnricher(fake_invoke_json).enrich(result)
        self.assertIn("A-MSI", enriched.tables[1].table_search_text or "")
        self.assertIn("MSI-H determination", enriched.tables[1].table_search_text or "")

    def test_build_tabular_documents_uses_table_search_text(self) -> None:
        csv_content = (
            "customer_id,amount,order_date\n"
            "101,12.5,2026-07-01\n"
            "102,15.0,2026-07-02\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sales.csv"
            file_path.write_text(csv_content, encoding="utf-8")
            result = DeterministicTabularParser().parse_file(
                file_path,
                relative_path="tables/sales.csv",
            )

        result.file_summary = "Sales transactions for customer orders."
        result.file_keywords = ["sales", "orders"]
        result.tables[0].summary = "One table of sales records."
        result.tables[0].keywords = ["revenue", "transactions"]
        result.tables[0].table_search_text = (
            "customer_id | amount | order_date\n"
            "One table of sales records.\n"
            "revenue, transactions"
        )

        documents = build_tabular_documents(result)

        self.assertEqual(1, len(documents))
        self.assertEqual(result.tables[0].table_id, documents[0].metadata["table_id"])
        self.assertEqual("tables/sales.csv", documents[0].metadata["relative_path"])
        self.assertIn("Sales transactions for customer orders.", documents[0].page_content)
        self.assertIn("customer_id | amount | order_date", documents[0].page_content)


def _write_multi_sheet_xlsx(
    path: Path,
    *,
    sheets: list[tuple[str, list[list[str]]]],
) -> None:
    shared_values: list[str] = []
    shared_index: dict[str, int] = {}

    def add_shared(value: str) -> int:
        if value not in shared_index:
            shared_index[value] = len(shared_values)
            shared_values.append(value)
        return shared_index[value]

    workbook_sheets_xml: list[str] = []
    workbook_rels_xml: list[str] = []
    sheet_xml_by_index: dict[int, str] = {}

    for sheet_index, (sheet_name, rows) in enumerate(sheets, start=1):
        sheet_rows: list[str] = []
        for row_index, row in enumerate(rows, start=1):
            cells: list[str] = []
            for col_index, value in enumerate(row, start=1):
                if value == "":
                    continue
                ref = f"{_excel_column(col_index)}{row_index}"
                shared_id = add_shared(value)
                cells.append(f'<c r="{ref}" t="s"><v>{shared_id}</v></c>')
            sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

        workbook_sheets_xml.append(
            f'<sheet name="{sheet_name}" sheetId="{sheet_index}" r:id="rId{sheet_index}"/>'
        )
        workbook_rels_xml.append(
            '<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet{idx}.xml"/>'.format(idx=sheet_index)
        )
        sheet_xml_by_index[sheet_index] = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<sheetData>{''.join(sheet_rows)}</sheetData>"
            "</worksheet>"
        )

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(workbook_sheets_xml)}</sheets>"
        "</workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(workbook_rels_xml)
        + "</Relationships>"
    )
    shared_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        f'count="{len(shared_values)}" uniqueCount="{len(shared_values)}">'
        + "".join(f"<si><t>{escape(value)}</t></si>" for value in shared_values)
        + "</sst>"
    )

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        for sheet_index, sheet_xml in sheet_xml_by_index.items():
            archive.writestr(f"xl/worksheets/sheet{sheet_index}.xml", sheet_xml)
        archive.writestr("xl/sharedStrings.xml", shared_xml)


def _excel_column(index: int) -> str:
    letters: list[str] = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))
