from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from lake_agent.indexing.tabular import DeterministicTabularParser


class DeterministicTabularParserTest(unittest.TestCase):
    def test_csv_parser_builds_table_and_column_profiles(self) -> None:
        csv_content = (
            "customer_id,amount,active,order_date\n"
            "101,12.5,true,2026-07-01\n"
            "102,15.0,false,2026-07-02\n"
            "103,12.5,true,2026-07-03\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sales.csv"
            file_path.write_text(csv_content, encoding="utf-8")

            result = DeterministicTabularParser().parse_file(
                file_path,
                relative_path="tables/sales.csv",
            )

        self.assertEqual("csv", result.file_format)
        self.assertEqual("tables/sales.csv", result.relative_path)
        self.assertEqual(1, len(result.tables))

        table = result.tables[0]
        self.assertEqual("sales", table.table_name)
        self.assertEqual(3, table.row_count)
        self.assertEqual(4, table.column_count)
        self.assertEqual(
            ["customer_id", "amount", "active", "order_date"],
            [column.name for column in table.columns],
        )
        self.assertEqual("integer", table.columns[0].inferred_type)
        self.assertEqual("float", table.columns[1].inferred_type)
        self.assertEqual("boolean", table.columns[2].inferred_type)
        self.assertEqual("date", table.columns[3].inferred_type)
        self.assertEqual(
            ["customer_id", "amount", "active", "order_date"],
            table.raw_header,
        )
        self.assertIn("sales", result.lexical_text or "")

    def test_tsv_parser_generates_column_names_when_header_is_missing(self) -> None:
        tsv_content = "1\talice\t10\n2\tbob\t20\n3\tcarol\t30\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "users.tsv"
            file_path.write_text(tsv_content, encoding="utf-8")

            result = DeterministicTabularParser().parse_file(file_path)

        table = result.tables[0]
        self.assertIsNone(table.header_row_index)
        self.assertEqual(
            ["column_1", "column_2", "column_3"],
            [column.name for column in table.columns],
        )
        self.assertTrue(table.warnings)
        self.assertEqual(3, table.row_count)

    def test_csv_parser_skips_title_row_and_finds_real_header(self) -> None:
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
        self.assertEqual(1, table.header_row_index)
        self.assertEqual(
            [
                "Wins",
                "Name",
                "Code",
                "School",
                "column_5",
                "column_6",
                "STANDARD TEAM NAME",
                "NUM ROUNDS",
                "WEIGHTING",
            ],
            [column.name for column in table.columns],
        )
        self.assertEqual(2, table.row_count)
        self.assertEqual("integer", table.columns[0].inferred_type)
        self.assertEqual(
            [
                "Wins",
                "Name",
                "Code",
                "School",
                "",
                "",
                "STANDARD TEAM NAME",
                "NUM ROUNDS",
                "WEIGHTING",
            ],
            table.raw_header,
        )
        self.assertIn("Skipped 1 leading row", table.warnings[0])

    def test_xlsx_parser_reads_sheet_and_raw_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "workbook.xlsx"
            _write_minimal_xlsx(
                file_path,
                sheet_name="Teams",
                rows=[
                    ["TOURNAMENT", "", ""],
                    ["Wins", "Name", "School"],
                    ["5", "Gatlin & Ramarao", "Archbishop Mitty"],
                    ["4", "Menotti & Bhasin", "James Logan"],
                ],
            )

            result = DeterministicTabularParser().parse_file(
                file_path,
                relative_path="tables/workbook.xlsx",
            )

        self.assertEqual("xlsx", result.file_format)
        self.assertEqual(1, len(result.tables))
        table = result.tables[0]
        self.assertEqual("Teams", table.table_name)
        self.assertEqual("Teams", table.sheet_name)
        self.assertEqual(1, table.header_row_index)
        self.assertEqual(["Wins", "Name", "School"], table.raw_header)
        self.assertEqual(2, table.row_count)
        self.assertEqual("integer", table.columns[0].inferred_type)
        self.assertEqual(["Wins", "Name", "School"], [c.name for c in table.columns])

    def test_xlsx_parser_keeps_large_numeric_values_as_raw_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "numbers.xlsx"
            _write_minimal_xlsx(
                file_path,
                sheet_name="Data",
                rows=[
                    ["Id", "Value"],
                    ["1", "999999999999"],
                    ["2", "123456789012345"],
                ],
                numeric_cells={(2, 1), (2, 2), (3, 1), (3, 2)},
            )

            result = DeterministicTabularParser().parse_file(file_path)

        table = result.tables[0]
        self.assertEqual("999999999999", table.preview_rows[0][1])
        self.assertEqual("123456789012345", table.preview_rows[1][1])

    def test_xlsx_parser_prefers_header_at_top_for_text_index_sheet(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "index_sheet.xlsx"
            _write_minimal_xlsx(
                file_path,
                sheet_name="Index",
                rows=[
                    ["Sheet", "Description"],
                    ["A-MSI", "MSI-H determination"],
                    ["B-SE-proteomics", "Significant genes by global proteomics"],
                ],
            )

            result = DeterministicTabularParser().parse_file(file_path)

        table = result.tables[0]
        self.assertEqual(0, table.header_row_index)
        self.assertEqual(["Sheet", "Description"], table.raw_header)
        self.assertEqual(["Sheet", "Description"], [c.name for c in table.columns])

    def test_xlsx_parser_skips_sparse_unit_row_before_real_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "scientific.xlsx"
            _write_minimal_xlsx(
                file_path,
                sheet_name="Supplement",
                rows=[
                    ["Data Supplement to paper", "", "", "", ""],
                    ["", "", "(ppm)", "(ppm)", "(ppm)"],
                    ["Site", "Hole", "Al", "Si", "Fe"],
                    ["O123", "A", "10", "20", "30"],
                ],
            )

            result = DeterministicTabularParser().parse_file(file_path)

        table = result.tables[0]
        self.assertEqual(2, table.header_row_index)
        self.assertEqual(["Site", "Hole", "Al", "Si", "Fe"], table.raw_header)
        self.assertEqual(["Site", "Hole", "Al", "Si", "Fe"], [c.name for c in table.columns])


def _write_minimal_xlsx(
    path: Path,
    *,
    sheet_name: str,
    rows: list[list[str]],
    numeric_cells: set[tuple[int, int]] | None = None,
) -> None:
    shared_values: list[str] = []
    shared_index: dict[str, int] = {}
    numeric_cells = numeric_cells or set()

    def add_shared(value: str) -> int:
        if value not in shared_index:
            shared_index[value] = len(shared_values)
            shared_values.append(value)
        return shared_index[value]

    sheet_rows: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells: list[str] = []
        for col_index, value in enumerate(row, start=1):
            if value == "":
                continue
            ref = f"{_excel_column(col_index)}{row_index}"
            if (row_index, col_index) in numeric_cells:
                cells.append(f'<c r="{ref}"><v>{escape(value)}</v></c>')
            else:
                shared_id = add_shared(value)
                cells.append(f'<c r="{ref}" t="s"><v>{shared_id}</v></c>')
        sheet_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        "<sheets>"
        f'<sheet name="{sheet_name}" sheetId="1" '
        'r:id="rId1"/>'
        "</sheets>"
        "</workbook>"
    )
    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/>'
        "</Relationships>"
    )
    sheet_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        "</worksheet>"
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
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        archive.writestr("xl/sharedStrings.xml", shared_xml)


def _excel_column(index: int) -> str:
    result = ""
    current = index
    while current > 0:
        current, remainder = divmod(current - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


if __name__ == "__main__":
    unittest.main()
