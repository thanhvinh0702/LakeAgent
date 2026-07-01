from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from lake_agent.config import LLMSettings
from lake_agent.domain.indexing_models import (
    EnrichedTabularResult,
    TableProfile,
    TabularIndexResult,
)

_SYSTEM_PROMPT = """
You enrich tabular parsing results for a data lake index.

Your job:
- infer concise business meaning for the file and each table
- keep the output grounded in the provided metadata
- do not invent facts that are not supported by the input
- prefer short, retrieval-friendly phrases
- write summaries that are useful for retrieval and dataset understanding
""".strip()


@dataclass(frozen=True, slots=True)
class TabularEnrichmentOptions:
    keyword_limit: int = 8
    preview_row_limit: int = 3


class TabularLLMEnricher:
    def __init__(
        self,
        invoke_enrichment: Callable[[str, str], EnrichedTabularResult],
        options: TabularEnrichmentOptions | None = None,
    ) -> None:
        self._invoke_enrichment = invoke_enrichment
        self._options = options or TabularEnrichmentOptions()

    @classmethod
    def from_env(
        cls,
        options: TabularEnrichmentOptions | None = None,
    ) -> "TabularLLMEnricher":
        settings = LLMSettings.from_env()
        return cls(
            invoke_enrichment=_build_langchain_enrichment_invoker(settings),
            options=options,
        )

    def enrich(self, result: TabularIndexResult) -> TabularIndexResult:
        payload = self._build_payload(result)
        enriched = self._invoke_enrichment(_SYSTEM_PROMPT, _build_user_prompt(payload))
        _apply_enrichment(result, enriched, self._options)
        return result

    def _build_payload(self, result: TabularIndexResult) -> dict[str, Any]:
        tables: list[dict[str, Any]] = []
        for table in result.tables:
            if table.is_context_sheet:
                continue
            table_payload: dict[str, Any] = {
                "table_id": table.table_id,
                "table_name": table.table_name,
                "sheet_name": table.sheet_name,
                "raw_header": table.raw_header,
                "preview_rows": table.preview_rows[: self._options.preview_row_limit],
            }
            if table.sheet_description:
                table_payload["sheet_description"] = table.sheet_description
            if table.header_row_index not in (None, 0) and table.context_before_header:
                table_payload["context_before_header"] = table.context_before_header
            tables.append(table_payload)

        payload = {
            "source_id": result.source_id,
            "relative_path": result.relative_path,
            "filename": result.filename,
            "file_format": result.file_format,
            "parse_warnings": result.parse_warnings,
            "tables": tables,
        }
        if result.workbook_sheet_descriptions:
            payload["workbook_sheet_descriptions"] = result.workbook_sheet_descriptions
        return payload


def _build_langchain_enrichment_invoker(
    settings: LLMSettings,
) -> Callable[[str, str], EnrichedTabularResult]:
    client = ChatOpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedTabularResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_enrichment(
        system_prompt: str,
        user_prompt: str,
    ) -> EnrichedTabularResult:
        response = client.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return _parse_enrichment_response(response, settings)

    return invoke_enrichment


def _build_user_prompt(payload: dict[str, Any]) -> str:
    instructions = {
        "rules": [
            "Preserve every table_id exactly as given.",
            "Prefer canonical names over repeating noisy source headers.",
            "If meaning is unclear, keep the field short and conservative.",
            "Do not invent business context that is not supported by the input.",
            "Summarize only at file level and table level.",
            "Make the file summary specific and retrieval-oriented, not generic.",
            "State what kind of file or workbook this is.",
            "Mention the main subject area or study context only if supported by the input.",
            "Name the major categories of content covered across tables when possible.",
            "Describe what a user can find in this file.",
            "Avoid vague summaries like 'supplementary data file containing ...' unless there is no better evidence.",
            "Prefer 1 to 3 sentences with concrete nouns from the workbook.",
        ],
        "input": payload,
    }
    return json.dumps(instructions, ensure_ascii=True, indent=2)


def _apply_enrichment(
    result: TabularIndexResult,
    enriched: EnrichedTabularResult,
    options: TabularEnrichmentOptions,
) -> None:
    result.file_summary = enriched.file_summary
    result.file_keywords = enriched.file_keywords[: options.keyword_limit]

    table_map = {table.table_id: table for table in result.tables}
    for enriched_table in enriched.tables:
        table = table_map.get(enriched_table.table_id)
        if table is None:
            continue
        table.summary = enriched_table.summary
        table.keywords = enriched_table.keywords[: options.keyword_limit]

    for table in result.tables:
        table.table_search_text = _build_table_search_text(table)
    result.file_search_text = _build_file_search_text(result)


def _build_file_search_text(result: TabularIndexResult) -> str | None:
    parts: list[str] = []
    if result.filename:
        parts.append(result.filename)
    if result.relative_path:
        parts.append(result.relative_path)
    if result.file_summary:
        parts.append(result.file_summary)
    if result.file_keywords:
        parts.append(", ".join(result.file_keywords))
    if result.workbook_sheet_descriptions:
        for sheet_name, description in result.workbook_sheet_descriptions.items():
            line = " | ".join(part for part in [sheet_name, description] if part)
            if line:
                parts.append(line)
    return "\n".join(part for part in parts if part).strip() or None


def _parse_enrichment_response(
    response: Any,
    settings: LLMSettings,
) -> EnrichedTabularResult:
    if isinstance(response, EnrichedTabularResult):
        return response

    if isinstance(response, dict) and "parsed" in response:
        parsed = response.get("parsed")
        if isinstance(parsed, EnrichedTabularResult):
            return parsed
        if parsed is not None:
            return EnrichedTabularResult.model_validate(parsed)
        raise RuntimeError(
            "LLM structured output returned no parsed result. "
            f"model={settings.model_name!r}, base_url={settings.base_url!r}, "
            f"parsing_error={response.get('parsing_error')!r}, "
            f"raw_response={response.get('raw')!r}"
        )

    if response is None:
        raise RuntimeError(
            "LLM returned None for structured output. "
            f"model={settings.model_name!r}, base_url={settings.base_url!r}."
        )

    return EnrichedTabularResult.model_validate(response)


def _build_table_search_text(table: TableProfile) -> str | None:
    parts: list[str] = []
    if table.sheet_name:
        parts.append(table.sheet_name)
    if table.sheet_description:
        parts.append(table.sheet_description)
    for row in table.context_before_header:
        text = " | ".join(value for value in row if value)
        if text:
            parts.append(text)
    if table.raw_header:
        parts.append(" | ".join(value for value in table.raw_header if value))
    if table.summary:
        parts.append(table.summary)
    if table.keywords:
        parts.append(", ".join(table.keywords))
    return "\n".join(part for part in parts if part).strip() or None
