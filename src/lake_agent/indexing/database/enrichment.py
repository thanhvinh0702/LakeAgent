from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

from lake_agent.config import LLMSettings
from lake_agent.domain.indexing_models import (
    EnrichedDatabaseResult,
    DbTableProfile,
    DatabaseIndexResult,
)


_SYSTEM_PROMPT = """
You enrich database parsing results for a data lake index.

Your job:
- infer concise semantic meaning for the database file and each of its tables
- keep the output grounded in the provided metadata (table names, column types, row counts, preview data)
- do not invent facts that are not supported by the input
- prefer short, retrieval-friendly phrases
- write summaries that are useful for retrieval and database understanding
""".strip()


@dataclass(frozen=True, slots=True)
class DatabaseEnrichmentOptions:
    keyword_limit: int = 8
    preview_row_limit: int = 3


class DatabaseLLMEnricher:
    def __init__(
        self,
        invoke_enrichment: Callable[[str, str], EnrichedDatabaseResult],
        invoke_batch_enrichment: Callable[[list[tuple[str, str]]], list[EnrichedDatabaseResult]] | None = None,
        options: DatabaseEnrichmentOptions | None = None,
    ) -> None:
        self._invoke_enrichment = invoke_enrichment
        self._invoke_batch_enrichment = invoke_batch_enrichment
        self._options = options or DatabaseEnrichmentOptions()

    @classmethod
    def from_env(
        cls,
        options: DatabaseEnrichmentOptions | None = None,
    ) -> "DatabaseLLMEnricher":
        settings = LLMSettings.from_env()
        return cls(
            invoke_enrichment=_build_langchain_enrichment_invoker(settings),
            invoke_batch_enrichment=_build_langchain_batch_enrichment_invoker(settings),
            options=options,
        )

    def enrich(self, result: DatabaseIndexResult) -> DatabaseIndexResult:
        payload = self._build_payload(result)
        enriched = self._invoke_enrichment(_SYSTEM_PROMPT, _build_user_prompt(payload))
        _apply_enrichment(result, enriched, self._options)
        return result

    def enrich_batch(self, results: list[DatabaseIndexResult]) -> list[DatabaseIndexResult]:
        if not results:
            return []
        if self._invoke_batch_enrichment is None or len(results) == 1:
            return [self.enrich(result) for result in results]

        prompt_pairs = [
            (_SYSTEM_PROMPT, _build_user_prompt(self._build_payload(result)))
            for result in results
        ]
        enriched_results = self._invoke_batch_enrichment(prompt_pairs)
        if len(enriched_results) != len(results):
            raise RuntimeError(
                "LLM batch enrichment returned a different number of results than inputs. "
                f"expected={len(results)}, actual={len(enriched_results)}"
            )
        for result, enriched in zip(results, enriched_results, strict=True):
            _apply_enrichment(result, enriched, self._options)
        return results

    def _build_payload(self, result: DatabaseIndexResult) -> dict[str, Any]:
        tables: list[dict[str, Any]] = []
        for table in result.tables:
            columns_payload = [
                {"name": col.name, "type": col.inferred_type, "nullable": col.nullable}
                for col in table.columns
            ]
            table_payload: dict[str, Any] = {
                "table_id": table.table_id,
                "table_name": table.table_name,
                "row_count": table.row_count,
                "column_count": table.column_count,
                "columns": columns_payload,
                "preview_rows": table.preview_rows[: self._options.preview_row_limit],
            }
            tables.append(table_payload)

        return {
            "source_id": result.source_id,
            "relative_path": result.relative_path,
            "filename": result.filename,
            "file_format": result.file_format,
            "parse_warnings": result.parse_warnings,
            "tables": tables,
        }


def _build_langchain_enrichment_invoker(
    settings: LLMSettings,
) -> Callable[[str, str], EnrichedDatabaseResult]:
    client = init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedDatabaseResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_enrichment(
        system_prompt: str,
        user_prompt: str,
    ) -> EnrichedDatabaseResult:
        response = client.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return _parse_enrichment_response(response, settings)

    return invoke_enrichment


def _build_langchain_batch_enrichment_invoker(
    settings: LLMSettings,
) -> Callable[[list[tuple[str, str]]], list[EnrichedDatabaseResult]]:
    client = init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedDatabaseResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_batch_enrichment(
        prompt_pairs: list[tuple[str, str]],
    ) -> list[EnrichedDatabaseResult]:
        responses = client.batch(
            [
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
                for system_prompt, user_prompt in prompt_pairs
            ]
        )
        return [
            _parse_enrichment_response(response, settings)
            for response in responses
        ]

    return invoke_batch_enrichment


def _build_user_prompt(payload: dict[str, Any]) -> str:
    instructions = {
        "rules": [
            "Preserve every table_id exactly as given.",
            "Prefer concise canonical table summaries over generic names.",
            "If meaning is unclear, keep the field short and conservative.",
            "Do not invent business context that is not supported by the input.",
            "Summarize only at file level and table level.",
            "Make the file summary specific and retrieval-oriented, not generic.",
            "State what kind of database this is.",
            "Name the major categories of database tables/schemas covered when possible.",
            "Describe what a user can find in this database.",
        ],
        "input": payload,
    }
    return json.dumps(instructions, ensure_ascii=True, indent=2)


def _apply_enrichment(
    result: DatabaseIndexResult,
    enriched: EnrichedDatabaseResult,
    options: DatabaseEnrichmentOptions,
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


def _build_file_search_text(result: DatabaseIndexResult) -> str | None:
    parts: list[str] = []
    if result.filename:
        parts.append(result.filename)
    if result.relative_path:
        parts.append(result.relative_path)
    if result.file_summary:
        parts.append(result.file_summary)
    if result.file_keywords:
        parts.append(", ".join(result.file_keywords))
    for table in result.tables[:3]:
        parts.append(f"Table: {table.table_name}")
        if table.summary:
            parts.append(table.summary)
    return "\n".join(part for part in parts if part).strip() or None


def _parse_enrichment_response(
    response: Any,
    settings: LLMSettings,
) -> EnrichedDatabaseResult:
    if isinstance(response, EnrichedDatabaseResult):
        return response

    if isinstance(response, dict) and "parsed" in response:
        parsed = response.get("parsed")
        if isinstance(parsed, EnrichedDatabaseResult):
            return parsed
        if parsed is not None:
            return EnrichedDatabaseResult.model_validate(parsed)
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

    return EnrichedDatabaseResult.model_validate(response)


def _build_table_search_text(table: DbTableProfile) -> str | None:
    parts: list[str] = []
    parts.append(f"Table: {table.table_name}")
    parts.append(f"Row count: {table.row_count or 0}")
    
    col_names = [col.name for col in table.columns]
    if col_names:
        parts.append("Columns: " + ", ".join(col_names))
        
    if table.summary:
        parts.append(table.summary)
    if table.keywords:
        parts.append(", ".join(table.keywords))
        
    preview_lines = []
    if col_names:
        preview_lines.append("| " + " | ".join(col_names) + " |")
        preview_lines.append("| " + " | ".join(["---"] * len(col_names)) + " |")
    for row in table.preview_rows:
        preview_lines.append("| " + " | ".join(row) + " |")
        
    if preview_lines:
        parts.append("Preview:\n" + "\n".join(preview_lines))
        
    return "\n".join(part for part in parts if part).strip() or None
