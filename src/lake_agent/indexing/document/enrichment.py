from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from lake_agent.config import LLMSettings
from lake_agent.domain.indexing_models import (
    DocumentIndexResult,
    EnrichedDocumentResult,
)

_SYSTEM_PROMPT = """
You enrich parsed document files for a data lake index.

Your job:
- infer concise whole-document meaning from the parsed chunks
- keep the output grounded in the provided content
- do not invent facts or context not supported by the input
- prefer short, retrieval-friendly summaries and keywords
""".strip()


@dataclass(frozen=True, slots=True)
class DocumentEnrichmentOptions:
    keyword_limit: int = 8
    section_character_limit: int = 1200
    section_count_limit: int = 12


class DocumentLLMEnricher:
    def __init__(
        self,
        invoke_enrichment: Callable[[str, str], EnrichedDocumentResult],
        invoke_batch_enrichment: Callable[[list[tuple[str, str]]], list[EnrichedDocumentResult]] | None = None,
        options: DocumentEnrichmentOptions | None = None,
    ) -> None:
        self._invoke_enrichment = invoke_enrichment
        self._invoke_batch_enrichment = invoke_batch_enrichment
        self._options = options or DocumentEnrichmentOptions()

    @classmethod
    def from_env(
        cls,
        options: DocumentEnrichmentOptions | None = None,
    ) -> "DocumentLLMEnricher":
        settings = LLMSettings.from_env()
        return cls(
            invoke_enrichment=_build_langchain_enrichment_invoker(settings),
            invoke_batch_enrichment=_build_langchain_batch_enrichment_invoker(settings),
            options=options,
        )

    def enrich(self, result: DocumentIndexResult) -> DocumentIndexResult:
        payload = self._build_payload(result)
        enriched = self._invoke_enrichment(_SYSTEM_PROMPT, _build_user_prompt(payload))
        _apply_enrichment(result, enriched, self._options)
        return result

    def enrich_batch(self, results: list[DocumentIndexResult]) -> list[DocumentIndexResult]:
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

    def _build_payload(self, result: DocumentIndexResult) -> dict[str, Any]:
        sections: list[dict[str, Any]] = []
        for section in result.sections[: self._options.section_count_limit]:
            sections.append(
                {
                    "chunk_index": section.chunk_index,
                    "heading": section.heading,
                    "page_start": section.page_start,
                    "page_end": section.page_end,
                    "content": section.content[: self._options.section_character_limit],
                }
            )

        return {
            "source_id": result.source_id,
            "relative_path": result.relative_path,
            "filename": result.filename,
            "file_format": result.file_format,
            "parse_warnings": result.parse_warnings,
            "sections": sections,
        }


def _build_langchain_enrichment_invoker(
    settings: LLMSettings,
) -> Callable[[str, str], EnrichedDocumentResult]:
    client = init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedDocumentResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_enrichment(system_prompt: str, user_prompt: str) -> EnrichedDocumentResult:
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
) -> Callable[[list[tuple[str, str]]], list[EnrichedDocumentResult]]:
    client = init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedDocumentResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_batch_enrichment(
        prompt_pairs: list[tuple[str, str]],
    ) -> list[EnrichedDocumentResult]:
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
            "Keep summaries grounded in the provided document chunks.",
            "Do not invent missing context, dates, entities, or citations.",
            "Prefer concise, retrieval-oriented wording.",
            "File summary should describe what the document contains and what a user can find in it.",
            "Keywords should be concrete nouns or short phrases, not full sentences.",
        ],
        "input": payload,
    }
    return json.dumps(instructions, ensure_ascii=True, indent=2)


def _apply_enrichment(
    result: DocumentIndexResult,
    enriched: EnrichedDocumentResult,
    options: DocumentEnrichmentOptions,
) -> None:
    result.file_summary = enriched.file_summary
    result.file_keywords = enriched.file_keywords[: options.keyword_limit]
    result.file_search_text = _build_file_search_text(result)


def _parse_enrichment_response(
    response: Any,
    settings: LLMSettings,
) -> EnrichedDocumentResult:
    if isinstance(response, EnrichedDocumentResult):
        return response

    if isinstance(response, dict) and "parsed" in response:
        parsed = response.get("parsed")
        if isinstance(parsed, EnrichedDocumentResult):
            return parsed
        if parsed is not None:
            return EnrichedDocumentResult.model_validate(parsed)
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

    return EnrichedDocumentResult.model_validate(response)


def _build_file_search_text(result: DocumentIndexResult) -> str | None:
    parts: list[str] = []
    if result.filename:
        parts.append(result.filename)
    if result.relative_path:
        parts.append(result.relative_path)
    if result.file_summary:
        parts.append(result.file_summary)
    if result.file_keywords:
        parts.append(", ".join(result.file_keywords))
    for section in result.sections[:3]:
        if section.heading:
            parts.append(section.heading)
    return "\n".join(part for part in parts if part).strip() or None
