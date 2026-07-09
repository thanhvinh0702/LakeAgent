from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from lake_agent.config import LLMSettings
from lake_agent.domain.indexing_models import (
    EnrichedEpubResult,
    EpubIndexResult,
)

_SYSTEM_PROMPT = """
You enrich parsed EPUB/light novel files for a data lake index.

Your job:
- infer concise whole-book meaning from sampled chapter chunks and image captions
- keep the output grounded in the provided content
- do not invent plot, character identities, dates, or facts not supported by the input
- prefer short, retrieval-friendly summaries and keywords
""".strip()


@dataclass(frozen=True, slots=True)
class EpubEnrichmentOptions:
    keyword_limit: int = 8
    section_character_limit: int = 1200
    section_count_limit: int = 12
    image_summary_count_limit: int = 6


class EpubLLMEnricher:
    def __init__(
        self,
        invoke_enrichment: Callable[[str, str], EnrichedEpubResult],
        invoke_batch_enrichment: Callable[[list[tuple[str, str]]], list[EnrichedEpubResult]] | None = None,
        options: EpubEnrichmentOptions | None = None,
    ) -> None:
        self._invoke_enrichment = invoke_enrichment
        self._invoke_batch_enrichment = invoke_batch_enrichment
        self._options = options or EpubEnrichmentOptions()

    @classmethod
    def from_env(
        cls,
        options: EpubEnrichmentOptions | None = None,
    ) -> "EpubLLMEnricher":
        settings = LLMSettings.from_env()
        return cls(
            invoke_enrichment=_build_langchain_enrichment_invoker(settings),
            invoke_batch_enrichment=_build_langchain_batch_enrichment_invoker(settings),
            options=options,
        )

    def enrich(self, result: EpubIndexResult) -> EpubIndexResult:
        payload = self._build_payload(result)
        enriched = self._invoke_enrichment(_SYSTEM_PROMPT, _build_user_prompt(payload))
        _apply_enrichment(result, enriched, self._options)
        return result

    def enrich_batch(self, results: list[EpubIndexResult]) -> list[EpubIndexResult]:
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

    def _build_payload(self, result: EpubIndexResult) -> dict[str, Any]:
        text_sections = [
            section for section in result.sections if section.section_type == "chapter_text"
        ]
        image_sections = [
            section for section in result.sections if section.section_type == "image_summary"
        ]

        sampled_sections = _sample_evenly(text_sections, self._options.section_count_limit)
        sampled_images = _sample_evenly(
            image_sections,
            self._options.image_summary_count_limit,
        )
        return {
            "source_id": result.source_id,
            "relative_path": result.relative_path,
            "filename": result.filename,
            "file_format": result.file_format,
            "title": result.title,
            "creators": result.creators,
            "language": result.language,
            "publisher": result.publisher,
            "identifier": result.identifier,
            "chapter_count": result.chapter_count,
            "image_count": result.image_count,
            "parse_warnings": result.parse_warnings,
            "sections": [
                {
                    "chunk_index": section.chunk_index,
                    "chapter_index": section.chapter_index,
                    "chapter_title": section.chapter_title,
                    "heading": section.heading,
                    "content": section.content[: self._options.section_character_limit],
                }
                for section in sampled_sections
            ],
            "embedded_image_summaries": [
                {
                    "image_index": section.image_index,
                    "heading": section.heading,
                    "summary": section.content[: self._options.section_character_limit],
                }
                for section in sampled_images
            ],
        }


def _build_langchain_enrichment_invoker(
    settings: LLMSettings,
) -> Callable[[str, str], EnrichedEpubResult]:
    client = init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedEpubResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_enrichment(system_prompt: str, user_prompt: str) -> EnrichedEpubResult:
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
) -> Callable[[list[tuple[str, str]]], list[EnrichedEpubResult]]:
    client = init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedEpubResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_batch_enrichment(
        prompt_pairs: list[tuple[str, str]],
    ) -> list[EnrichedEpubResult]:
        responses = client.batch(
            [
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
                for system_prompt, user_prompt in prompt_pairs
            ]
        )
        return [_parse_enrichment_response(response, settings) for response in responses]

    return invoke_batch_enrichment


def _build_user_prompt(payload: dict[str, Any]) -> str:
    instructions = {
        "rules": [
            "Keep summaries grounded in the sampled EPUB text and image captions.",
            "Do not invent missing plot points, character relationships, dates, or citations.",
            "Prefer concise, retrieval-oriented wording.",
            "File summary should describe what the EPUB contains and what a user can find in it.",
            "Keywords should be concrete nouns, character names, places, themes, or short phrases.",
        ],
        "input": payload,
    }
    return json.dumps(instructions, ensure_ascii=True, indent=2)


def _apply_enrichment(
    result: EpubIndexResult,
    enriched: EnrichedEpubResult,
    options: EpubEnrichmentOptions,
) -> None:
    result.file_summary = enriched.file_summary
    result.file_keywords = enriched.file_keywords[: options.keyword_limit]
    result.file_search_text = _build_file_search_text(result)


def _parse_enrichment_response(
    response: Any,
    settings: LLMSettings,
) -> EnrichedEpubResult:
    if isinstance(response, EnrichedEpubResult):
        return response
    if isinstance(response, dict) and "parsed" in response:
        parsed = response.get("parsed")
        if isinstance(parsed, EnrichedEpubResult):
            return parsed
        if parsed is not None:
            return EnrichedEpubResult.model_validate(parsed)
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
    return EnrichedEpubResult.model_validate(response)


def _build_file_search_text(result: EpubIndexResult) -> str | None:
    parts: list[str] = []
    if result.title:
        parts.append(result.title)
    if result.filename:
        parts.append(result.filename)
    if result.relative_path:
        parts.append(result.relative_path)
    if result.creators:
        parts.append(", ".join(result.creators))
    if result.file_summary:
        parts.append(result.file_summary)
    if result.file_keywords:
        parts.append(", ".join(result.file_keywords))
    for section in result.sections[:3]:
        if section.heading:
            parts.append(section.heading)
    return "\n".join(part for part in parts if part).strip() or None


def _sample_evenly(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    max_index = len(items) - 1
    sampled_indices = [
        (position * max_index) // (limit - 1)
        for position in range(limit)
    ]
    return [items[index] for index in sampled_indices]
