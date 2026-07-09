from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from lake_agent.config import LLMSettings
from lake_agent.domain.indexing_models import AudioIndexResult, EnrichedAudioResult

_SYSTEM_PROMPT = """
You enrich parsed audio transcripts for a data lake index.

Your job:
- infer concise file-level meaning from the provided transcript
- keep the output grounded in the provided transcript and metadata
- do not invent facts or context not supported by the input
- prefer short, retrieval-friendly summaries and keywords
""".strip()


@dataclass(frozen=True, slots=True)
class AudioEnrichmentOptions:
    keyword_limit: int = 8
    transcript_character_limit: int = 2400
    section_character_limit: int = 1200
    section_count_limit: int = 12


class AudioLLMEnricher:
    def __init__(
        self,
        invoke_enrichment: Callable[[str, str], EnrichedAudioResult],
        invoke_batch_enrichment: Callable[[list[tuple[str, str]]], list[EnrichedAudioResult]] | None = None,
        options: AudioEnrichmentOptions | None = None,
    ) -> None:
        self._invoke_enrichment = invoke_enrichment
        self._invoke_batch_enrichment = invoke_batch_enrichment
        self._options = options or AudioEnrichmentOptions()

    @classmethod
    def from_env(
        cls,
        options: AudioEnrichmentOptions | None = None,
    ) -> "AudioLLMEnricher":
        settings = LLMSettings.from_env()
        return cls(
            invoke_enrichment=_build_langchain_enrichment_invoker(settings),
            invoke_batch_enrichment=_build_langchain_batch_enrichment_invoker(settings),
            options=options,
        )

    def enrich(self, result: AudioIndexResult) -> AudioIndexResult:
        payload = self._build_payload(result)
        enriched = self._invoke_enrichment(_SYSTEM_PROMPT, _build_user_prompt(payload))
        _apply_enrichment(result, enriched, self._options)
        return result

    def enrich_batch(self, results: list[AudioIndexResult]) -> list[AudioIndexResult]:
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

    def _build_payload(self, result: AudioIndexResult) -> dict[str, Any]:
        sections = []
        sampled_sections = _sample_evenly(result.sections, self._options.section_count_limit)
        for section in sampled_sections:
            sections.append(
                {
                    "chunk_index": section.chunk_index,
                    "start_seconds": section.start_seconds,
                    "end_seconds": section.end_seconds,
                    "content": section.content[: self._options.section_character_limit],
                }
            )
        transcript_excerpt = None
        if result.transcript_text:
            transcript_excerpt = result.transcript_text[: self._options.transcript_character_limit]
        return {
            "source_id": result.source_id,
            "relative_path": result.relative_path,
            "filename": result.filename,
            "file_format": result.file_format,
            "duration_seconds": result.duration_seconds,
            "transcript_language": result.transcript_language,
            "codec_name": result.codec_name,
            "sample_rate": result.sample_rate,
            "channels": result.channels,
            "parse_warnings": result.parse_warnings,
            "transcript_excerpt": transcript_excerpt,
            "sections": sections,
        }


def _build_langchain_enrichment_invoker(
    settings: LLMSettings,
) -> Callable[[str, str], EnrichedAudioResult]:
    client = init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedAudioResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_enrichment(system_prompt: str, user_prompt: str) -> EnrichedAudioResult:
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
) -> Callable[[list[tuple[str, str]]], list[EnrichedAudioResult]]:
    client = init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedAudioResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_batch_enrichment(
        prompt_pairs: list[tuple[str, str]],
    ) -> list[EnrichedAudioResult]:
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
            "Keep summaries grounded in the provided transcript.",
            "Do not invent missing context, dates, or entities.",
            "Prefer concise, retrieval-oriented wording.",
            "File summary should describe what the audio contains and what a user can find in it.",
            "Keywords should be concrete nouns or short phrases, not full sentences.",
        ],
        "input": payload,
    }
    return json.dumps(instructions, ensure_ascii=True, indent=2)


def _apply_enrichment(
    result: AudioIndexResult,
    enriched: EnrichedAudioResult,
    options: AudioEnrichmentOptions,
) -> None:
    result.file_summary = enriched.file_summary
    result.file_keywords = enriched.file_keywords[: options.keyword_limit]
    result.file_search_text = _build_file_search_text(result)


def _parse_enrichment_response(
    response: Any,
    settings: LLMSettings,
) -> EnrichedAudioResult:
    if isinstance(response, EnrichedAudioResult):
        return response
    if isinstance(response, dict) and "parsed" in response:
        parsed = response.get("parsed")
        if isinstance(parsed, EnrichedAudioResult):
            return parsed
        if parsed is not None:
            return EnrichedAudioResult.model_validate(parsed)
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
    return EnrichedAudioResult.model_validate(response)


def _build_file_search_text(result: AudioIndexResult) -> str | None:
    parts = [result.filename, result.relative_path, "Audio transcript"]
    if result.duration_seconds is not None:
        parts.append(f"Duration: {result.duration_seconds:.1f} seconds")
    if result.file_summary:
        parts.append(result.file_summary)
    if result.file_keywords:
        parts.append(", ".join(result.file_keywords))
    if result.transcript_text:
        parts.append(result.transcript_text[:1600])
    return "\n".join(part for part in parts if part).strip() or None


def _sample_evenly[T](items: list[T], limit: int) -> list[T]:
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return items
    if limit == 1:
        return [items[0]]
    max_index = len(items) - 1
    sampled_indices = [(position * max_index) // (limit - 1) for position in range(limit)]
    return [items[index] for index in sampled_indices]
