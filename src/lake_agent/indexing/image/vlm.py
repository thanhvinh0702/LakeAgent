from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from lake_agent.config import VLSettings
from lake_agent.domain.indexing_models import EnrichedImageResult, ImageIndexResult

_SYSTEM_PROMPT = """
You enrich image files for a data lake index.

Your job:
- describe what is visibly present in the image
- use filename context when it likely names the subject, document, or event
- keep the output grounded in the image and provided context
- do not invent identities or facts that are not reasonably supported
- prefer concise, retrieval-friendly summaries and keywords
""".strip()

@dataclass(frozen=True, slots=True)
class ImageEnrichmentOptions:
    keyword_limit: int = 8
    ocr_character_limit: int = 1200
    max_long_edge: int = 1536
    jpeg_quality: int = 85


class ImageVLMEnricher:
    def __init__(
        self,
        invoke_enrichment: Callable[[str, str, Path], EnrichedImageResult],
        invoke_batch_enrichment: Callable[[list[tuple[str, str, Path]]], list[EnrichedImageResult]] | None = None,
        options: ImageEnrichmentOptions | None = None,
    ) -> None:
        self._invoke_enrichment = invoke_enrichment
        self._invoke_batch_enrichment = invoke_batch_enrichment
        self._options = options or ImageEnrichmentOptions()

    @classmethod
    def from_env(
        cls,
        options: ImageEnrichmentOptions | None = None,
    ) -> "ImageVLMEnricher":
        settings = VLSettings.from_env()
        resolved_options = options or ImageEnrichmentOptions()
        return cls(
            invoke_enrichment=_build_langchain_enrichment_invoker(settings, resolved_options),
            invoke_batch_enrichment=_build_langchain_batch_enrichment_invoker(settings, resolved_options),
            options=resolved_options,
        )

    def enrich(self, image_path: str | Path, result: ImageIndexResult) -> ImageIndexResult:
        path = Path(image_path).expanduser().resolve()
        payload = self._build_payload(result)
        enriched = self._invoke_enrichment(_SYSTEM_PROMPT, _build_user_prompt(payload), path)
        _apply_enrichment(result, enriched, self._options)
        return result

    def enrich_batch(
        self,
        image_paths: list[str | Path],
        results: list[ImageIndexResult],
    ) -> list[ImageIndexResult]:
        if not image_paths or not results:
            return []
        if len(image_paths) != len(results):
            raise ValueError("image_paths and results must have the same length")
        if self._invoke_batch_enrichment is None or len(results) == 1:
            return [
                self.enrich(image_path, result)
                for image_path, result in zip(image_paths, results, strict=True)
            ]

        payloads = [
            (
                _SYSTEM_PROMPT,
                _build_user_prompt(self._build_payload(result)),
                Path(image_path).expanduser().resolve(),
            )
            for image_path, result in zip(image_paths, results, strict=True)
        ]
        enriched_results = self._invoke_batch_enrichment(payloads)
        if len(enriched_results) != len(results):
            raise RuntimeError(
                "VLM batch enrichment returned a different number of results than inputs. "
                f"expected={len(results)}, actual={len(enriched_results)}"
            )
        for result, enriched in zip(results, enriched_results, strict=True):
            _apply_enrichment(result, enriched, self._options)
        return results

    def _build_payload(self, result: ImageIndexResult) -> dict[str, Any]:
        return {
            "source_id": result.source_id,
            "relative_path": result.relative_path,
            "filename": result.filename,
            "filename_context": _filename_context(result.filename),
            "file_format": result.file_format,
            "width": result.width,
            "height": result.height,
            "color_mode": result.color_mode,
            "has_alpha": result.has_alpha,
            "is_animated": result.is_animated,
            "parse_warnings": result.parse_warnings,
            "ocr_excerpt": _ocr_excerpt(result, limit=self._options.ocr_character_limit),
        }


def _build_langchain_enrichment_invoker(
    settings: VLSettings,
    options: ImageEnrichmentOptions,
) -> Callable[[str, str, Path], EnrichedImageResult]:
    client = init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedImageResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_enrichment(
        system_prompt: str,
        user_prompt: str,
        image_path: Path,
    ) -> EnrichedImageResult:
        response = client.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=_build_image_message_content(user_prompt, image_path, options)),
            ]
        )
        return _parse_enrichment_response(response, settings)

    return invoke_enrichment


def _build_langchain_batch_enrichment_invoker(
    settings: VLSettings,
    options: ImageEnrichmentOptions,
) -> Callable[[list[tuple[str, str, Path]]], list[EnrichedImageResult]]:
    client = init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
    ).with_structured_output(
        EnrichedImageResult,
        method="function_calling",
        include_raw=True,
    )

    def invoke_batch_enrichment(
        prompt_triplets: list[tuple[str, str, Path]],
    ) -> list[EnrichedImageResult]:
        responses = client.batch(
            [
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=_build_image_message_content(user_prompt, image_path, options)),
                ]
                for system_prompt, user_prompt, image_path in prompt_triplets
            ]
        )
        return [_parse_enrichment_response(response, settings) for response in responses]

    return invoke_batch_enrichment


def _build_user_prompt(payload: dict[str, Any]) -> str:
    instructions = {
        "rules": [
            "Describe what is actually visible in the image.",
            "Use filename context when it likely identifies the subject or scene.",
            "For example, a portrait file named Nguyen_A.jpg likely refers to Nguyen A.",
            "Do not claim a person's identity unless the filename or image gives reasonable support.",
            "If visible text appears in the image, you may mention it briefly.",
            "Keep the summary concise and retrieval-oriented.",
            "Keywords should be short noun phrases, not sentences.",
        ],
        "input": payload,
    }
    return json.dumps(instructions, ensure_ascii=True, indent=2)


def _build_image_message_content(
    user_prompt: str,
    image_path: Path,
    options: ImageEnrichmentOptions,
) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": user_prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": _image_data_uri(image_path, options),
            },
        },
    ]


def _image_data_uri(image_path: Path, options: ImageEnrichmentOptions) -> str:
    payload = _prepare_vlm_image_payload(image_path, options)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _prepare_vlm_image_payload(
    image_path: Path,
    options: ImageEnrichmentOptions,
) -> bytes:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return image_path.read_bytes()

    with Image.open(image_path) as source_image:
        image = ImageOps.exif_transpose(source_image)
        if getattr(image, "is_animated", False):
            image.seek(0)
            image = image.copy()
        else:
            image = image.copy()

    resized = _resize_for_vlm(image, max_long_edge=options.max_long_edge)
    return _encode_vlm_image(resized, jpeg_quality=options.jpeg_quality)


def _resize_for_vlm(image: Any, *, max_long_edge: int) -> Any:
    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= max_long_edge:
        return image

    scale = max_long_edge / longest_edge
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resampling_module = getattr(image, "Resampling", None)
    if resampling_module is not None:
        resample_filter = resampling_module.LANCZOS
    else:
        resample_filter = getattr(image, "LANCZOS", 1)
    return image.resize((resized_width, resized_height), resample=resample_filter)


def _encode_vlm_image(
    image: Any,
    *,
    jpeg_quality: int,
) -> bytes:
    output = io.BytesIO()
    converted = _flatten_to_rgb(image)
    converted.save(output, format="JPEG", quality=jpeg_quality, optimize=True)
    return output.getvalue()


def _flatten_to_rgb(image: Any) -> Any:
    from PIL import Image

    if image.mode == "RGB":
        return image

    if image.mode in {"RGBA", "LA"}:
        background = image.getchannel("A")
        canvas = image.convert("RGBA")
        flattened = Image.new("RGB", canvas.size, (255, 255, 255))
        flattened.paste(canvas, mask=background)
        return flattened

    if image.mode == "P":
        return image.convert("RGBA").convert("RGB")

    return image.convert("RGB")


def _parse_enrichment_response(
    response: Any,
    settings: VLSettings,
) -> EnrichedImageResult:
    if isinstance(response, EnrichedImageResult):
        return response

    if isinstance(response, dict) and "parsed" in response:
        parsed = response.get("parsed")
        if isinstance(parsed, EnrichedImageResult):
            return parsed
        if parsed is not None:
            return EnrichedImageResult.model_validate(parsed)
        raise RuntimeError(
            "VLM structured output returned no parsed result. "
            f"model={settings.model_name!r}, base_url={settings.base_url!r}, "
            f"parsing_error={response.get('parsing_error')!r}, "
            f"raw_response={response.get('raw')!r}"
        )

    if response is None:
        raise RuntimeError(
            "VLM returned None for structured output. "
            f"model={settings.model_name!r}, base_url={settings.base_url!r}."
        )

    return EnrichedImageResult.model_validate(response)


def _apply_enrichment(
    result: ImageIndexResult,
    enriched: EnrichedImageResult,
    options: ImageEnrichmentOptions,
) -> None:
    result.file_summary = enriched.file_summary
    result.file_keywords = enriched.file_keywords[: options.keyword_limit]
    result.file_search_text = _build_file_search_text(result)


def _build_file_search_text(result: ImageIndexResult) -> str:
    parts = [
        result.filename,
        result.relative_path,
        result.file_format,
        f"{result.width}x{result.height}",
        result.color_mode,
    ]
    if result.file_summary:
        parts.append(result.file_summary)
    if result.file_keywords:
        parts.append(", ".join(result.file_keywords))
    for section in result.sections[:2]:
        if section.heading:
            parts.append(section.heading)
    if result.is_animated:
        parts.append("animated")
    if result.has_alpha:
        parts.append("transparent")
    return "\n".join(part for part in parts if part).strip()


def _filename_context(filename: str) -> str | None:
    stem = Path(filename).stem
    cleaned = re.sub(r"[_\-]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _ocr_excerpt(result: ImageIndexResult, *, limit: int) -> str | None:
    if not result.sections:
        return None
    text = "\n\n".join(section.content.strip() for section in result.sections if section.content.strip())
    if not text:
        return None
    return text[:limit]
