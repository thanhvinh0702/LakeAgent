from __future__ import annotations

import base64
import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from lake_agent.config import OCRSettings
from lake_agent.domain.indexing_models import ImageSection
from lake_agent.indexing.text.chunking import (
    build_basic_search_text,
    chunk_markdown_text,
    normalize_text,
)

_OCR_SYSTEM_PROMPT = """
You extract readable text from an image for data lake indexing.

Rules:
- Perform OCR from the image content.
- Return only the extracted text.
- Preserve visible structure when possible using plain text or simple markdown.
- Do not describe the scene unless it is needed to disambiguate visible text.
- Do not add commentary, explanations, or confidence notes.
- If no readable text is visible, return an empty string.
""".strip()


@dataclass(frozen=True, slots=True)
class OCRExtractionOptions:
    batch_size: int = 3
    max_chars_per_chunk: int = 2400
    min_chunk_chars: int = 400
    max_long_edge: int = 1536
    jpeg_quality: int = 85


class OCRMarkdownExtractor:
    def __init__(
        self,
        client: Any,
        options: OCRExtractionOptions | None = None,
    ) -> None:
        self._client = client
        self._options = options or OCRExtractionOptions()

    @classmethod
    def from_env(
        cls,
        options: OCRExtractionOptions | None = None,
    ) -> "OCRMarkdownExtractor":
        settings = OCRSettings.from_env()
        client = _build_langchain_ocr_client(settings)
        return cls(client=client, options=options)

    def extract_sections(self, image_path: str | Path, *, source_id: str) -> list[ImageSection]:
        markdown = self._invoke_ocr_batch([Path(image_path)])[0]
        return _build_sections_from_markdown(
            markdown,
            source_id=source_id,
            options=self._options,
        )

    def extract_sections_batch(
        self,
        image_paths: list[str | Path],
        *,
        source_ids: list[str],
    ) -> dict[str, list[ImageSection]]:
        if len(image_paths) != len(source_ids):
            raise ValueError("image_paths and source_ids must have the same length")
        if not image_paths:
            return {}

        markdown_results = self._invoke_ocr_batch([Path(path) for path in image_paths])
        return {
            source_id: _build_sections_from_markdown(
                markdown,
                source_id=source_id,
                options=self._options,
            )
            for source_id, markdown in zip(source_ids, markdown_results, strict=True)
        }

    @property
    def batch_size(self) -> int:
        return self._options.batch_size

    def _invoke_ocr_batch(self, image_paths: list[Path]) -> list[str]:
        try:
            responses = self._client.batch(
                [
                    [
                        SystemMessage(content=_OCR_SYSTEM_PROMPT),
                        HumanMessage(
                            content=_build_ocr_message_content(
                                path.expanduser().resolve(),
                                self._options,
                            )
                        ),
                    ]
                    for path in image_paths
                ]
            )
        except Exception as exc:
            batch_preview = ", ".join(path.name for path in image_paths[:3])
            if len(image_paths) > 3:
                batch_preview += ", ..."
            raise RuntimeError(
                f"OCR request failed for batch of {len(image_paths)} image(s) "
                f"[{batch_preview}]: {exc}"
            ) from exc
        return [_extract_text_response(response).strip() for response in responses]


def _build_sections_from_markdown(
    markdown: str,
    *,
    source_id: str,
    options: OCRExtractionOptions,
) -> list[ImageSection]:
    normalized = normalize_text(markdown)
    if not normalized:
        return []

    chunks = chunk_markdown_text(
        normalized,
        max_chars=options.max_chars_per_chunk,
        min_chars=options.min_chunk_chars,
    )
    return [
        ImageSection(
            section_id=_stable_id(
                f"{source_id}:ocr:{chunk.chunk_index}:{chunk.heading or ''}",
                prefix="imgsec",
            ),
            section_type="ocr_chunk",
            chunk_index=chunk.chunk_index,
            heading=chunk.heading,
            content=chunk.content,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            char_count=len(chunk.content),
            search_text=build_basic_search_text(chunk.heading, chunk.content),
        )
        for chunk in chunks
    ]


def _build_langchain_ocr_client(settings: OCRSettings) -> Any:
    return init_chat_model(
        model_provider="openai",
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model_name,
        temperature=0,
        timeout=settings.request_timeout_seconds,
    )


def _build_ocr_message_content(
    image_path: Path,
    options: OCRExtractionOptions,
) -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": "Extract all readable text from this image. Return only the extracted text.",
        },
        {
            "type": "image_url",
            "image_url": {
                "url": _image_data_uri(image_path, options),
            },
        },
    ]


def _image_data_uri(image_path: Path, options: OCRExtractionOptions) -> str:
    payload = _prepare_image_payload(image_path, options)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _prepare_image_payload(
    image_path: Path,
    options: OCRExtractionOptions,
) -> bytes:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return image_path.read_bytes()

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
            category=UserWarning,
        )
        source_image = Image.open(image_path)
        if source_image.mode == "P" and "transparency" in source_image.info:
            source_image = source_image.convert("RGBA")
        image = ImageOps.exif_transpose(source_image)
        if getattr(image, "is_animated", False):
            image.seek(0)
            image = image.copy()
        else:
            image = image.copy()
        source_image.close()

    resized = _resize_for_ocr(image, max_long_edge=options.max_long_edge)
    return _encode_ocr_image(resized, jpeg_quality=options.jpeg_quality)


def _resize_for_ocr(image: Any, *, max_long_edge: int) -> Any:
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


def _encode_ocr_image(image: Any, *, jpeg_quality: int) -> bytes:
    if image.mode not in {"RGB", "L"}:
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
    )
    return buffer.getvalue()


def _extract_text_response(response: Any) -> str:
    content = getattr(response, "text", None)
    if isinstance(content, str):
        return content

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(response) if response is not None else ""


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
