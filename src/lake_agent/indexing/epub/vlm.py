from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from lake_agent.config import EpubVLSettings
from lake_agent.domain.indexing_models import EpubEmbeddedImage


@dataclass(frozen=True, slots=True)
class EpubImageCaption:
    image_id: str
    image_index: int
    content: str
    model_name: str
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class EpubVLMOptions:
    max_long_edge: int = 768
    jpeg_quality: int = 82


class EpubImageVLMCaptioner:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model_name: str,
        options: EpubVLMOptions | None = None,
    ) -> None:
        self._model_name = model_name
        self._options = options or EpubVLMOptions()
        self._model = init_chat_model(
            model_provider="openai",
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            temperature=0,
        )

    @classmethod
    def from_env(
        cls,
        options: EpubVLMOptions | None = None,
    ) -> "EpubImageVLMCaptioner":
        settings = EpubVLSettings.from_env()
        return cls(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model_name=settings.model_name,
            options=options,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def caption_image(
        self,
        image: EpubEmbeddedImage,
        *,
        epub_title: str | None,
        epub_filename: str,
    ) -> EpubImageCaption:
        image_path = Path(image.path).expanduser().resolve()
        prompt = (
            "Describe this EPUB/light novel image for search indexing. Be factual and concise. "
            "Mention visible people, objects, scene, actions, text if readable, and visual style. "
            "Do not infer identities or story events that are not visible. "
            f"EPUB title: {epub_title or epub_filename}. "
            f"EPUB filename: {epub_filename}. "
            f"Image href: {image.href}. "
            f"Image filename/context: {image.caption or image.filename}."
        )
        response = self._model.invoke(
            [
                SystemMessage(content="You write short grounded captions for EPUB images."),
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_data_uri(image_path, self._options)},
                        },
                    ]
                ),
            ]
        )
        content = str(getattr(response, "content", response)).strip()
        return EpubImageCaption(
            image_id=image.image_id,
            image_index=image.image_index,
            content=content,
            model_name=self._model_name,
            warnings=list(image.warnings),
        )


def _image_data_uri(image_path: Path, options: EpubVLMOptions) -> str:
    payload = _prepare_vlm_image_payload(image_path, options)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _prepare_vlm_image_payload(image_path: Path, options: EpubVLMOptions) -> bytes:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return image_path.read_bytes()

    with Image.open(image_path) as source_image:
        image = ImageOps.exif_transpose(source_image)
        if getattr(image, "is_animated", False):
            image.seek(0)
        image = image.convert("RGB").copy()

    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge > options.max_long_edge:
        scale = options.max_long_edge / longest_edge
        image = image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            resample=getattr(Image.Resampling, "LANCZOS", 1),
        )

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=options.jpeg_quality, optimize=True)
    return buffer.getvalue()
