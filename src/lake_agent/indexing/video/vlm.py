from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from lake_agent.config import VideoVLSettings


@dataclass(frozen=True, slots=True)
class VideoFrameCaption:
    timestamp_seconds: float
    frame_index: int
    content: str
    model_name: str
    warnings: list[str]


@dataclass(frozen=True, slots=True)
class VideoVLMOptions:
    max_long_edge: int = 1024
    jpeg_quality: int = 82


class VideoFrameVLMCaptioner:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        model_name: str,
        options: VideoVLMOptions | None = None,
    ) -> None:
        self._model_name = model_name
        self._options = options or VideoVLMOptions()
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
        options: VideoVLMOptions | None = None,
    ) -> "VideoFrameVLMCaptioner":
        settings = VideoVLSettings.from_env()
        return cls(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model_name=settings.model_name,
            options=options,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def caption_frame(
        self,
        frame_path: str | Path,
        *,
        timestamp_seconds: float,
        frame_index: int,
        video_filename: str,
    ) -> VideoFrameCaption:
        path = Path(frame_path).expanduser().resolve()
        prompt = (
            "Describe this video frame for search indexing. Be factual and concise. "
            "Mention visible people, objects, scene, actions, screen text if readable, "
            "and any notable visual state. Do not infer identities. "
            f"Video filename: {video_filename}. "
            f"Frame timestamp: {timestamp_seconds:.1f} seconds."
        )
        response = self._model.invoke(
            [
                SystemMessage(content="You write short grounded video frame captions."),
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_data_uri(path, self._options),
                            },
                        },
                    ]
                ),
            ]
        )
        content = str(getattr(response, "content", response)).strip()
        return VideoFrameCaption(
            timestamp_seconds=timestamp_seconds,
            frame_index=frame_index,
            content=content,
            model_name=self._model_name,
            warnings=[],
        )


def _image_data_uri(image_path: Path, options: VideoVLMOptions) -> str:
    payload = _prepare_vlm_image_payload(image_path, options)
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _prepare_vlm_image_payload(image_path: Path, options: VideoVLMOptions) -> bytes:
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return image_path.read_bytes()

    with Image.open(image_path) as source_image:
        image = ImageOps.exif_transpose(source_image).convert("RGB")
        image = image.copy()

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
