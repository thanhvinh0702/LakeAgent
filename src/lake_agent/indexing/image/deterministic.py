from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from lake_agent.domain.indexing_models import ImageIndexResult

_SUPPORTED_FORMATS = {"jpeg", "png", "gif", "webp", "tiff"}
_ALPHA_MODES = {"RGBA", "LA", "PA"}


@dataclass(frozen=True, slots=True)
class ImageParseOptions:
    pass


class DeterministicImageParser:
    def __init__(self, options: ImageParseOptions | None = None) -> None:
        self._options = options or ImageParseOptions()

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> ImageIndexResult:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Image indexing requires Pillow. Install the project dependencies first."
            ) from exc

        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        normalized_relative_path = relative_path or path.name
        normalized_relative_path = PurePosixPath(
            normalized_relative_path.replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        warnings: list[str] = []
        with Image.open(path) as image:
            detected_format = (image.format or "").lower()
            if detected_format == "jpg":
                detected_format = "jpeg"
            if detected_format not in _SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported deterministic image format: {detected_format or 'unknown'}")

            width, height = image.size
            color_mode = image.mode
            frame_count = getattr(image, "n_frames", 1)
            is_animated = frame_count > 1
            has_alpha = color_mode in _ALPHA_MODES or "transparency" in image.info

        result = ImageIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format=detected_format,
            width=width,
            height=height,
            color_mode=color_mode,
            has_alpha=has_alpha,
            is_animated=is_animated,
            frame_count=frame_count,
            parse_warnings=warnings,
        )
        result.file_search_text = _build_file_search_text(result)
        return result


def _build_file_search_text(result: ImageIndexResult) -> str:
    parts = [
        result.filename,
        result.relative_path,
        result.file_format,
        f"{result.width}x{result.height}",
        result.color_mode,
    ]
    if result.is_animated:
        parts.append("animated")
    if result.has_alpha:
        parts.append("transparent")
    return "\n".join(part for part in parts if part).strip()


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
