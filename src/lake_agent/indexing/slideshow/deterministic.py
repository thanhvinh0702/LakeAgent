from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from lake_agent.domain.indexing_models import (
    SlideshowEmbeddedImage,
    SlideshowIndexResult,
    SlideshowSection,
)

_SUPPORTED_DIRECT_FORMATS = {"pptx"}
_SUPPORTED_CONVERTIBLE_FORMATS = {"ppt"}
_SUPPORTED_FORMATS = _SUPPORTED_DIRECT_FORMATS | _SUPPORTED_CONVERTIBLE_FORMATS
_SLIDESHOW_CONVERTER: Any | None = None


@dataclass(frozen=True, slots=True)
class SlideshowParseOptions:
    libreoffice_binary: str = "soffice"
    min_section_chars: int = 140
    target_section_chars: int = 500
    max_section_chars: int = 1200


@dataclass(slots=True)
class _PreparedSource:
    path: Path
    warnings: list[str]
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


@dataclass(frozen=True, slots=True)
class _RawSlideChunk:
    heading: str | None
    content: str
    slide_start: int | None
    slide_end: int | None
    search_text: str


class DeterministicSlideshowParser:
    def __init__(
        self,
        options: SlideshowParseOptions | None = None,
        *,
        load_document: Callable[[Path], Any] | None = None,
        build_chunker: Callable[[], Any] | None = None,
        prepare_source: Callable[[Path], _PreparedSource] | None = None,
        extract_embedded_images: Callable[[Any, str], tuple[list[SlideshowEmbeddedImage], str | None, list[str]]] | None = None,
    ) -> None:
        self._options = options or SlideshowParseOptions()
        self._load_document = load_document or _default_load_document
        self._build_chunker = build_chunker or _default_build_chunker
        self._prepare_source = prepare_source or self._default_prepare_source
        self._extract_embedded_images = extract_embedded_images or _extract_embedded_images

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> SlideshowIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported deterministic slideshow format: {extension}")

        normalized_relative_path = relative_path or path.name
        normalized_relative_path = PurePosixPath(
            normalized_relative_path.replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        prepared = self._prepare_source(path)
        try:
            dl_doc = self._load_document(prepared.path)
            chunker = self._build_chunker()
            sections = self._build_sections(
                list(chunker.chunk(dl_doc=dl_doc)),
                chunker=chunker,
                source_id=normalized_source_id,
            )
            embedded_images, artifact_dir, image_warnings = self._extract_embedded_images(
                dl_doc,
                normalized_source_id,
            )
        finally:
            prepared.cleanup()

        warnings = list(prepared.warnings)
        warnings.extend(image_warnings)
        if not sections:
            warnings.append("The slideshow did not contain any readable slide chunks.")

        result = SlideshowIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format=extension,
            sections=sections,
            parse_warnings=warnings,
            embedded_images=embedded_images,
            artifact_dir=artifact_dir,
        )
        result.file_search_text = _build_file_search_text(result)
        return result

    def _build_sections(
        self,
        chunks: list[Any],
        *,
        chunker: Any,
        source_id: str,
    ) -> list[SlideshowSection]:
        raw_chunks: list[_RawSlideChunk] = []
        for chunk in chunks:
            content = (getattr(chunk, "text", "") or "").strip()
            if not content:
                continue
            headings = list(getattr(getattr(chunk, "meta", None), "headings", []) or [])
            heading = " > ".join(part.strip() for part in headings if part and part.strip()) or None
            slide_start, slide_end = _extract_slide_range(chunk)
            search_text = chunker.contextualize(chunk=chunk).strip() or content
            raw_chunks.append(
                _RawSlideChunk(
                    heading=heading,
                    content=content,
                    slide_start=slide_start,
                    slide_end=slide_end,
                    search_text=search_text,
                )
            )

        merged_chunks = self._build_slide_sections(raw_chunks)
        sections: list[SlideshowSection] = []
        for chunk_index, chunk in enumerate(merged_chunks, start=1):
            sections.append(
                SlideshowSection(
                    section_id=_stable_id(
                        f"{source_id}:{chunk_index}:{chunk.heading or ''}:{chunk.content[:80]}",
                        prefix="slide_section",
                    ),
                    section_type="slide_chunk",
                    chunk_index=chunk_index,
                    heading=chunk.heading,
                    content=chunk.content,
                    slide_start=chunk.slide_start,
                    slide_end=chunk.slide_end,
                    char_count=len(chunk.content),
                    search_text=chunk.search_text,
                )
            )
        return sections

    def _build_slide_sections(self, chunks: list[_RawSlideChunk]) -> list[_RawSlideChunk]:
        if not chunks:
            return []

        sections: list[_RawSlideChunk] = []
        for slide_chunks in _group_chunks_by_slide(chunks):
            sections.extend(self._split_within_slide(slide_chunks))
        return sections

    def _split_within_slide(self, slide_chunks: list[_RawSlideChunk]) -> list[_RawSlideChunk]:
        if not slide_chunks:
            return []

        sections: list[_RawSlideChunk] = []
        current = slide_chunks[0]
        for next_chunk in slide_chunks[1:]:
            if self._should_merge_within_slide(current, next_chunk):
                current = _merge_raw_chunks(current, next_chunk)
                continue
            sections.append(current)
            current = next_chunk
        sections.append(current)
        return sections

    def _should_merge_within_slide(
        self,
        current: _RawSlideChunk,
        next_chunk: _RawSlideChunk,
    ) -> bool:
        if len(current.content) < self._options.min_section_chars:
            return len(current.content) + len(next_chunk.content) <= self._options.max_section_chars

        if len(current.content) >= self._options.target_section_chars:
            return False

        if len(current.content) + len(next_chunk.content) > self._options.max_section_chars:
            return False

        if not _headings_compatible(current.heading, next_chunk.heading):
            return False

        return True

    def _default_prepare_source(self, path: Path) -> _PreparedSource:
        extension = path.suffix.lower().removeprefix(".")
        if extension in _SUPPORTED_DIRECT_FORMATS:
            return _PreparedSource(path=path, warnings=[])

        if extension not in _SUPPORTED_CONVERTIBLE_FORMATS:
            raise ValueError(f"Unsupported deterministic slideshow format: {extension}")

        libreoffice_binary = _resolve_office_binary(self._options.libreoffice_binary)
        if not libreoffice_binary:
            raise RuntimeError(
                f"Legacy slideshow format '.{extension}' requires LibreOffice conversion, "
                f"but none of the candidate binaries were found. "
                f"requested={self._options.libreoffice_binary!r}, fallback=('soffice', 'libreoffice')."
            )
        if _is_snap_libreoffice_wrapper(libreoffice_binary):
            raise RuntimeError(
                "Legacy .ppt conversion requires a native LibreOffice/soffice binary. "
                "This machine currently resolves LibreOffice to the snap/App Center wrapper "
                f"at {libreoffice_binary}, which reports conversion success without creating "
                "a usable .pptx output on this system. Use .pptx files directly or install "
                "a native LibreOffice package that provides 'soffice'."
            )

        temp_dir = tempfile.TemporaryDirectory(prefix="lake_agent_slideshow_")
        temp_path = Path(temp_dir.name)
        subprocess.run(
            [
                libreoffice_binary,
                "--headless",
                "--convert-to",
                "pptx",
                "--outdir",
                str(temp_path),
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        converted_path = _resolve_converted_output(
            output_dir=temp_path,
            original_path=path,
            expected_suffix=".pptx",
        )
        if converted_path is None:
            snap_hint = ""
            if "/snap/bin/" in libreoffice_binary:
                snap_hint = (
                    " The current LibreOffice binary comes from snap and reported a converted file "
                    "path without actually creating the .pptx on this machine. Prefer a native "
                    "'soffice' binary if available."
                )
            raise RuntimeError(
                f"LibreOffice conversion did not produce a usable .pptx output for {path.name}."
                f"{snap_hint}"
            )
        return _PreparedSource(
            path=converted_path,
            warnings=[
                f"Converted legacy .{extension} file to .pptx before Docling parsing."
            ],
            temp_dir=temp_dir,
        )


def _default_load_document(path: Path) -> Any:
    converter = _get_default_slideshow_converter()
    return converter.convert(source=str(path)).document


def _get_default_slideshow_converter() -> Any:
    global _SLIDESHOW_CONVERTER
    if _SLIDESHOW_CONVERTER is not None:
        return _SLIDESHOW_CONVERTER

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import ConvertPipelineOptions
    from docling.document_converter import DocumentConverter, PowerpointFormatOption

    _SLIDESHOW_CONVERTER = DocumentConverter(
        format_options={
            InputFormat.PPTX: PowerpointFormatOption(
                pipeline_options=ConvertPipelineOptions()
            ),
        }
    )
    return _SLIDESHOW_CONVERTER


def _default_build_chunker() -> Any:
    from docling.chunking import HierarchicalChunker

    return HierarchicalChunker()


def _extract_slide_range(chunk: Any) -> tuple[int | None, int | None]:
    direct_prov = getattr(chunk, "prov", None)
    if direct_prov:
        slide_numbers = [
            prov.page_no
            for prov in direct_prov
            if isinstance(getattr(prov, "page_no", None), int)
        ]
        if slide_numbers:
            return min(slide_numbers), max(slide_numbers)

    meta = getattr(chunk, "meta", None)
    if meta is None:
        return None, None

    slide_numbers: list[int] = []
    for doc_item in getattr(meta, "doc_items", []) or []:
        for prov in getattr(doc_item, "prov", []) or []:
            slide_no = getattr(prov, "page_no", None)
            if isinstance(slide_no, int):
                slide_numbers.append(slide_no)
    if not slide_numbers:
        return None, None
    return min(slide_numbers), max(slide_numbers)


def _build_file_search_text(result: SlideshowIndexResult) -> str | None:
    parts = [result.filename, result.relative_path]
    for section in result.sections[:3]:
        if section.heading:
            parts.append(section.heading)
        if section.slide_start is not None:
            if section.slide_end is not None and section.slide_end != section.slide_start:
                parts.append(f"slides {section.slide_start}-{section.slide_end}")
            else:
                parts.append(f"slide {section.slide_start}")
    return "\n".join(part for part in parts if part).strip() or None


def _extract_embedded_images(
    dl_doc: Any,
    source_id: str,
) -> tuple[list[SlideshowEmbeddedImage], str | None, list[str]]:
    pictures = list(getattr(dl_doc, "pictures", []) or [])
    if not pictures:
        return [], None, []

    temp_dir = tempfile.mkdtemp(prefix="lake_agent_slideshow_images_")
    extracted: list[SlideshowEmbeddedImage] = []
    warnings: list[str] = []
    try:
        for image_index, picture in enumerate(pictures, start=1):
            try:
                pil_image = picture.get_image(dl_doc)
            except Exception as exc:
                warnings.append(f"Unable to extract slideshow image {image_index}: {exc}")
                continue
            if pil_image is None:
                warnings.append(f"Slideshow image {image_index} did not expose a readable bitmap.")
                continue

            image_filename = f"{source_id}_image_{image_index:03d}.png"
            image_path = Path(temp_dir) / image_filename
            with io.BytesIO() as output:
                pil_image.save(output, format="PNG")
                image_path.write_bytes(output.getvalue())

            slide_start, slide_end = _extract_slide_range(picture)
            caption = None
            caption_text = getattr(picture, "caption_text", None)
            if callable(caption_text):
                try:
                    caption = caption_text(dl_doc).strip() or None
                except Exception:
                    caption = None

            extracted.append(
                SlideshowEmbeddedImage(
                    image_id=_stable_id(
                        f"{source_id}:image:{image_index}:{image_filename}",
                        prefix="slideimg",
                    ),
                    image_index=image_index,
                    path=os.fspath(image_path),
                    filename=image_filename,
                    width=pil_image.width,
                    height=pil_image.height,
                    color_mode=pil_image.mode,
                    slide_start=slide_start,
                    slide_end=slide_end,
                    caption=caption,
                )
            )
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    if not extracted:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return [], None, warnings
    return extracted, temp_dir, warnings


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _resolve_converted_output(
    *,
    output_dir: Path,
    original_path: Path,
    expected_suffix: str,
) -> Path | None:
    direct_match = output_dir / f"{original_path.stem}{expected_suffix}"
    if direct_match.exists():
        return direct_match

    candidates = sorted(
        output_dir.glob(f"*{expected_suffix}"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    normalized_stem = _normalize_filename(original_path.stem)
    for candidate in candidates:
        if _normalize_filename(candidate.stem) == normalized_stem:
            return candidate
    for candidate in candidates:
        candidate_stem = _normalize_filename(candidate.stem)
        if candidate_stem.startswith(normalized_stem) or normalized_stem.startswith(candidate_stem):
            return candidate
    return candidates[0]


def _normalize_filename(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _resolve_office_binary(requested_binary: str) -> str | None:
    candidates: list[str] = []
    if requested_binary:
        candidates.append(requested_binary)
    for fallback in ("soffice", "libreoffice"):
        if fallback not in candidates:
            candidates.append(fallback)
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _is_snap_libreoffice_wrapper(binary_path: str) -> bool:
    normalized = binary_path.replace("\\", "/")
    return normalized == "/snap/bin/libreoffice"


def _headings_compatible(left: str | None, right: str | None) -> bool:
    if left == right:
        return True
    if not left or not right:
        return True
    return left.startswith(right) or right.startswith(left)


def _join_content(left: str, right: str) -> str:
    left = left.strip()
    right = right.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}\n{right}"


def _min_defined(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _max_defined(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _group_chunks_by_slide(chunks: list[_RawSlideChunk]) -> list[list[_RawSlideChunk]]:
    grouped: list[list[_RawSlideChunk]] = []
    current_group: list[_RawSlideChunk] = []
    current_key: tuple[int | None, int | None] | None = None

    for chunk in chunks:
        key = (chunk.slide_start, chunk.slide_end)
        if not current_group or key == current_key:
            current_group.append(chunk)
            current_key = key
            continue
        grouped.append(current_group)
        current_group = [chunk]
        current_key = key

    if current_group:
        grouped.append(current_group)
    return grouped


def _merge_raw_chunks(left: _RawSlideChunk, right: _RawSlideChunk) -> _RawSlideChunk:
    return _RawSlideChunk(
        heading=left.heading or right.heading,
        content=_join_content(left.content, right.content),
        slide_start=_min_defined(left.slide_start, right.slide_start),
        slide_end=_max_defined(left.slide_end, right.slide_end),
        search_text=_join_content(left.search_text, right.search_text),
    )
