from __future__ import annotations

import hashlib
import os
import posixpath
import re
import tempfile
import zipfile
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree

from lake_agent.domain.indexing_models import (
    EpubEmbeddedImage,
    EpubIndexResult,
    EpubSection,
)
from lake_agent.indexing.text.chunking import (
    build_basic_search_text,
    chunk_plain_text,
    normalize_text,
)

_SUPPORTED_FORMATS = {"epub"}
_HTML_MEDIA_TYPES = {
    "application/xhtml+xml",
    "text/html",
}


@dataclass(frozen=True, slots=True)
class EpubParseOptions:
    max_chars_per_chunk: int = 2400
    min_chunk_chars: int = 400
    target_chars_per_chunk: int = 1000
    extract_images: bool = True
    max_images_per_file: int = 20


@dataclass(frozen=True, slots=True)
class _ManifestItem:
    item_id: str
    href: str
    archive_path: str
    media_type: str
    properties: str | None = None


@dataclass(frozen=True, slots=True)
class _Chapter:
    index: int
    href: str
    title: str | None
    text: str


class DeterministicEpubParser:
    def __init__(self, options: EpubParseOptions | None = None) -> None:
        self._options = options or EpubParseOptions()
        if self._options.max_images_per_file < 0:
            raise ValueError("max_images_per_file must not be negative")

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> EpubIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported EPUB format: {extension}")

        normalized_relative_path = PurePosixPath(
            (relative_path or path.name).replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        warnings: list[str] = []
        with zipfile.ZipFile(path) as archive:
            package_path = _find_package_path(archive)
            package_dir = posixpath.dirname(package_path)
            package_xml = _read_xml(archive, package_path)
            metadata = _extract_metadata(package_xml)
            manifest = _extract_manifest(package_xml, package_dir)
            spine = _extract_spine(package_xml)
            chapters = _extract_chapters(archive, manifest, spine, warnings)
            embedded_images, artifact_dir = self._extract_images(
                archive,
                manifest,
                normalized_source_id,
                warnings,
            )

        sections = self._build_sections(
            chapters=chapters,
            source_id=normalized_source_id,
            book_title=metadata.get("title") or path.stem,
            relative_path=normalized_relative_path,
        )
        if not sections:
            warnings.append("The EPUB did not contain any readable text sections.")

        result = EpubIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format="epub",
            sections=sections,
            embedded_images=embedded_images,
            title=metadata.get("title"),
            creators=metadata.get("creators", []),
            language=metadata.get("language"),
            publisher=metadata.get("publisher"),
            identifier=metadata.get("identifier"),
            chapter_count=len(chapters),
            image_count=len(embedded_images),
            artifact_dir=artifact_dir,
            parse_warnings=warnings,
        )
        result.file_search_text = _build_file_search_text(result)
        return result

    def _build_sections(
        self,
        *,
        chapters: list[_Chapter],
        source_id: str,
        book_title: str,
        relative_path: str,
    ) -> list[EpubSection]:
        sections: list[EpubSection] = []
        chunk_index = 1
        for chapter in chapters:
            heading = chapter.title or f"Chapter {chapter.index}"
            contextual_text = f"# {heading}\n\n{chapter.text}".strip()
            chunks = chunk_plain_text(
                contextual_text,
                max_chars=self._options.max_chars_per_chunk,
                min_chars=self._options.min_chunk_chars,
                target_chars=self._options.target_chars_per_chunk,
            )
            for chunk in chunks:
                content = chunk.content.strip()
                if not content:
                    continue
                search_text = build_basic_search_text(
                    heading,
                    "\n".join(
                        part
                        for part in (
                            f"EPUB: {book_title}",
                            f"Path: {relative_path}",
                            f"Chapter {chapter.index}: {heading}",
                            content,
                        )
                        if part
                    ),
                )
                sections.append(
                    EpubSection(
                        section_id=_stable_id(
                            f"{source_id}:chapter:{chapter.index}:chunk:{chunk.chunk_index}",
                            prefix="section",
                        ),
                        section_type="chapter_text",
                        chunk_index=chunk_index,
                        heading=heading,
                        content=content,
                        chapter_index=chapter.index,
                        chapter_title=chapter.title,
                        chapter_href=chapter.href,
                        char_count=len(content),
                        search_text=search_text,
                    )
                )
                chunk_index += 1
        return sections

    def _extract_images(
        self,
        archive: zipfile.ZipFile,
        manifest: dict[str, _ManifestItem],
        source_id: str,
        warnings: list[str],
    ) -> tuple[list[EpubEmbeddedImage], str | None]:
        if not self._options.extract_images or self._options.max_images_per_file == 0:
            return [], None

        image_items = [
            item
            for item in manifest.values()
            if item.media_type.startswith("image/")
        ][: self._options.max_images_per_file]
        if not image_items:
            return [], None

        artifact_dir = tempfile.mkdtemp(prefix="lake_agent_epub_images_")
        embedded_images: list[EpubEmbeddedImage] = []
        seen_digests: set[str] = set()
        for item in image_items:
            try:
                payload = archive.read(item.archive_path)
            except KeyError:
                warnings.append(f"EPUB image missing from archive: {item.archive_path}")
                continue
            digest = hashlib.sha1(payload).hexdigest()
            if digest in seen_digests:
                continue
            seen_digests.add(digest)
            image_index = len(embedded_images) + 1
            filename = f"{source_id}_image_{image_index:03d}{Path(item.href).suffix or '.img'}"
            target_path = Path(artifact_dir) / filename
            target_path.write_bytes(payload)
            width, height, color_mode, image_warnings = _probe_image(target_path)
            embedded_images.append(
                EpubEmbeddedImage(
                    image_id=_stable_id(
                        f"{source_id}:image:{image_index}:{item.archive_path}",
                        prefix="image",
                    ),
                    image_index=image_index,
                    href=item.archive_path,
                    path=os.fspath(target_path),
                    filename=Path(item.href).name or filename,
                    media_type=item.media_type,
                    width=width,
                    height=height,
                    color_mode=color_mode,
                    caption=_image_caption(item),
                    warnings=image_warnings,
                )
            )
        return embedded_images, artifact_dir


def _find_package_path(archive: zipfile.ZipFile) -> str:
    try:
        container = _read_xml(archive, "META-INF/container.xml")
    except KeyError as exc:
        raise ValueError("EPUB container.xml was not found") from exc
    for element in container.iter():
        if _local_name(element.tag) == "rootfile":
            full_path = element.attrib.get("full-path")
            if full_path:
                return full_path
    raise ValueError("EPUB container.xml did not point to a package file")


def _read_xml(archive: zipfile.ZipFile, path: str) -> ElementTree.Element:
    payload = archive.read(path)
    return ElementTree.fromstring(payload)


def _extract_metadata(package_xml: ElementTree.Element) -> dict[str, Any]:
    metadata: dict[str, Any] = {"creators": []}
    for element in package_xml.iter():
        name = _local_name(element.tag)
        text = (element.text or "").strip()
        if not text:
            continue
        if name == "title" and "title" not in metadata:
            metadata["title"] = text
        elif name == "creator":
            metadata["creators"].append(text)
        elif name == "language" and "language" not in metadata:
            metadata["language"] = text
        elif name == "publisher" and "publisher" not in metadata:
            metadata["publisher"] = text
        elif name == "identifier" and "identifier" not in metadata:
            metadata["identifier"] = text
    return metadata


def _extract_manifest(
    package_xml: ElementTree.Element,
    package_dir: str,
) -> dict[str, _ManifestItem]:
    manifest: dict[str, _ManifestItem] = {}
    for element in package_xml.iter():
        if _local_name(element.tag) != "item":
            continue
        item_id = element.attrib.get("id")
        href = element.attrib.get("href")
        media_type = element.attrib.get("media-type")
        if not item_id or not href or not media_type:
            continue
        archive_path = _archive_path(package_dir, href)
        manifest[item_id] = _ManifestItem(
            item_id=item_id,
            href=href,
            archive_path=archive_path,
            media_type=media_type,
            properties=element.attrib.get("properties"),
        )
    return manifest


def _extract_spine(package_xml: ElementTree.Element) -> list[str]:
    idrefs: list[str] = []
    in_spine = False
    for event, element in _iter_element_events(package_xml):
        name = _local_name(element.tag)
        if event == "start" and name == "spine":
            in_spine = True
        elif event == "end" and name == "spine":
            in_spine = False
        elif in_spine and event == "start" and name == "itemref":
            idref = element.attrib.get("idref")
            if idref:
                idrefs.append(idref)
    return idrefs


def _iter_element_events(root: ElementTree.Element) -> list[tuple[str, ElementTree.Element]]:
    events: list[tuple[str, ElementTree.Element]] = []

    def visit(element: ElementTree.Element) -> None:
        events.append(("start", element))
        for child in list(element):
            visit(child)
        events.append(("end", element))

    visit(root)
    return events


def _extract_chapters(
    archive: zipfile.ZipFile,
    manifest: dict[str, _ManifestItem],
    spine: list[str],
    warnings: list[str],
) -> list[_Chapter]:
    chapters: list[_Chapter] = []
    for item_id in spine:
        item = manifest.get(item_id)
        if item is None or item.media_type not in _HTML_MEDIA_TYPES:
            continue
        try:
            raw_html = archive.read(item.archive_path).decode("utf-8-sig", errors="replace")
        except KeyError:
            warnings.append(f"EPUB chapter missing from archive: {item.archive_path}")
            continue
        title, text = _extract_html_text(raw_html)
        normalized = normalize_text(text)
        if not normalized.strip():
            continue
        chapters.append(
            _Chapter(
                index=len(chapters) + 1,
                href=item.archive_path,
                title=title,
                text=normalized,
            )
        )
    return chapters


def _extract_html_text(raw_html: str) -> tuple[str | None, str]:
    parser = _ReadableHTMLParser()
    parser.feed(raw_html)
    parser.close()
    return parser.title, parser.text


class _ReadableHTMLParser(HTMLParser):
    _BLOCK_TAGS = {
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "figcaption",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "p",
        "section",
        "title",
        "tr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    @property
    def title(self) -> str | None:
        value = _clean_text(" ".join(self._title_parts))
        return value or None

    @property
    def text(self) -> str:
        return _clean_text("".join(self._parts))

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "nav"}:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style", "nav"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = unescape(data)
        if self._in_title:
            self._title_parts.append(text)
        self._parts.append(text)


def _clean_text(value: str) -> str:
    value = value.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in value.split("\n")]
    paragraphs: list[str] = []
    blank = False
    for line in lines:
        if not line:
            blank = True
            continue
        if blank and paragraphs:
            paragraphs.append("")
        paragraphs.append(line)
        blank = False
    return "\n".join(paragraphs).strip()


def _archive_path(package_dir: str, href: str) -> str:
    href = href.split("#", 1)[0]
    if href.startswith("/"):
        return href.lstrip("/")
    return posixpath.normpath(posixpath.join(package_dir, href)).lstrip("./")


def _probe_image(path: Path) -> tuple[int | None, int | None, str | None, list[str]]:
    try:
        from PIL import Image
    except ImportError:
        return None, None, None, []

    try:
        with Image.open(path) as image:
            return image.width, image.height, image.mode, []
    except Exception as exc:
        return None, None, None, [f"Unable to probe EPUB image: {exc}"]


def _image_caption(item: _ManifestItem) -> str | None:
    parts = [Path(item.href).stem.replace("_", " ").replace("-", " ").strip()]
    if item.properties:
        parts.append(item.properties)
    caption = " ".join(part for part in parts if part)
    return caption or None


def _build_file_search_text(result: EpubIndexResult) -> str | None:
    parts = [
        result.title,
        result.filename,
        result.relative_path,
        ", ".join(result.creators),
        result.language,
        f"{result.chapter_count} chapters",
    ]
    for section in result.sections[:3]:
        if section.heading:
            parts.append(section.heading)
    return "\n".join(part for part in parts if part).strip() or None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
