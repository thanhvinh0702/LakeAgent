from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from lake_agent.domain.indexing_models import DocumentIndexResult, DocumentSection

_SUPPORTED_DIRECT_FORMATS = {"pdf", "docx"}
_SUPPORTED_CONVERTIBLE_FORMATS = {"doc", "rtf"}
_SUPPORTED_FORMATS = _SUPPORTED_DIRECT_FORMATS | _SUPPORTED_CONVERTIBLE_FORMATS


@dataclass(frozen=True, slots=True)
class DocumentParseOptions:
    soffice_binary: str = "soffice"


@dataclass(slots=True)
class _PreparedSource:
    path: Path
    warnings: list[str]
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def cleanup(self) -> None:
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


class DeterministicDocumentParser:
    def __init__(
        self,
        options: DocumentParseOptions | None = None,
        *,
        load_document: Callable[[Path], Any] | None = None,
        build_chunker: Callable[[], Any] | None = None,
        prepare_source: Callable[[Path], _PreparedSource] | None = None,
    ) -> None:
        self._options = options or DocumentParseOptions()
        self._load_document = load_document or _default_load_document
        self._build_chunker = build_chunker or _default_build_chunker
        self._prepare_source = prepare_source or self._default_prepare_source

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> DocumentIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported deterministic document format: {extension}")

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
        finally:
            prepared.cleanup()

        warnings = list(prepared.warnings)
        if not sections:
            warnings.append("The file did not contain any readable document chunks.")

        result = DocumentIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format=extension,
            sections=sections,
            parse_warnings=warnings,
        )
        result.file_search_text = _build_file_search_text(result)
        return result

    def _build_sections(
        self,
        chunks: list[Any],
        *,
        chunker: Any,
        source_id: str,
    ) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        for chunk_index, chunk in enumerate(chunks, start=1):
            content = getattr(chunk, "text", "") or ""
            content = content.strip()
            if not content:
                continue
            headings = list(getattr(getattr(chunk, "meta", None), "headings", []) or [])
            heading = " > ".join(part.strip() for part in headings if part and part.strip()) or None
            page_start, page_end = _extract_page_range(chunk)
            search_text = chunker.contextualize(chunk=chunk).strip()
            sections.append(
                DocumentSection(
                    section_id=_stable_id(
                        f"{source_id}:{chunk_index}:{heading or ''}:{content[:80]}",
                        prefix="section",
                    ),
                    chunk_index=chunk_index,
                    heading=heading,
                    content=content,
                    page_start=page_start,
                    page_end=page_end,
                    char_count=len(content),
                    search_text=search_text or content,
                )
            )
        return sections

    def _default_prepare_source(self, path: Path) -> _PreparedSource:
        extension = path.suffix.lower().removeprefix(".")
        if extension in _SUPPORTED_DIRECT_FORMATS:
            return _PreparedSource(path=path, warnings=[])

        if extension not in _SUPPORTED_CONVERTIBLE_FORMATS:
            raise ValueError(f"Unsupported deterministic document format: {extension}")

        soffice_binary = shutil.which(self._options.soffice_binary)
        if not soffice_binary:
            raise RuntimeError(
                f"Legacy document format '.{extension}' requires LibreOffice for conversion, "
                f"but '{self._options.soffice_binary}' was not found on PATH."
            )

        temp_dir = tempfile.TemporaryDirectory(prefix="lake_agent_docling_")
        temp_path = Path(temp_dir.name)
        subprocess.run(
            [
                soffice_binary,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(temp_path),
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        converted_path = temp_path / f"{path.stem}.docx"
        if not converted_path.exists():
            raise RuntimeError(
                f"LibreOffice conversion did not produce expected file: {converted_path.name}"
            )
        return _PreparedSource(
            path=converted_path,
            warnings=[
                f"Converted legacy .{extension} file to .docx before Docling parsing."
            ],
            temp_dir=temp_dir,
        )


def _default_load_document(path: Path) -> Any:
    from docling.document_converter import DocumentConverter

    return DocumentConverter().convert(source=str(path)).document


def _default_build_chunker() -> Any:
    from docling.chunking import HierarchicalChunker

    return HierarchicalChunker()


def _extract_page_range(chunk: Any) -> tuple[int | None, int | None]:
    meta = getattr(chunk, "meta", None)
    if meta is None:
        return None, None

    page_numbers: list[int] = []
    for doc_item in getattr(meta, "doc_items", []) or []:
        for prov in getattr(doc_item, "prov", []) or []:
            page_no = getattr(prov, "page_no", None)
            if isinstance(page_no, int):
                page_numbers.append(page_no)
    if not page_numbers:
        return None, None
    return min(page_numbers), max(page_numbers)


def _build_file_search_text(result: DocumentIndexResult) -> str | None:
    parts = [result.filename, result.relative_path]
    for section in result.sections[:3]:
        if section.heading:
            parts.append(section.heading)
        if section.page_start is not None:
            if section.page_end is not None and section.page_end != section.page_start:
                parts.append(f"pages {section.page_start}-{section.page_end}")
            else:
                parts.append(f"page {section.page_start}")
    return "\n".join(part for part in parts if part).strip() or None


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
