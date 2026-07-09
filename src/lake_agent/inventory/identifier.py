from __future__ import annotations

import mimetypes
from dataclasses import dataclass, replace

from lake_agent.domain.enums import Modality
from lake_agent.domain.models import FileMetadata
from lake_agent.storage.base import ObjectStore


@dataclass(frozen=True, slots=True)
class FormatInfo:
    mime_type: str
    format_name: str
    modality: Modality


_EXTENSIONS: dict[str, FormatInfo] = {
    ".csv": FormatInfo("text/csv", "csv", Modality.TABULAR),
    ".tsv": FormatInfo("text/tab-separated-values", "tsv", Modality.TABULAR),
    ".xls": FormatInfo("application/vnd.ms-excel", "xls", Modality.TABULAR),
    ".xlsx": FormatInfo(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xlsx",
        Modality.TABULAR,
    ),
    ".json": FormatInfo("application/json", "json", Modality.SEMI_STRUCTURE),
    ".jsonl": FormatInfo("application/x-ndjson", "jsonl", Modality.SEMI_STRUCTURE),
    ".xml": FormatInfo("application/xml", "xml", Modality.SEMI_STRUCTURE),
    ".yaml": FormatInfo("application/yaml", "yaml", Modality.SEMI_STRUCTURE),
    ".yml": FormatInfo("application/yaml", "yaml", Modality.SEMI_STRUCTURE),
    ".sql": FormatInfo("application/sql", "sql", Modality.SQL_SCRIPT),
    ".db": FormatInfo("application/vnd.sqlite3", "database", Modality.DATABASE),
    ".sqlite": FormatInfo("application/vnd.sqlite3", "sqlite", Modality.DATABASE),
    ".sqlite3": FormatInfo("application/vnd.sqlite3", "sqlite", Modality.DATABASE),
    ".pdf": FormatInfo("application/pdf", "pdf", Modality.DOCUMENT),
    ".doc": FormatInfo("application/msword", "doc", Modality.DOCUMENT),
    ".docx": FormatInfo(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx",
        Modality.DOCUMENT,
    ),
    ".rtf": FormatInfo("application/rtf", "rtf", Modality.DOCUMENT),
    ".epub": FormatInfo("application/epub+zip", "epub", Modality.EPUB),
    ".ppt": FormatInfo("application/vnd.ms-powerpoint", "ppt", Modality.SLIDE_SHOW),
    ".pptx": FormatInfo(
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "pptx",
        Modality.SLIDE_SHOW,
    ),
    ".txt": FormatInfo("text/plain", "text", Modality.TEXT),
    ".md": FormatInfo("text/markdown", "markdown", Modality.TEXT),
    ".html": FormatInfo("text/html", "html", Modality.WEB),
    ".htm": FormatInfo("text/html", "html", Modality.WEB),
    ".jpg": FormatInfo("image/jpeg", "jpeg", Modality.IMAGE),
    ".jpeg": FormatInfo("image/jpeg", "jpeg", Modality.IMAGE),
    ".png": FormatInfo("image/png", "png", Modality.IMAGE),
    ".gif": FormatInfo("image/gif", "gif", Modality.IMAGE),
    ".webp": FormatInfo("image/webp", "webp", Modality.IMAGE),
    ".tif": FormatInfo("image/tiff", "tiff", Modality.IMAGE),
    ".tiff": FormatInfo("image/tiff", "tiff", Modality.IMAGE),
    ".mp3": FormatInfo("audio/mpeg", "mp3", Modality.AUDIO),
    ".wav": FormatInfo("audio/wav", "wav", Modality.AUDIO),
    ".m4a": FormatInfo("audio/mp4", "m4a", Modality.AUDIO),
    ".flac": FormatInfo("audio/flac", "flac", Modality.AUDIO),
    ".ogg": FormatInfo("audio/ogg", "ogg", Modality.AUDIO),
    ".mp4": FormatInfo("video/mp4", "mp4", Modality.VIDEO),
    ".mov": FormatInfo("video/quicktime", "mov", Modality.VIDEO),
    ".avi": FormatInfo("video/x-msvideo", "avi", Modality.VIDEO),
    ".mkv": FormatInfo("video/x-matroska", "mkv", Modality.VIDEO),
    ".webm": FormatInfo("video/webm", "webm", Modality.VIDEO),
    ".zip": FormatInfo("application/zip", "zip", Modality.ARCHIVE),
    ".gz": FormatInfo("application/gzip", "gzip", Modality.ARCHIVE),
    ".tar": FormatInfo("application/x-tar", "tar", Modality.ARCHIVE),
    ".7z": FormatInfo("application/x-7z-compressed", "7z", Modality.ARCHIVE),
}

_TEXT_MODALITIES = {
    Modality.TEXT,
    Modality.WEB,
    Modality.SQL_SCRIPT,
    Modality.SEMI_STRUCTURE,
    Modality.TABULAR,
}

_CANONICAL_EXTENSIONS: dict[str, str] = {
    ".csv": ".csv",
    ".tsv": ".tsv",
    ".xls": ".xls",
    ".xlsx": ".xlsx",
    ".json": ".json",
    ".jsonl": ".jsonl",
    ".xml": ".xml",
    ".yaml": ".yaml",
    ".yml": ".yaml",
    ".sql": ".sql",
    ".db": ".db",
    ".sqlite": ".sqlite",
    ".sqlite3": ".sqlite",
    ".pdf": ".pdf",
    ".doc": ".doc",
    ".docx": ".docx",
    ".rtf": ".rtf",
    ".epub": ".epub",
    ".ppt": ".ppt",
    ".pptx": ".pptx",
    ".txt": ".txt",
    ".md": ".md",
    ".html": ".html",
    ".htm": ".html",
    ".jpg": ".jpg",
    ".jpeg": ".jpg",
    ".png": ".png",
    ".gif": ".gif",
    ".webp": ".webp",
    ".tif": ".tiff",
    ".tiff": ".tiff",
    ".mp3": ".mp3",
    ".wav": ".wav",
    ".m4a": ".m4a",
    ".flac": ".flac",
    ".ogg": ".ogg",
    ".mp4": ".mp4",
    ".mov": ".mov",
    ".avi": ".avi",
    ".mkv": ".mkv",
    ".webm": ".webm",
    ".zip": ".zip",
    ".gz": ".gz",
    ".tar": ".tar",
    ".7z": ".7z",
}

_FORMAT_TO_EXTENSION = {
    info.format_name: _CANONICAL_EXTENSIONS[extension]
    for extension, info in _EXTENSIONS.items()
    if extension in _CANONICAL_EXTENSIONS
}


class ObjectIdentifier:
    def __init__(self, store: ObjectStore, sample_size: int = 64 * 1024) -> None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        self._store = store
        self._sample_size = sample_size

    def identify(self, obj: FileMetadata) -> FileMetadata:
        sample = self._store.read_range(
            obj,
            offset=0,
            length=min(self._sample_size, obj.size_bytes),
        )
        extension_info = _EXTENSIONS.get(obj.extension)
        signature_info = _identify_signature(sample, obj.extension)
        warnings: list[str] = []

        if signature_info is not None:
            info = signature_info
            confidence = 0.99
            if extension_info and extension_info.format_name != info.format_name:
                compatible = {
                    extension_info.format_name,
                    info.format_name,
                } <= {"database", "sqlite"}
                if not compatible:
                    warnings.append(
                        "File signature and extension disagree: "
                        f"{info.format_name!r} vs {extension_info.format_name!r}."
                    )
        elif extension_info is not None:
            info = extension_info
            confidence = 0.85
        else:
            guessed_mime, _ = mimetypes.guess_type(obj.filename)
            info = FormatInfo(
                guessed_mime or obj.declared_content_type or "application/octet-stream",
                obj.extension.removeprefix(".") or "unknown",
                Modality.UNKNOWN,
            )
            confidence = 0.35 if guessed_mime else 0.1
            warnings.append("Format was not recognized from signature or extension.")

        encoding = _detect_encoding(sample) if info.modality in _TEXT_MODALITIES else None
        return replace(
            obj,
            detected_mime_type=info.mime_type,
            detected_format=info.format_name,
            modality=info.modality,
            encoding=encoding,
            identification_confidence=confidence,
            warnings=tuple(warnings),
        )


def canonical_extension_for_format(format_name: str | None) -> str | None:
    if not format_name:
        return None
    return _FORMAT_TO_EXTENSION.get(format_name)


def _identify_signature(sample: bytes, extension: str) -> FormatInfo | None:
    if sample.startswith(b"%PDF-"):
        return _EXTENSIONS[".pdf"]
    if sample.startswith(b"\x89PNG\r\n\x1a\n"):
        return _EXTENSIONS[".png"]
    if sample.startswith(b"\xff\xd8\xff"):
        return _EXTENSIONS[".jpg"]
    if sample.startswith((b"GIF87a", b"GIF89a")):
        return _EXTENSIONS[".gif"]
    if sample[:4] in {b"II*\x00", b"MM\x00*"}:
        return _EXTENSIONS[".tiff"]
    if sample.startswith(b"SQLite format 3\x00"):
        return _EXTENSIONS[".sqlite"]
    if sample.startswith(b"PK\x03\x04"):
        if extension in {".xlsx", ".docx", ".pptx", ".epub"}:
            return _EXTENSIONS[extension]
        return _EXTENSIONS[".zip"]
    if sample.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        if extension in {".xls", ".doc", ".ppt"}:
            return _EXTENSIONS[extension]
        return FormatInfo("application/x-ole-storage", "ole", Modality.UNKNOWN)
    if sample.startswith(b"ID3") or sample.startswith(b"\xff\xfb"):
        return _EXTENSIONS[".mp3"]
    if sample.startswith(b"fLaC"):
        return _EXTENSIONS[".flac"]
    if sample.startswith(b"OggS"):
        return _EXTENSIONS[".ogg"]
    if sample.startswith(b"RIFF") and sample[8:12] == b"WAVE":
        return _EXTENSIONS[".wav"]
    if sample.startswith(b"RIFF") and sample[8:12] == b"WEBP":
        return _EXTENSIONS[".webp"]
    if sample.startswith(b"RIFF") and sample[8:12] == b"AVI ":
        return _EXTENSIONS[".avi"]
    if len(sample) >= 12 and sample[4:8] == b"ftyp":
        return _EXTENSIONS[".m4a"] if extension == ".m4a" else _EXTENSIONS[".mp4"]
    return None


def _detect_encoding(sample: bytes) -> str | None:
    if not sample:
        return "utf-8"
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    if b"\x00" in sample:
        return None
    try:
        sample.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        try:
            sample.decode("windows-1252")
            return "windows-1252"
        except UnicodeDecodeError:
            return None
