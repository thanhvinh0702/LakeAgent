from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from lake_agent.domain.indexing_models.json import JsonIndexResult, JsonSection

_SUPPORTED_FORMATS = {"json", "jsonl", "ndjson"}
_SCALAR_TYPES = (str, int, float, bool, type(None))


@dataclass(frozen=True, slots=True)
class JsonParseOptions:
    max_chars_per_section: int = 2400
    max_scalar_chars: int = 1000
    max_entries: int | None = None
    include_container_entries: bool = True


@dataclass(frozen=True, slots=True)
class _FlattenedEntry:
    path: str
    value_type: str
    value: str
    depth: int

    @property
    def text(self) -> str:
        return f"{self.path}: {self.value}"


class DeterministicJsonParser:
    def __init__(self, options: JsonParseOptions | None = None) -> None:
        self._options = options or JsonParseOptions()
        if self._options.max_chars_per_section <= 0:
            raise ValueError("max_chars_per_section must be positive")
        if self._options.max_scalar_chars <= 0:
            raise ValueError("max_scalar_chars must be positive")
        if self._options.max_entries is not None and self._options.max_entries <= 0:
            raise ValueError("max_entries must be positive when provided")

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> JsonIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported deterministic JSON format: {extension}")

        normalized_relative_path = relative_path or path.name
        normalized_relative_path = PurePosixPath(
            normalized_relative_path.replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        warnings: list[str] = []
        payload = self._load_payload(path, extension, warnings)
        top_level_type = _json_type_name(payload)
        entries = list(self._flatten(payload))

        if self._options.max_entries is not None and len(entries) > self._options.max_entries:
            entries = entries[: self._options.max_entries]
            warnings.append(
                f"Flattened JSON entries were truncated to {self._options.max_entries}."
            )

        if not entries:
            warnings.append("The file did not contain any readable JSON values.")

        sections = self._build_sections(entries, normalized_source_id)
        result = JsonIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format=extension,
            sections=sections,
            parse_warnings=warnings,
            top_level_type=top_level_type,
            entry_count=len(entries),
            max_depth=max((entry.depth for entry in entries), default=0),
        )
        result.file_search_text = _build_file_search_text(result)
        return result

    def _load_payload(self, path: Path, extension: str, warnings: list[str]) -> Any:
        if extension == "json":
            return json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))

        records: list[Any] = []
        with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON record on line {line_number}: {exc.msg}"
                    ) from exc
        if not records:
            warnings.append("JSON lines file did not contain any non-empty records.")
        return records

    def _flatten(self, value: Any) -> Iterable[_FlattenedEntry]:
        yield from self._flatten_value(value, path="$", depth=0)

    def _flatten_value(
        self,
        value: Any,
        *,
        path: str,
        depth: int,
    ) -> Iterable[_FlattenedEntry]:
        if isinstance(value, dict):
            if self._options.include_container_entries:
                yield _FlattenedEntry(
                    path=path,
                    value_type="object",
                    value=f"object with {len(value)} key(s)",
                    depth=depth,
                )
            for key in sorted(value):
                yield from self._flatten_value(
                    value[key],
                    path=f"{path}.{_escape_key(str(key))}",
                    depth=depth + 1,
                )
            return

        if isinstance(value, list):
            if self._options.include_container_entries:
                yield _FlattenedEntry(
                    path=path,
                    value_type="array",
                    value=f"array with {len(value)} item(s)",
                    depth=depth,
                )
            for index, item in enumerate(value):
                yield from self._flatten_value(
                    item,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                )
            return

        if not isinstance(value, _SCALAR_TYPES):
            value = str(value)

        yield _FlattenedEntry(
            path=path,
            value_type=_json_type_name(value),
            value=_format_scalar(value, self._options.max_scalar_chars),
            depth=depth,
        )

    def _build_sections(
        self,
        entries: list[_FlattenedEntry],
        source_id: str,
    ) -> list[JsonSection]:
        sections: list[JsonSection] = []
        current: list[_FlattenedEntry] = []
        current_chars = 0

        for entry in entries:
            entry_text = entry.text
            entry_chars = len(entry_text) + 1
            if current and current_chars + entry_chars > self._options.max_chars_per_section:
                sections.append(_build_section(source_id, len(sections) + 1, current))
                current = []
                current_chars = 0
            current.append(entry)
            current_chars += entry_chars

        if current:
            sections.append(_build_section(source_id, len(sections) + 1, current))
        return sections


def _build_section(
    source_id: str,
    chunk_index: int,
    entries: list[_FlattenedEntry],
) -> JsonSection:
    content = "\n".join(entry.text for entry in entries).strip()
    path_start = entries[0].path if entries else None
    path_end = entries[-1].path if entries else None
    return JsonSection(
        section_id=_stable_id(
            f"{source_id}:{chunk_index}:{path_start or ''}:{path_end or ''}",
            prefix="section",
        ),
        chunk_index=chunk_index,
        path_start=path_start,
        path_end=path_end,
        entry_count=len(entries),
        content=content,
        char_count=len(content),
        search_text=content,
    )


def _format_scalar(value: Any, max_chars: int) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    else:
        text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}... [truncated]"


def _escape_key(key: str) -> str:
    if key.isidentifier():
        return key
    escaped = key.replace("\\", "\\\\").replace("'", "\\'")
    return f"['{escaped}']"


def _json_type_name(value: Any) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _build_file_search_text(result: JsonIndexResult) -> str | None:
    parts = [
        result.filename,
        result.relative_path,
        f"top-level type: {result.top_level_type}" if result.top_level_type else None,
        f"flattened entries: {result.entry_count}",
        f"max depth: {result.max_depth}",
    ]
    for section in result.sections[:3]:
        if section.path_start and section.path_end:
            parts.append(f"{section.path_start} to {section.path_end}")
    return "\n".join(part for part in parts if part).strip() or None


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
