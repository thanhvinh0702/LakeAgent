from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


AudioFormat = Literal["mp3", "wav", "m4a", "flac", "ogg"]


@dataclass(slots=True)
class AudioSection:
    section_id: str
    chunk_index: int
    content: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    char_count: int = 0
    search_text: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AudioIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: AudioFormat
    sections: list[AudioSection] = field(default_factory=list)
    duration_seconds: float | None = None
    codec_name: str | None = None
    sample_rate: int | None = None
    channels: int | None = None
    transcript_language: str | None = None
    transcript_text: str | None = None
    asr_model_name: str | None = None
    asr_cost_usd: float | None = None
    asr_usage: dict[str, Any] = field(default_factory=dict)
    parser_version: str = "asr_wav16k_v1"
    parse_warnings: list[str] = field(default_factory=list)
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None
