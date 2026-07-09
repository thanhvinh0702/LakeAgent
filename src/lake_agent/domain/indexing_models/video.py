from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


VideoFormat = Literal["mp4", "mkv", "mov", "avi", "webm"]


@dataclass(slots=True)
class VideoSection:
    section_id: str
    section_type: str
    chunk_index: int
    content: str
    timestamp_seconds: float | None = None
    start_seconds: float | None = None
    end_seconds: float | None = None
    frame_index: int | None = None
    char_count: int = 0
    search_text: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VideoIndexResult:
    source_id: str
    relative_path: str
    filename: str
    file_format: VideoFormat
    sections: list[VideoSection] = field(default_factory=list)
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    has_audio: bool = False
    sampled_frame_count: int = 0
    asr_model_name: str | None = None
    asr_cost_usd: float | None = None
    asr_usage: dict[str, Any] = field(default_factory=dict)
    vl_model_name: str | None = None
    parser_version: str = "video_audio_frame_v1"
    parse_warnings: list[str] = field(default_factory=list)
    file_summary: str | None = None
    file_keywords: list[str] = field(default_factory=list)
    file_search_text: str | None = None
