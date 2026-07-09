from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from lake_agent.domain.indexing_models import VideoIndexResult, VideoSection
from lake_agent.indexing.audio.deterministic import AudioTranscriptParser, format_timestamp
from lake_agent.indexing.text.chunking import build_basic_search_text, normalize_text
from lake_agent.indexing.video.vlm import VideoFrameVLMCaptioner

_SUPPORTED_FORMATS = {"mp4", "mkv", "mov", "avi", "webm"}


@dataclass(frozen=True, slots=True)
class VideoParseOptions:
    max_frames: int = 8


@dataclass(frozen=True, slots=True)
class VideoProbe:
    duration_seconds: float | None
    width: int | None
    height: int | None
    fps: float | None
    video_codec: str | None
    audio_codec: str | None
    has_audio: bool
    warnings: tuple[str, ...] = ()


class DeterministicVideoParser:
    def __init__(
        self,
        *,
        audio_parser: AudioTranscriptParser | None = None,
        vlm_captioner: VideoFrameVLMCaptioner | None = None,
        options: VideoParseOptions | None = None,
    ) -> None:
        self._audio_parser = audio_parser
        self._vlm_captioner = vlm_captioner
        self._options = options or VideoParseOptions()
        if self._options.max_frames < 0:
            raise ValueError("max_frames must not be negative")

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> VideoIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported video format: {extension}")

        normalized_relative_path = relative_path or path.name
        normalized_relative_path = PurePosixPath(
            normalized_relative_path.replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        probe = probe_video(path)
        warnings = list(probe.warnings)
        sections: list[VideoSection] = []
        asr_model_name = None
        asr_cost_usd = None
        asr_usage: dict[str, Any] = {}
        sampled_frame_count = 0
        vl_model_name = None

        with tempfile.TemporaryDirectory(prefix="lakeagent_video_") as temp_dir:
            temp_root = Path(temp_dir)
            if probe.has_audio and self._audio_parser is not None:
                audio_path = temp_root / f"{path.stem}.wav"
                extract_audio_to_wav(path, audio_path)
                audio_result = self._audio_parser.parse_file(
                    audio_path,
                    relative_path=normalized_relative_path,
                    source_id=normalized_source_id,
                )
                asr_model_name = audio_result.asr_model_name
                asr_cost_usd = audio_result.asr_cost_usd
                asr_usage = audio_result.asr_usage
                warnings.extend(audio_result.parse_warnings)
                sections.extend(
                    _audio_sections_to_video_sections(
                        audio_result.sections,
                        source_id=normalized_source_id,
                    )
                )
            elif probe.has_audio:
                warnings.append("Audio stream was present but audio transcription was disabled.")
            else:
                warnings.append("No audio stream was found.")

            if self._vlm_captioner is not None and self._options.max_frames > 0:
                frame_infos = extract_sampled_frames(
                    path,
                    temp_root,
                    duration_seconds=probe.duration_seconds or 0.0,
                    max_frames=self._options.max_frames,
                )
                sampled_frame_count = len(frame_infos)
                vl_model_name = self._vlm_captioner.model_name
                for chunk_index, (timestamp_seconds, frame_path) in enumerate(
                    frame_infos,
                    start=len(sections) + 1,
                ):
                    caption = self._vlm_captioner.caption_frame(
                        frame_path,
                        timestamp_seconds=timestamp_seconds,
                        frame_index=chunk_index,
                        video_filename=path.name,
                    )
                    content = normalize_text(caption.content)
                    if not content:
                        continue
                    sections.append(
                        VideoSection(
                            section_id=_stable_id(
                                f"{normalized_source_id}:frame:{timestamp_seconds:.3f}",
                                prefix="vidsec",
                            ),
                            section_type="frame_caption",
                            chunk_index=chunk_index,
                            timestamp_seconds=timestamp_seconds,
                            frame_index=caption.frame_index,
                            content=content,
                            char_count=len(content),
                            search_text=_build_frame_search_text(
                                timestamp_seconds=timestamp_seconds,
                                content=content,
                            ),
                            warnings=caption.warnings,
                        )
                    )

        if not sections:
            warnings.append("The video did not produce any searchable sections.")

        result = VideoIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format=extension,  # type: ignore[arg-type]
            sections=sections,
            duration_seconds=probe.duration_seconds,
            width=probe.width,
            height=probe.height,
            fps=probe.fps,
            video_codec=probe.video_codec,
            audio_codec=probe.audio_codec,
            has_audio=probe.has_audio,
            sampled_frame_count=sampled_frame_count,
            asr_model_name=asr_model_name,
            asr_cost_usd=asr_cost_usd,
            asr_usage=asr_usage,
            vl_model_name=vl_model_name,
            parse_warnings=warnings,
        )
        result.file_search_text = _build_file_search_text(result)
        return result


def probe_video(path: Path) -> VideoProbe:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        os.fspath(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(result.stdout)
    except FileNotFoundError as exc:
        raise RuntimeError("Video indexing requires ffprobe on PATH.") from exc
    except Exception as exc:
        return VideoProbe(
            duration_seconds=None,
            width=None,
            height=None,
            fps=None,
            video_codec=None,
            audio_codec=None,
            has_audio=False,
            warnings=(f"ffprobe failed: {exc}",),
        )

    streams = payload.get("streams") or []
    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        {},
    )
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    duration = _optional_float(video_stream.get("duration"))
    if duration is None:
        duration = _optional_float((payload.get("format") or {}).get("duration"))

    return VideoProbe(
        duration_seconds=duration,
        width=_optional_int(video_stream.get("width")),
        height=_optional_int(video_stream.get("height")),
        fps=_parse_rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate")),
        video_codec=video_stream.get("codec_name"),
        audio_codec=audio_stream.get("codec_name") if isinstance(audio_stream, dict) else None,
        has_audio=isinstance(audio_stream, dict),
    )


def extract_audio_to_wav(source: Path, target: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        os.fspath(source),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        os.fspath(target),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("Video indexing requires ffmpeg on PATH.") from exc


def extract_sampled_frames(
    source: Path,
    output_dir: Path,
    *,
    duration_seconds: float,
    max_frames: int,
) -> list[tuple[float, Path]]:
    timestamps = build_frame_timestamps(
        duration_seconds=duration_seconds,
        max_frames=max_frames,
    )
    frame_infos: list[tuple[float, Path]] = []
    for index, timestamp_seconds in enumerate(timestamps, start=1):
        frame_path = output_dir / f"{source.stem}_frame_{index:04d}.jpg"
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{timestamp_seconds:.3f}",
            "-i",
            os.fspath(source),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            os.fspath(frame_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        if frame_path.exists():
            frame_infos.append((timestamp_seconds, frame_path))
    return frame_infos


def build_frame_timestamps(
    *,
    duration_seconds: float,
    max_frames: int,
) -> list[float]:
    if max_frames <= 0 or duration_seconds <= 0:
        return []
    frame_count = min(max_frames, max(1, int(duration_seconds)))
    step = duration_seconds / frame_count
    last_safe_timestamp = max(duration_seconds - 0.001, 0.0)
    return [
        min(last_safe_timestamp, (index + 0.5) * step)
        for index in range(frame_count)
    ]


def _audio_sections_to_video_sections(
    audio_sections: list[Any],
    *,
    source_id: str,
) -> list[VideoSection]:
    sections: list[VideoSection] = []
    for index, section in enumerate(audio_sections, start=1):
        content = section.content.strip()
        sections.append(
            VideoSection(
                section_id=_stable_id(
                    f"{source_id}:transcript:{index}:{section.start_seconds}:{section.end_seconds}",
                    prefix="vidsec",
                ),
                section_type="transcript_chunk",
                chunk_index=index,
                content=content,
                start_seconds=section.start_seconds,
                end_seconds=section.end_seconds,
                char_count=len(content),
                search_text=_build_transcript_search_text(
                    content=content,
                    start_seconds=section.start_seconds,
                    end_seconds=section.end_seconds,
                ),
                warnings=list(section.warnings),
            )
        )
    return sections


def _build_transcript_search_text(
    *,
    content: str,
    start_seconds: float | None,
    end_seconds: float | None,
) -> str:
    heading = "Video transcript"
    if start_seconds is not None and end_seconds is not None:
        heading = f"Video transcript {format_timestamp(start_seconds)}-{format_timestamp(end_seconds)}"
    return build_basic_search_text(heading, content)


def _build_frame_search_text(
    *,
    timestamp_seconds: float,
    content: str,
) -> str:
    heading = f"Video frame {format_timestamp(timestamp_seconds)}"
    return build_basic_search_text(heading, content)


def _build_file_search_text(result: VideoIndexResult) -> str | None:
    parts = [result.filename, result.relative_path, "Video"]
    if result.duration_seconds is not None:
        parts.append(f"Duration: {result.duration_seconds:.1f} seconds")
    if result.width and result.height:
        parts.append(f"Resolution: {result.width}x{result.height}")
    for section in result.sections[:5]:
        parts.append(section.search_text or section.content)
    return "\n".join(part for part in parts if part).strip() or None


def _stable_id(value: str, *, prefix: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rate(value: Any) -> float | None:
    if not value:
        return None
    text = str(value)
    if "/" not in text:
        return _optional_float(text)
    numerator, denominator = text.split("/", 1)
    try:
        denominator_float = float(denominator)
        if denominator_float == 0:
            return None
        return float(numerator) / denominator_float
    except (TypeError, ValueError):
        return None
