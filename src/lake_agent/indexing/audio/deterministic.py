from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from lake_agent.domain.indexing_models import AudioIndexResult, AudioSection
from lake_agent.indexing.audio.transcriber import (
    AudioTranscription,
    OpenRouterAudioTranscriber,
)
from lake_agent.indexing.text.chunking import (
    build_basic_search_text,
    chunk_plain_text,
    normalize_text,
)

_SUPPORTED_FORMATS = {"mp3", "wav", "m4a", "flac", "ogg"}


@dataclass(frozen=True, slots=True)
class AudioParseOptions:
    max_chars_per_chunk: int = 2400
    min_chunk_chars: int = 400
    max_chunk_seconds: int = 600
    chunk_overlap_seconds: int = 8


@dataclass(frozen=True, slots=True)
class AudioProbe:
    duration_seconds: float | None
    codec_name: str | None
    sample_rate: int | None
    channels: int | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TimedTranscript:
    text: str
    start_seconds: float | None
    end_seconds: float | None
    model_name: str
    usage: dict[str, Any]
    cost_usd: float | None
    warnings: tuple[str, ...] = ()


class AudioTranscriptParser:
    def __init__(
        self,
        transcriber: OpenRouterAudioTranscriber | None = None,
        *,
        options: AudioParseOptions | None = None,
        transcript_dir: str | Path | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._options = options or AudioParseOptions()
        if self._options.max_chunk_seconds <= 0:
            raise ValueError("max_chunk_seconds must be positive")
        if self._options.chunk_overlap_seconds < 0:
            raise ValueError("chunk_overlap_seconds must not be negative")
        if self._options.chunk_overlap_seconds >= self._options.max_chunk_seconds:
            raise ValueError("chunk_overlap_seconds must be smaller than max_chunk_seconds")
        self._transcript_dir = (
            Path(transcript_dir).expanduser().resolve() if transcript_dir else None
        )

    def parse_file(
        self,
        file_path: str | Path,
        *,
        relative_path: str | None = None,
        source_id: str | None = None,
    ) -> AudioIndexResult:
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(path)

        extension = path.suffix.lower().removeprefix(".")
        if extension not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported audio format: {extension}")

        normalized_relative_path = relative_path or path.name
        normalized_relative_path = PurePosixPath(
            normalized_relative_path.replace("\\", "/")
        ).as_posix()
        normalized_source_id = source_id or _stable_id(
            normalized_relative_path,
            prefix="source",
        )

        probe = probe_audio(path)
        imported = self._load_imported_transcript(
            path,
            relative_path=normalized_relative_path,
        )
        transcripts = imported
        if transcripts is None:
            if self._transcriber is None:
                raise ValueError(
                    "No imported transcript was found and no ASR transcriber is configured."
                )
            transcripts = self._transcribe_file(path, probe)

        full_text = normalize_text(
            "\n\n".join(part.text.strip() for part in transcripts if part.text.strip())
        )
        warnings = list(probe.warnings)
        for part in transcripts:
            warnings.extend(part.warnings)

        sections = self._build_sections(
            transcripts,
            source_id=normalized_source_id,
        )
        if not sections:
            warnings.append("The audio file did not produce any transcript sections.")

        asr_models = [part.model_name for part in transcripts if part.model_name]
        usage = _merge_usage(transcripts)
        result = AudioIndexResult(
            source_id=normalized_source_id,
            relative_path=normalized_relative_path,
            filename=path.name,
            file_format=extension,  # type: ignore[arg-type]
            sections=sections,
            duration_seconds=probe.duration_seconds,
            codec_name=probe.codec_name,
            sample_rate=probe.sample_rate,
            channels=probe.channels,
            transcript_text=full_text or None,
            asr_model_name=", ".join(dict.fromkeys(asr_models)) or None,
            asr_cost_usd=_sum_cost(transcripts),
            asr_usage=usage,
            parse_warnings=warnings,
        )
        result.file_search_text = _build_file_search_text(result)
        return result

    def _transcribe_file(self, path: Path, probe: AudioProbe) -> list[TimedTranscript]:
        duration = probe.duration_seconds or 0.0
        windows = _build_windows(
            duration,
            max_chunk_seconds=self._options.max_chunk_seconds,
            overlap_seconds=self._options.chunk_overlap_seconds,
        )
        transcripts: list[TimedTranscript] = []

        with tempfile.TemporaryDirectory(prefix="lakeagent_audio_") as temp_dir:
            temp_root = Path(temp_dir)
            for index, (start_seconds, end_seconds) in enumerate(windows, start=1):
                wav_path = temp_root / f"{path.stem}_{index:04d}.wav"
                convert_to_wav(
                    path,
                    wav_path,
                    start_seconds=start_seconds,
                    duration_seconds=end_seconds - start_seconds,
                )
                transcription = self._transcriber.transcribe(wav_path)
                transcripts.append(
                    _to_timed_transcript(
                        transcription,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                    )
                )
        return transcripts

    def _load_imported_transcript(
        self,
        audio_path: Path,
        *,
        relative_path: str,
    ) -> list[TimedTranscript] | None:
        if self._transcript_dir is None:
            return None
        candidates = [
            self._transcript_dir / f"{relative_path}.json",
            self._transcript_dir / f"{audio_path.name}.json",
            self._transcript_dir / f"{audio_path.stem}.json",
        ]
        transcript_path = next((path for path in candidates if path.exists()), None)
        if transcript_path is None:
            return None

        data = json.loads(transcript_path.read_text(encoding="utf-8-sig"))
        expected_sha1 = data.get("source_sha1")
        warnings: list[str] = [f"Transcript imported from {transcript_path}."]
        if expected_sha1:
            actual_sha1 = _sha1_file(audio_path)
            if expected_sha1 != actual_sha1:
                raise ValueError(
                    f"Imported transcript checksum mismatch for {relative_path}: "
                    f"{expected_sha1} != {actual_sha1}"
                )

        model_name = str(data.get("model") or "imported_transcript")
        segments = data.get("segments")
        if isinstance(segments, list) and segments:
            imported_segments: list[TimedTranscript] = []
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                text = normalize_text(str(segment.get("text") or ""))
                if not text:
                    continue
                imported_segments.append(
                    TimedTranscript(
                        text=text,
                        start_seconds=_optional_float(segment.get("start_seconds")),
                        end_seconds=_optional_float(segment.get("end_seconds")),
                        model_name=model_name,
                        usage={},
                        cost_usd=0.0,
                        warnings=tuple(warnings),
                    )
                )
            return imported_segments

        full_text = normalize_text(str(data.get("full_text") or data.get("text") or ""))
        if not full_text:
            raise ValueError(f"Imported transcript has no text: {transcript_path}")
        return [
            TimedTranscript(
                text=full_text,
                start_seconds=0.0,
                end_seconds=_optional_float(data.get("duration_seconds")),
                model_name=model_name,
                usage={},
                cost_usd=0.0,
                warnings=tuple(warnings),
            )
        ]

    def _build_sections(
        self,
        transcripts: list[TimedTranscript],
        *,
        source_id: str,
    ) -> list[AudioSection]:
        sections: list[AudioSection] = []
        chunk_index = 0
        for transcript in transcripts:
            chunks = chunk_plain_text(
                normalize_text(transcript.text),
                max_chars=self._options.max_chars_per_chunk,
                min_chars=self._options.min_chunk_chars,
            )
            for local_index, chunk in enumerate(chunks, start=1):
                chunk_index += 1
                start_seconds, end_seconds = _estimate_chunk_window(
                    transcript,
                    local_index=local_index,
                    local_count=len(chunks),
                )
                content = chunk.content.strip()
                sections.append(
                    AudioSection(
                        section_id=_stable_id(
                            f"{source_id}:audio:{chunk_index}:{start_seconds}:{end_seconds}",
                            prefix="section",
                        ),
                        chunk_index=chunk_index,
                        content=content,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        char_count=len(content),
                        search_text=_build_section_search_text(
                            content=content,
                            start_seconds=start_seconds,
                            end_seconds=end_seconds,
                        ),
                    )
                )
        return sections


def probe_audio(path: Path) -> AudioProbe:
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
        raise RuntimeError("Audio indexing requires ffprobe on PATH.") from exc
    except Exception as exc:
        return AudioProbe(
            duration_seconds=None,
            codec_name=None,
            sample_rate=None,
            channels=None,
            warnings=(f"ffprobe failed: {exc}",),
        )

    streams = payload.get("streams") or []
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )
    if not isinstance(audio_stream, dict):
        return AudioProbe(
            duration_seconds=_optional_float((payload.get("format") or {}).get("duration")),
            codec_name=None,
            sample_rate=None,
            channels=None,
            warnings=("No audio stream was reported by ffprobe.",),
        )

    duration = _optional_float(audio_stream.get("duration"))
    if duration is None:
        duration = _optional_float((payload.get("format") or {}).get("duration"))
    return AudioProbe(
        duration_seconds=duration,
        codec_name=audio_stream.get("codec_name"),
        sample_rate=_optional_int(audio_stream.get("sample_rate")),
        channels=_optional_int(audio_stream.get("channels")),
    )


def convert_to_wav(
    source: Path,
    target: Path,
    *,
    start_seconds: float,
    duration_seconds: float,
) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-i",
        os.fspath(source),
        "-t",
        f"{max(duration_seconds, 0.001):.3f}",
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
        raise RuntimeError("Audio indexing requires ffmpeg on PATH.") from exc


def _to_timed_transcript(
    transcription: AudioTranscription,
    *,
    start_seconds: float,
    end_seconds: float,
) -> TimedTranscript:
    return TimedTranscript(
        text=normalize_text(transcription.text),
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        model_name=transcription.model_name,
        usage=transcription.usage,
        cost_usd=transcription.cost_usd,
        warnings=tuple(transcription.warnings),
    )


def _build_windows(
    duration_seconds: float,
    *,
    max_chunk_seconds: int,
    overlap_seconds: int,
) -> list[tuple[float, float]]:
    if duration_seconds <= 0 or duration_seconds <= max_chunk_seconds:
        return [(0.0, max(duration_seconds, 0.001))]

    windows: list[tuple[float, float]] = []
    step = max_chunk_seconds - overlap_seconds
    start = 0.0
    while start < duration_seconds:
        end = min(duration_seconds, start + max_chunk_seconds)
        windows.append((start, end))
        if end >= duration_seconds:
            break
        start += step
    return windows


def _estimate_chunk_window(
    transcript: TimedTranscript,
    *,
    local_index: int,
    local_count: int,
) -> tuple[float | None, float | None]:
    if transcript.start_seconds is None or transcript.end_seconds is None:
        return transcript.start_seconds, transcript.end_seconds
    if local_count <= 1:
        return transcript.start_seconds, transcript.end_seconds
    total = transcript.end_seconds - transcript.start_seconds
    start = transcript.start_seconds + total * ((local_index - 1) / local_count)
    end = transcript.start_seconds + total * (local_index / local_count)
    return start, end


def _build_section_search_text(
    *,
    content: str,
    start_seconds: float | None,
    end_seconds: float | None,
) -> str:
    heading = None
    if start_seconds is not None and end_seconds is not None:
        heading = f"Audio transcript {format_timestamp(start_seconds)}-{format_timestamp(end_seconds)}"
    return build_basic_search_text(heading, content)


def _build_file_search_text(result: AudioIndexResult) -> str | None:
    parts = [result.filename, result.relative_path, "Audio transcript"]
    if result.duration_seconds is not None:
        parts.append(f"Duration: {result.duration_seconds:.1f} seconds")
    if result.transcript_text:
        parts.append(result.transcript_text[:1600])
    return "\n".join(part for part in parts if part).strip() or None


def _merge_usage(transcripts: list[TimedTranscript]) -> dict[str, Any]:
    if not transcripts:
        return {}
    merged: dict[str, Any] = {}
    total_cost = _sum_cost(transcripts)
    if total_cost is not None:
        merged["cost"] = total_cost
    total_seconds = 0.0
    has_seconds = False
    for transcript in transcripts:
        seconds = _optional_float(transcript.usage.get("seconds"))
        if seconds is not None:
            total_seconds += seconds
            has_seconds = True
    if has_seconds:
        merged["seconds"] = total_seconds
    merged["chunks"] = [part.usage for part in transcripts if part.usage]
    return merged


def _sum_cost(transcripts: list[TimedTranscript]) -> float | None:
    costs = [part.cost_usd for part in transcripts if part.cost_usd is not None]
    if not costs:
        return None
    return float(sum(costs))


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def format_timestamp(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, sec = divmod(total_seconds, 60)
    hours, minute = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minute:02d}:{sec:02d}"
    return f"{minute:02d}:{sec:02d}"
