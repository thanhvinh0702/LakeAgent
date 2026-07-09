from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lake_agent.config import ASRSettings


@dataclass(frozen=True, slots=True)
class AudioTranscription:
    text: str
    model_name: str
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    warnings: list[str] = field(default_factory=list)


class OpenRouterAudioTranscriber:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model_name: str,
        fallback_model_name: str | None = None,
        prompt: str = "",
        language: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._model_name = model_name
        self._fallback_model_name = fallback_model_name
        self._prompt = prompt
        self._language = language
        self._client = _build_client(api_key=api_key, base_url=base_url)

    @classmethod
    def from_env(
        cls,
        *,
        prompt: str = "",
        language: str | None = None,
    ) -> "OpenRouterAudioTranscriber":
        settings = ASRSettings.from_env()
        return cls(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model_name=settings.model_name,
            fallback_model_name=settings.fallback_model_name,
            prompt=prompt,
            language=language,
        )

    def transcribe(self, audio_path: Path) -> AudioTranscription:
        warnings: list[str] = []
        try:
            return self._transcribe_with_model(audio_path, self._model_name, warnings=warnings)
        except Exception as exc:
            if not self._fallback_model_name:
                raise
            warnings.append(
                f"Primary ASR model {self._model_name!r} failed: {exc}"
            )
            return self._transcribe_with_model(
                audio_path,
                self._fallback_model_name,
                warnings=warnings,
            )

    def _transcribe_with_model(
        self,
        audio_path: Path,
        model_name: str,
        *,
        warnings: list[str],
    ) -> AudioTranscription:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "file": audio_path.open("rb"),
        }
        if self._prompt:
            kwargs["prompt"] = self._prompt
        if self._language:
            kwargs["language"] = self._language

        try:
            result = self._client.audio.transcriptions.create(**kwargs)
        finally:
            file_obj = kwargs.get("file")
            if file_obj is not None:
                file_obj.close()

        usage = _extract_usage(result)
        return AudioTranscription(
            text=_extract_text(result),
            model_name=model_name,
            usage=usage,
            cost_usd=_extract_cost(usage),
            warnings=list(warnings),
        )


def _build_client(*, api_key: str, base_url: str) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "ASR transcription requires openai. Install the project dependencies first."
        ) from exc
    return OpenAI(api_key=api_key, base_url=base_url)


def _extract_text(result: Any) -> str:
    text = getattr(result, "text", None)
    if isinstance(text, str):
        return text
    if isinstance(result, dict):
        text = result.get("text")
        if isinstance(text, str):
            return text
    return str(result)


def _extract_usage(result: Any) -> dict[str, Any]:
    usage = getattr(result, "usage", None)
    if usage is None and isinstance(result, dict):
        usage = result.get("usage")
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    dump = getattr(usage, "model_dump", None)
    if callable(dump):
        return dump()
    return {}


def _extract_cost(usage: dict[str, Any]) -> float | None:
    value = usage.get("cost")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
