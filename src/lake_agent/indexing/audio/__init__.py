"""Audio indexing via local normalization and ASR transcripts."""

from lake_agent.indexing.audio.deterministic import (
    AudioParseOptions,
    AudioProbe,
    AudioTranscriptParser,
    TimedTranscript,
    convert_to_wav,
    format_timestamp,
    probe_audio,
)
from lake_agent.indexing.audio.service import (
    AudioIndexingError,
    AudioIndexingProgress,
    AudioIndexingService,
)
from lake_agent.indexing.audio.transcriber import (
    AudioTranscription,
    OpenRouterAudioTranscriber,
)
from lake_agent.indexing.audio.vector_store import (
    add_audio_result,
    add_audio_results,
    build_audio_documents,
    build_batch_audio_documents,
    build_openai_embeddings,
    build_pgvector_store,
)

__all__ = [
    "AudioIndexingError",
    "AudioIndexingProgress",
    "AudioIndexingService",
    "AudioParseOptions",
    "AudioProbe",
    "AudioTranscriptParser",
    "AudioTranscription",
    "OpenRouterAudioTranscriber",
    "TimedTranscript",
    "add_audio_result",
    "add_audio_results",
    "build_audio_documents",
    "build_batch_audio_documents",
    "build_openai_embeddings",
    "build_pgvector_store",
    "convert_to_wav",
    "format_timestamp",
    "probe_audio",
]
