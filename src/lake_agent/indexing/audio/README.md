# Audio Indexing

Audio indexing transcribes audio files, chunks the transcript, stores it in Postgres,
and writes embeddings to the `audio_index` vector table.

## Environment

Set these in `.env`:

```env
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

ASR_MODEL_NAME=qwen/qwen3-asr-flash-2026-02-10
ASR_FALLBACK_MODEL_NAME=mistralai/voxtral-mini-transcribe
ASR_MAX_CHUNK_SECONDS=600
ASR_CHUNK_OVERLAP_SECONDS=8
```

`ASR_API_KEY` and `ASR_BASE_URL` are optional. If blank, the code uses
`OPENROUTER_API_KEY` and `OPENROUTER_BASE_URL`.

`ffmpeg` and `ffprobe` must be available on `PATH`. The pipeline converts every
audio request to 16 kHz mono WAV before ASR because some providers reject `m4a`
or other containers.

## Run

Index all audio files:

```powershell
lake-index-audio
```

Index one subfolder:

```powershell
lake-index-audio --prefix meetings
```

Re-run ASR for unchanged files while debugging:

```powershell
lake-index-audio --force
```

Persist transcripts without creating embeddings:

```powershell
lake-index-audio --no-vector
```

Use a shorter ASR request size for long files:

```powershell
lake-index-audio --max-chunk-seconds 300 --chunk-overlap-seconds 5
```

## Imported Transcripts

To avoid ASR cost, place transcript JSON files in a transcript directory and run:

```powershell
lake-index-audio --transcript-dir transcripts
```

The preferred JSON shape is:

```json
{
  "relative_path": "workshop_03.22.m4a",
  "source_sha1": "",
  "duration_seconds": 51.861,
  "language": "en",
  "model": "external/asr-model",
  "segments": [
    {
      "start_seconds": 0.0,
      "end_seconds": 12.4,
      "speaker": null,
      "text": "Transcript text..."
    }
  ],
  "full_text": "Transcript text..."
}
```

Accepted filenames are:

- `transcripts/<relative_path>.json`
- `transcripts/<audio_filename>.json`
- `transcripts/<audio_stem>.json`

If `source_sha1` is present, the importer verifies it against the audio file and
fails on mismatch.

## Skip Behavior

By default, a file is skipped when `relative_path`, `size_bytes`, and
`last_modified` match the previous indexed row. This prevents repeated ASR cost.
Use `--force` whenever you intentionally want to transcribe/index it again.
