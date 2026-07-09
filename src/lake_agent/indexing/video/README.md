# Video Indexing

Video indexing extracts the audio track for ASR and optionally captions sampled
frames with a VLM. It does not run OCR.

## Environment

Audio transcription uses the same ASR settings as audio indexing:

```env
OPENROUTER_API_KEY=...
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
ASR_MODEL_NAME=qwen/qwen3-asr-flash-2026-02-10
ASR_FALLBACK_MODEL_NAME=mistralai/voxtral-mini-transcribe
```

Optional VLM frame captioning uses the existing OpenAI-compatible chat settings.
`VIDEO_VL_MODEL_NAME` falls back to `VL_MODEL_NAME`, then `OPENAI_MODEL_NAME`.
Frames are sampled evenly from the video duration, up to `VIDEO_MAX_FRAMES`.

```env
VIDEO_MAX_FRAMES=8
VIDEO_FRAME_LONG_EDGE=768
VIDEO_VL_MODEL_NAME=
```

`ffmpeg` and `ffprobe` must be available on `PATH`.

## Run

Index videos with audio ASR only:

```powershell
lake-index-video --no-vector --force
```

Caption sampled frames with VLM, capped to one frame per video for a cheap smoke test:

```powershell
lake-index-video --no-vector --vlm --max-frames 1 --force
```

Change frame sampling:

```powershell
lake-index-video --vlm --max-frames 8
```

Skip audio and only caption sampled frames:

```powershell
lake-index-video --no-audio --vlm --max-frames 3
```

## Cost Controls

By default, VLM is off. Frame captioning samples up to `VIDEO_MAX_FRAMES`
evenly spaced frames across the video duration. `VIDEO_FRAME_LONG_EDGE` defaults
to 768 to keep image payloads small. Re-running without `--force` skips unchanged
files by `relative_path`, `size_bytes`, and `last_modified`.
