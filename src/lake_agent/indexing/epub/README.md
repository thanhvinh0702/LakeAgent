# EPUB Indexing

EPUB indexing reads `.epub` files directly as ZIP-based books. It extracts
chapter XHTML text, chunks that text for embedding, creates a file-level summary,
and captions embedded images with a VLM by default.

## Environment

EPUB text parsing is deterministic and does not call an LLM. Embedded image
captioning uses an OpenAI-compatible chat endpoint through LangChain.

```env
EPUB_VL_API_KEY=
EPUB_VL_BASE_URL=http://localhost:20128/v1
EPUB_VL_MODEL_NAME=openrouter/qwen/qwen3-vl-32b-instruct
EPUB_VL_LONG_EDGE=768
EPUB_MAX_IMAGES_PER_FILE=20
EPUB_ENRICH_SECTION_COUNT=12
```

`EPUB_VL_API_KEY` falls back to `VIDEO_VL_API_KEY`, then `OPENAI_API_KEY`, then
`API_KEY`. `EPUB_VL_BASE_URL` falls back to video/VL/OpenAI base URLs, then
defaults to local 9Router at `http://localhost:20128/v1`.

The `high` setting for Qwen is selected in the 9Router UI. The model id passed
to LangChain remains `openrouter/qwen/qwen3-vl-32b-instruct`.

## Run

Index EPUB files with default image VLM captioning:

```powershell
lake-index-epub --prefix Light_novel
```

Cheap smoke test without embeddings:

```powershell
lake-index-epub --prefix Light_novel --no-vector --max-images-per-file 1 --force
```

Smoke test one EPUB file:

```powershell
lake-index-epub --prefix "Light_novel/<book>.epub" --no-vector --max-images-per-file 1 --force
```

Text-only indexing without VLM cost:

```powershell
lake-index-epub --prefix Light_novel --no-vlm
```

Skip file-level summary/keywords while debugging:

```powershell
lake-index-epub --prefix Light_novel --no-enrich --no-vlm --no-vector
```

## Retrieval

EPUB creates one vector document for the file plus section documents for:

- `chapter_text`: chapter text chunks with chapter metadata
- `image_summary`: VLM captions for embedded images

Agents can search these with `search_epub_data`, and `search_all_indexed_data`
also includes EPUB hits.
