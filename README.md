# LakeAgent: Multi-Modal Data Lake Inventory & Question Answering

LakeAgent is a comprehensive solution designed to automatically scan, inventory, parse, enrich with AI (LLM/VLM/OCR), and vector-index multi-modal data lake files. It is built to tackle complex question-answering tasks over heterogeneous data repositories containing **Text**, **Tabular**, **Image**, **Database/SQL**, **Audio/Video**, and other formats across multiple languages.

---

## 🎯 Problem Description & Challenge Overview

This project is designed to address the multi-modal data lake question-answering challenge. The objective is to build a system capable of answering natural language queries by automatically discovering, understanding, and reasoning over a diverse collection of files.

### 1. Core Objectives
Given an input question in natural language (which can be in various languages), the system must:
1. **Retrieve:** Automatically identify the relevant files (evidence) from the multi-modal data lake.
2. **Comprehend:** Parse and understand the content within those files, regardless of format or language.
3. **Reason & Calculate:** Perform any necessary calculations, aggregations, or logical reasoning.
4. **Answer:** Return a precise answer formatted according to the query instructions.

### 2. Input Data
* **Data Lake:** A diverse set of files in different formats (such as `.txt`, `.md`, `.csv`, `.tsv`, `.xlsx`, `.png`, `.jpg`, etc.) and multiple languages.
* **Natural Language Queries:** Questions ranging from simple information extraction to complex quantitative calculations, comparisons, and statistical analyses.
  * *Example 1:* `Which file contains the highest average sales in Q2?`
  * *Example 2:* `Did lightning or humans cause more fires impacting above 100 acres?`

### 3. Output Requirements
For each question, the system must return:
1. **Final Answer:** The exact answer text or number. For quantitative questions, the value must precisely adhere to requested units, rounding rules, date/percentage formatting, or other specified formats.
2. **Evidences:** A JSON-formatted array of filenames representing the source files utilized to derive the answer.

### 4. Submission & Evaluation
* **Submission Format:** A CSV file named `submission.csv` with exactly three columns:
  ```csv
  id,answer,evidences
  1,2026,"["file_1.csv"]"
  2,Human,"["file_2.txt","file_3.pdf"]"
  3,No,"[]"
  ```
* **Evaluation Metrics:**
  * **Exact Match (EM):** For short, objective queries (e.g., numbers, classification labels, Yes/No answers).
  * **LLM Judge:** For open-ended questions. It assesses the semantic equivalence and quality of the generated explanation compared to the ground truth.
  * *Tie-breakers:* Runtime efficiency, accuracy of file retrieval, and evidence completeness.

---

## 🏗️ Project Structure

Below is the detailed directory structure and module distribution of LakeAgent:

```text
lake-agent/
├── .env.example                # Template for configuring environment variables
├── .gitignore                  # Git ignore patterns
├── pyproject.toml              # Project metadata, dependencies, and CLI script registrations
├── test.ipynb                  # Jupyter Notebook for testing and rapid prototyping
├── deployments/
│   └── docker/
│       └── docker-compose.yaml # Docker setup for Postgres (with pgvector) & Adminer
├── src/
│   └── lake_agent/
│       ├── __init__.py
│       ├── config.py           # Loads and validates configuration from environment variables
│       ├── cli/                # Command-line entry points
│       │   ├── __init__.py
│       │   ├── index_image.py  # CLI for indexing image data
│       │   ├── index_tabular.py# CLI for indexing tabular files (CSV, Excel)
│       │   ├── index_text.py   # CLI for indexing text files (Markdown, TXT)
│       │   └── inventory.py    # CLI for data lake discovery and file cataloging
│       ├── domain/             # Domain models and enums
│       │   ├── __init__.py
│       │   ├── enums.py        # Modality (Tabular, Image, Text, etc.) and FileStatus definitions
│       │   ├── models.py       # Core FileMetadata definition
│       │   └── indexing_models/ # Models representing structured outputs of parser and enrichment stages
│       │       ├── __init__.py
│       │       ├── image.py
│       │       ├── image_enrichment.py
│       │       ├── tabular.py
│       │       ├── tabular_enrichment.py
│       │       ├── text.py
│       │       └── text_enrichment.py
│       ├── indexing/           # Modality-specific parsing and AI enrichment pipelines
│       │   ├── __init__.py
│       │   ├── image/          # Image parsing, OCR (to Markdown), and VLM analysis
│       │   ├── tabular/        # Tabular data parsing (CSV, TSV, Excel) and LLM summary generation
│       │   └── text/           # Text document chunking and LLM enrichment
│       ├── inventory/          # Discovering and identifying storage objects
│       │   ├── __init__.py
│       │   ├── identifier.py   # Magic-byte/Extension identifier for mime-type and format detection
│       │   ├── scanner.py      # Traverses files inside the object storage
│       │   └── service.py      # Core service managing file classification and canonical extension renaming
│       ├── persistence/        # Database layer (PostgreSQL)
│       │   ├── __init__.py
│       │   ├── database.py     # Database connection manager
│       │   ├── repositories.py # Repositories for storage objects, tabular, text, and image indices
│       │   └── schema.sql      # Database schema definition
│       └── storage/            # Storage abstraction layer
│           ├── __init__.py
│           ├── base.py         # ObjectStore abstract interface
│           └── local_store.py  # Local filesystem-based implementation of ObjectStore
└── tests/                      # Unit and integration tests
    ├── test_image_indexing.py
    ├── test_inventory.py
    ├── test_tabular_enrichment.py
    ├── test_tabular_indexing.py
    ├── test_tabular_persistence.py
    └── test_text_indexing.py
```

---

## 🛠️ Local Installation

### Prerequisites
* Python $\ge$ 3.11
* Docker & Docker Compose (to run PostgreSQL with `pgvector`)

### Installation Steps

**Using traditional `pip`:**
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# Install the package in editable mode
pip install -e .
```

**Using `uv` (recommended for faster dependency resolution):**
```bash
# Install the package in editable mode with uv
uv pip install -e .
```

---

## ⚙️ Configuration

Copy the example environment file and fill in the required credentials:

```bash
cp .env.example .env
```

Key environment variables in `.env`:

* **Database (PostgreSQL):**
  * `POSTGRES_DB_HOST=localhost`
  * `POSTGRES_DB_PORT=5432`
  * `POSTGRES_DB=lakeagent_db`
  * `POSTGRES_DB_USER=lakeagent`
  * `POSTGRES_DB_PASSWORD=lakeagent123` (or set a single `POSTGRES_DSN`)
* **Datalake Path:**
  * `DATALAKE_DIR=/path/to/your/Data-Lake` (Points to the directory containing files to inventory and index)
* **LLM & Embeddings (OpenAI API):**
  * `OPENAI_API_KEY=sk-...`
  * `OPENAI_MODEL_NAME=gpt-4o` (Used for LLM Enrichment)
  * `OPENAI_EMBEDDING_MODEL_NAME=text-embedding-3-small` (Used to generate vector embeddings)
  * `OPENAI_BASE_URL=` (Optional gateway proxy URL)
  * `OPENAI_EMBEDDING_DIMENSIONS=1536`
* **OCR & Vision Language Model (Optional for Image Processing):**
  * `OCR_MODEL_URL=http://...` (Service URL for converting images to Markdown text)
  * `VL_MODEL_NAME=gpt-4o` (VLM name for image summary extraction)

---

## 🚀 Usage Guide

### 1. Start the Database
Run the following command to start PostgreSQL (equipped with `pgvector`) and Adminer in the background:

```bash
docker compose -f deployments/docker/docker-compose.yaml up -d
```
* **PostgreSQL:** Port `5432`
* **Adminer (Database Client UI):** Accessible at [http://localhost:8080](http://localhost:8080)

### 2. Discover and Catalog the Data Lake (Inventory)
Run `lake-inventory` to scan `DATALAKE_DIR`, identify file formats (via extensions and signatures), automatically rename extensions to canonical forms if they disagree, and register the files in the `storage_objects` table.

```bash
# Scan the entire data lake
lake-inventory

# Scan a specific sub-folder inside DATALAKE_DIR
lake-inventory --prefix subfolder_name

# Skip check for filesystem stats (stat_object) on new/changed files
lake-inventory --no-stat
```

### 3. Index Text Data (Text Indexing)
Run `lake-index-text` to chunk text documents (`.txt`, `.md`), invoke LLM for key summary/keyword enrichment, and insert the sections and embeddings into the database and the vector store.

```bash
lake-index-text

# Limit indexing to a sub-folder
lake-index-text --prefix documents

# Disable LLM enrichment (save only deterministic parsed content)
lake-index-text --no-enrich
```

### 4. Index Tabular Data (Tabular Indexing)
Run `lake-index-tabular` to parse tables (`.csv`, `.tsv`, `.xlsx`), extract headers/previews, generate structural summaries using LLM, and ingest them into the vector store.

```bash
lake-index-tabular

# Disable LLM enrichment for faster indexing
lake-index-tabular --no-enrich
```

### 5. Index Image Data (Image Indexing)
Run `lake-index-image` to index visual assets (`.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.tiff`). If OCR and VLM options are enabled, it will convert text within images to Markdown and generate detailed image descriptions.

```bash
lake-index-image

# Disable OCR extraction
lake-index-image --no-ocr

# Disable Vision Language Model enrichment
lake-index-image --no-vlm
```

---

## 🧪 Testing

The codebase includes a test suite managed via `pytest`. Run the tests to verify the installation:

```bash
# Run all tests
pytest

# Run tests for a specific module
pytest tests/test_inventory.py
```
