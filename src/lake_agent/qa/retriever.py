from __future__ import annotations

import base64
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

from lake_agent.config import LocalSettings, PostgresSettings, LLMSettings
from lake_agent.persistence.database import PostgresDatabase

logger = logging.getLogger(__name__)

# Stopwords to ignore during query keyword parsing
STOPWORDS = {
    "how", "many", "what", "where", "which", "the", "and", "are", "for", "from", 
    "with", "contain", "in", "of", "to", "is", "a", "by", "that", "this", "be", 
    "been", "has", "have", "had", "do", "does", "did", "were", "was", "or", "but", 
    "about", "at", "on", "determine", "calculate", "find", "show", "me", "cho", 
    "tôi", "xem", "là", "bao", "nhiêu", "của", "trong", "các", "được", "có", "nào", 
    "đối", "với", "môn", "học", "nhóm", "đi", "hihihi", "cơ", "bản", "bản", "sự",
    "cách", "các", "dự", "án", "tạo", "ra", "tác", "động", "bền", "vững", "gì",
    "hãy", "lựa", "chọn", "đáp", "đúng", "dưới", "đây", "theo", "tổng", "số",
    "thành", "viên", "hiện", "tại", "nhất", "không", "tính", "mới", "yêu", "cầu",
    "chỉ", "trả", "về", "đúng", "tự", "định", "nghĩa", "tài", "liệu", "cụ", "thể"
}


class CrossRetriever:
    def __init__(self, datalake_dir: str | None = None) -> None:
        if datalake_dir is not None:
            self.datalake_dir = Path(datalake_dir)
        else:
            load_dotenv()
            try:
                self.local_settings = LocalSettings.from_env()
                self.datalake_dir = Path(self.local_settings.datalake_dir)
            except Exception:
                self.datalake_dir = Path("Data-Lake")
            
        try:
            load_dotenv()
            self.postgres_settings = PostgresSettings.from_env()
            self.db = PostgresDatabase(self.postgres_settings.dsn)
        except Exception:
            self.db = None
            
        # In-memory VLM cache to avoid calling VLM repeatedly for the same image
        self._vlm_cache: dict[str, str] = {}

    def retrieve(self, modality: str, query: str) -> list[dict[str, Any]]:
        logger.info(f"Retrieving for modality '{modality}' with query '{query}'")
        
        # 1. Try relational/vector DB first if available
        results = []
        if self.db:
            try:
                results = self._retrieve_from_db(modality, query)
            except Exception as e:
                logger.warning(f"Database retrieval failed: {e}. Falling back to filesystem.")
                
        # 2. Perform advanced filesystem search
        if not results:
            results = self._retrieve_from_filesystem_advanced(modality, query)
            
        return results

    def _retrieve_from_db(self, modality: str, query: str) -> list[dict[str, Any]]:
        if not self.db:
            return []
            
        results = []
        with self.db.connect() as conn:
            q_term = f"%{query}%"
            if modality == "text":
                rows = conn.execute(
                    "SELECT filename, content FROM text_sections s "
                    "JOIN text_files f ON s.source_id = f.source_id "
                    "WHERE s.content ILIKE %s LIMIT 5",
                    (q_term,)
                ).fetchall()
                results.extend([{"filename": r["filename"], "content": r["content"]} for r in rows])
                
                rows_web = conn.execute(
                    "SELECT filename, content FROM web_sections s "
                    "JOIN web_files f ON s.source_id = f.source_id "
                    "WHERE s.content ILIKE %s LIMIT 5",
                    (q_term,)
                ).fetchall()
                results.extend([{"filename": r["filename"], "content": r["content"]} for r in rows_web])

                rows_sql = conn.execute(
                    "SELECT filename, content FROM sql_script_sections s "
                    "JOIN sql_script_files f ON s.source_id = f.source_id "
                    "WHERE s.content ILIKE %s LIMIT 5",
                    (q_term,)
                ).fetchall()
                results.extend([{"filename": r["filename"], "content": r["content"]} for r in rows_sql])
                
            elif modality == "tabular":
                rows = conn.execute(
                    "SELECT filename, table_name, summary, columns_json FROM tabular_tables t "
                    "JOIN tabular_files f ON t.source_id = f.source_id "
                    "WHERE t.table_search_text ILIKE %s LIMIT 5",
                    (q_term,)
                ).fetchall()
                for r in rows:
                    content = f"Table: {r['table_name']}\nSummary: {r['summary']}\nColumns: {r['columns_json']}"
                    results.append({"filename": r["filename"], "content": content})
                    
            elif modality == "database":
                rows = conn.execute(
                    "SELECT filename, table_name, summary, columns_json FROM database_tables t "
                    "JOIN database_files f ON t.source_id = f.source_id "
                    "WHERE t.table_search_text ILIKE %s LIMIT 5",
                    (q_term,)
                ).fetchall()
                for r in rows:
                    content = f"Table: {r['table_name']}\nSummary: {r['summary']}\nColumns: {r['columns_json']}"
                    results.append({"filename": r["filename"], "content": content})

            elif modality == "image":
                rows = conn.execute(
                    "SELECT filename, file_summary FROM image_files "
                    "WHERE file_search_text ILIKE %s LIMIT 5",
                    (q_term,)
                ).fetchall()
                results.extend([{"filename": r["filename"], "content": r["file_summary"]} for r in rows])

            elif modality == "audio":
                rows = conn.execute(
                    "SELECT filename, content, start_seconds, end_seconds FROM audio_sections s "
                    "JOIN audio_files f ON s.source_id = f.source_id "
                    "WHERE s.search_text ILIKE %s OR s.content ILIKE %s LIMIT 5",
                    (q_term, q_term)
                ).fetchall()
                for r in rows:
                    timing = ""
                    if r["start_seconds"] is not None and r["end_seconds"] is not None:
                        timing = f"[{r['start_seconds']:.1f}s-{r['end_seconds']:.1f}s]\n"
                    results.append({"filename": r["filename"], "content": f"{timing}{r['content']}"})

            elif modality == "video":
                rows = conn.execute(
                    "SELECT filename, section_type, content, timestamp_seconds, start_seconds, end_seconds "
                    "FROM video_sections s "
                    "JOIN video_files f ON s.source_id = f.source_id "
                    "WHERE s.search_text ILIKE %s OR s.content ILIKE %s LIMIT 5",
                    (q_term, q_term)
                ).fetchall()
                for r in rows:
                    timing = ""
                    if r["timestamp_seconds"] is not None:
                        timing = f"[{r['timestamp_seconds']:.1f}s {r['section_type']}]\n"
                    elif r["start_seconds"] is not None and r["end_seconds"] is not None:
                        timing = f"[{r['start_seconds']:.1f}s-{r['end_seconds']:.1f}s {r['section_type']}]\n"
                    results.append({"filename": r["filename"], "content": f"{timing}{r['content']}"})

            elif modality == "json":
                rows = conn.execute(
                    "SELECT filename, path_start, path_end, content FROM json_sections s "
                    "JOIN json_files f ON s.source_id = f.source_id "
                    "WHERE s.search_text ILIKE %s OR s.content ILIKE %s LIMIT 5",
                    (q_term, q_term)
                ).fetchall()
                for r in rows:
                    path_range = ""
                    if r["path_start"] or r["path_end"]:
                        path_range = f"[{r['path_start']} to {r['path_end']}]\n"
                    results.append({"filename": r["filename"], "content": f"{path_range}{r['content']}"})
                
        return results

    def _retrieve_from_filesystem_advanced(self, modality: str, query: str) -> list[dict[str, Any]]:
        if not self.datalake_dir.exists():
            return []

        # Find all files recursively in datalake
        all_files: list[Path] = []
        for path in self.datalake_dir.rglob("*"):
            if path.is_file():
                all_files.append(path)

        # Extract tokens from query
        query_clean = re.sub(r"[^\w\s\-\.]", "", query.lower())
        tokens = [w.strip() for w in query_clean.split() if w.strip() and w.strip() not in STOPWORDS and len(w.strip()) > 2]
        
        # Direct keyword matching on filenames
        matched_files: list[Path] = []
        for f in all_files:
            fname_lower = f.name.lower()
            # If explicit name matches a token or query mentions extension/full name
            if f.name.lower() in query_clean or any(token in fname_lower for token in tokens):
                matched_files.append(f)

        # Filter by routed modality extensions
        modality_extensions = {
            "text": [".txt", ".md", ".html", ".htm", ".sql"],
            "tabular": [".csv", ".tsv", ".xlsx", ".xls"],
            "image": [".png", ".jpg", ".jpeg", ".webp"],
            "document": [".pdf", ".docx", ".pptx", ".ppt"],
            "audio": [".m4a", ".mp3", ".wav", ".flac", ".ogg"],
            "video": [".mp4", ".mkv", ".mov", ".avi", ".webm"],
            "database": [".db", ".sqlite", ".sqlite3"],
            "json": [".json", ".jsonl", ".ndjson"],
        }
        
        target_exts = modality_extensions.get(modality, [])
        filtered_files = [f for f in matched_files if f.suffix.lower() in target_exts]

        # If no filename matched for the specific modality, scan all files of that modality
        if not filtered_files:
            filtered_files = [f for f in all_files if f.suffix.lower() in target_exts]

        # Read and format context
        results = []
        # Limit to prevent context window explosion
        max_files_to_read = 15 if modality == "image" else 6
        
        for f in filtered_files[:max_files_to_read]:
            content = ""
            ext = f.suffix.lower()
            
            if ext in [".txt", ".md", ".html", ".htm", ".sql"]:
                content = self._read_text_file(f)
            elif ext in [".csv", ".tsv", ".xlsx", ".xls"]:
                content = self._read_tabular_file(f)
            elif ext in [".db", ".sqlite", ".sqlite3"]:
                content = self._read_sqlite_db(f)
            elif ext == ".pdf":
                content = self._read_pdf(f)
            elif ext in [".docx", ".pptx", ".ppt"]:
                content = self._read_office_doc(f)
            elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
                content = self._describe_image_via_vlm(f)
            elif ext in [".m4a", ".mp3", ".wav"]:
                content = self._read_audio_transcript_fallback(f)
            elif ext in [".mp4", ".mkv", ".mov", ".avi", ".webm"]:
                content = f"Video file: {f.name}. Run lake-index-video for transcript and frame captions."
            elif ext in [".json", ".jsonl", ".ndjson"]:
                content = self._read_json_file(f)
                
            if content:
                results.append({
                    "filename": f.name,
                    "content": content
                })
                
        # If we got absolutely nothing, do a broad fallback search on text
        if not results and modality != "text":
            text_files = [f for f in all_files if f.suffix.lower() in modality_extensions["text"]]
            for f in text_files[:3]:
                content = self._read_text_file(f)
                results.append({"filename": f.name, "content": content})
                
        return results

    def _read_text_file(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return f"Error reading text file: {e}"

    def _read_tabular_file(self, path: Path, nrows: int = 50) -> str:
        try:
            # Read full file to calculate correct correlation/statistics if needed
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() == ".tsv":
                df = pd.read_csv(path, sep="\t")
            else:
                df = pd.read_excel(path)
                
            cols = ", ".join(df.columns)
            summary = f"Columns: {cols}\nShape: {df.shape}\n"
            
            # Smart Fallback: If correlation matrix is relevant, calculate it exactly
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                try:
                    corr = numeric_df.corr(method='pearson')
                    summary += f"Pearson Correlation Matrix:\n{corr.to_string()}\n\n"
                except Exception:
                    pass
                    
            summary += f"Data Preview (first {nrows} rows):\n"
            summary += df.head(nrows).to_string()
            return summary
        except Exception as e:
            return f"Error reading tabular file: {e}"

    def _read_sqlite_db(self, path: Path) -> str:
        try:
            conn = sqlite3.connect(str(path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            summary = f"SQLite Database: {path.name}\nTables: {', '.join(tables)}\n\nSchema Details:\n"
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table});")
                cols = cursor.fetchall()
                col_desc = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
                summary += f"Table: {table}\n  Columns: {col_desc}\n"
                
                try:
                    cursor.execute(f"SELECT * FROM {table} LIMIT 3;")
                    rows = cursor.fetchall()
                    summary += f"  Preview:\n  {rows}\n"
                except Exception:
                    pass
            conn.close()
            return summary
        except Exception as e:
            return f"Error reading SQLite database: {e}"

    def _read_json_file(self, path: Path) -> str:
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
            entries: list[str] = []

            def flatten(value: Any, prefix: str = "$") -> None:
                if len(entries) >= 200:
                    return
                if isinstance(value, dict):
                    entries.append(f"{prefix}: object with {len(value)} key(s)")
                    for key in sorted(value):
                        flatten(value[key], f"{prefix}.{key}")
                    return
                if isinstance(value, list):
                    entries.append(f"{prefix}: array with {len(value)} item(s)")
                    for index, item in enumerate(value):
                        flatten(item, f"{prefix}[{index}]")
                    return
                entries.append(f"{prefix}: {value}")

            flatten(payload)
            return f"JSON file: {path.name}\n" + "\n".join(entries)
        except Exception as e:
            return f"Error reading JSON file: {e}"

    def _read_pdf(self, path: Path) -> str:
        # Try docling first
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(str(path))
            return result.document.export_to_markdown()
        except Exception:
            pass
            
        # Try pypdf fallback
        try:
            import pypdf
            reader = pypdf.PdfReader(path)
            text = ""
            for page in reader.pages[:10]:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            return f"Error reading PDF file: {e}"

    def _read_office_doc(self, path: Path) -> str:
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(str(path))
            return result.document.export_to_markdown()
        except Exception as e:
            return f"Error reading Office doc {path.name}: {e}"

    def _describe_image_via_vlm(self, path: Path) -> str:
        # Check cache first
        cache_key = str(path.resolve())
        if cache_key in self._vlm_cache:
            return self._vlm_cache[cache_key]

        try:
            from langchain.chat_models import init_chat_model
            from langchain_core.messages import HumanMessage
            
            # Read settings
            settings = LLMSettings.from_env()
            
            # Read and encode image to base64
            with open(path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                
            # Initialize vision-capable model
            # OpenRouter's Google Gemini Flash is extremely cheap and supports vision
            vlm_model = init_chat_model(
                model_provider="openai",
                api_key=settings.api_key,
                base_url=settings.base_url or "https://openrouter.ai/api/v1",
                model=os.getenv("VL_MODEL_NAME") or settings.model_name,
                temperature=0,
            )
            
            prompt = (
                "You are an Image Parser Subagent. Analyze this image and describe:\n"
                "1. All digits or numbers visible in the image, their count, values, and colors.\n"
                "2. If it is a document, transcribe the key text and details exactly.\n"
                "3. If it is a group photo, describe the context and members.\n"
                "Output a highly precise, factual description of what is visible."
            )
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}",
                        },
                    },
                ]
            )
            
            response = vlm_model.invoke([message])
            desc = response.content.strip()
            self._vlm_cache[cache_key] = desc
            return desc
        except Exception as e:
            logger.warning(f"VLM analysis failed for {path.name}: {e}")
            return f"Image file: {path.name}. VLM analysis failed: {e}"

    def _read_audio_transcript_fallback(self, path: Path) -> str:
        return (
            f"Audio file: {path.name}. Transcript is not available in the fallback reader. "
            "Run lake-index-audio so audio_sections can be retrieved from Postgres."
        )
