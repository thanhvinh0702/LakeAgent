CREATE TABLE IF NOT EXISTS storage_objects (
    object_id TEXT PRIMARY KEY,
    object_identity TEXT NOT NULL UNIQUE,
    object_key TEXT NOT NULL,
    version_id TEXT,
    etag TEXT,
    filename TEXT NOT NULL,
    extension TEXT NOT NULL DEFAULT '',
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    declared_content_type TEXT,
    detected_mime_type TEXT,
    detected_format TEXT,
    modality TEXT,
    encoding TEXT,
    identification_confidence DOUBLE PRECISION,
    user_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_storage_objects_key
    ON storage_objects(object_key);

CREATE INDEX IF NOT EXISTS idx_storage_objects_modality
    ON storage_objects(modality);

CREATE INDEX IF NOT EXISTS idx_storage_objects_format
    ON storage_objects(detected_format);


CREATE TABLE IF NOT EXISTS tabular_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    parser_version TEXT NOT NULL DEFAULT 'v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    workbook_sheet_descriptions JSONB NOT NULL DEFAULT '{}'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tabular_files_relative_path
    ON tabular_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_tabular_files_format
    ON tabular_files(file_format);

CREATE INDEX IF NOT EXISTS idx_tabular_files_status
    ON tabular_files(status);


CREATE TABLE IF NOT EXISTS tabular_tables (
    table_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES tabular_files(source_id) ON DELETE CASCADE,
    table_name TEXT NOT NULL,
    sheet_name TEXT,
    is_context_sheet BOOLEAN NOT NULL DEFAULT FALSE,
    sheet_description TEXT,
    header_row_index INTEGER,
    context_before_header JSONB NOT NULL DEFAULT '[]'::jsonb,
    raw_header JSONB NOT NULL DEFAULT '[]'::jsonb,
    row_count INTEGER,
    column_count INTEGER NOT NULL DEFAULT 0,
    columns_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    preview_rows JSONB NOT NULL DEFAULT '[]'::jsonb,
    summary TEXT,
    keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    table_search_text TEXT,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tabular_tables_source_id
    ON tabular_tables(source_id);

CREATE INDEX IF NOT EXISTS idx_tabular_tables_sheet_name
    ON tabular_tables(sheet_name);


CREATE TABLE IF NOT EXISTS text_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    parser_version TEXT NOT NULL DEFAULT 'v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_text_files_relative_path
    ON text_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_text_files_format
    ON text_files(file_format);

CREATE INDEX IF NOT EXISTS idx_text_files_status
    ON text_files(status);


CREATE TABLE IF NOT EXISTS text_sections (
    section_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES text_files(source_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    char_count INTEGER NOT NULL DEFAULT 0,
    search_text TEXT,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_text_sections_source_id
    ON text_sections(source_id);

CREATE INDEX IF NOT EXISTS idx_text_sections_chunk_index
    ON text_sections(chunk_index);


CREATE TABLE IF NOT EXISTS epub_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    title TEXT,
    creators JSONB NOT NULL DEFAULT '[]'::jsonb,
    language TEXT,
    publisher TEXT,
    identifier TEXT,
    chapter_count INTEGER NOT NULL DEFAULT 0,
    image_count INTEGER NOT NULL DEFAULT 0,
    vl_model_name TEXT,
    parser_version TEXT NOT NULL DEFAULT 'epub_zip_xhtml_v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_epub_files_relative_path
    ON epub_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_epub_files_format
    ON epub_files(file_format);

CREATE INDEX IF NOT EXISTS idx_epub_files_status
    ON epub_files(status);


CREATE TABLE IF NOT EXISTS epub_sections (
    section_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES epub_files(source_id) ON DELETE CASCADE,
    section_type TEXT NOT NULL DEFAULT 'chapter_text',
    chunk_index INTEGER NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    chapter_index INTEGER,
    chapter_title TEXT,
    chapter_href TEXT,
    image_id TEXT,
    image_index INTEGER,
    image_href TEXT,
    char_count INTEGER NOT NULL DEFAULT 0,
    search_text TEXT,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_epub_sections_source_id
    ON epub_sections(source_id);

CREATE INDEX IF NOT EXISTS idx_epub_sections_chunk_index
    ON epub_sections(chunk_index);

CREATE INDEX IF NOT EXISTS idx_epub_sections_chapter_index
    ON epub_sections(chapter_index);


CREATE TABLE IF NOT EXISTS audio_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    duration_seconds DOUBLE PRECISION,
    codec_name TEXT,
    sample_rate INTEGER,
    channels INTEGER,
    transcript_language TEXT,
    transcript_text TEXT,
    asr_model_name TEXT,
    asr_cost_usd DOUBLE PRECISION,
    asr_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    parser_version TEXT NOT NULL DEFAULT 'asr_wav16k_v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audio_files_relative_path
    ON audio_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_audio_files_format
    ON audio_files(file_format);

CREATE INDEX IF NOT EXISTS idx_audio_files_status
    ON audio_files(status);

CREATE TABLE IF NOT EXISTS video_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    duration_seconds DOUBLE PRECISION,
    width INTEGER,
    height INTEGER,
    fps DOUBLE PRECISION,
    video_codec TEXT,
    audio_codec TEXT,
    has_audio BOOLEAN NOT NULL DEFAULT FALSE,
    sampled_frame_count INTEGER NOT NULL DEFAULT 0,
    asr_model_name TEXT,
    asr_cost_usd DOUBLE PRECISION,
    asr_usage JSONB NOT NULL DEFAULT '{}'::jsonb,
    vl_model_name TEXT,
    parser_version TEXT NOT NULL DEFAULT 'video_audio_frame_v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_video_files_relative_path
    ON video_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_video_files_format
    ON video_files(file_format);

CREATE INDEX IF NOT EXISTS idx_video_files_status
    ON video_files(status);

CREATE TABLE IF NOT EXISTS web_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    parser_version TEXT NOT NULL DEFAULT 'html_v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_web_files_relative_path
    ON web_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_web_files_format
    ON web_files(file_format);

CREATE INDEX IF NOT EXISTS idx_web_files_status
    ON web_files(status);


CREATE TABLE IF NOT EXISTS web_sections (
    section_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES web_files(source_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    char_count INTEGER NOT NULL DEFAULT 0,
    search_text TEXT,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_web_sections_source_id
    ON web_sections(source_id);

CREATE INDEX IF NOT EXISTS idx_web_sections_chunk_index
    ON web_sections(chunk_index);


CREATE TABLE IF NOT EXISTS document_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    parser_version TEXT NOT NULL DEFAULT 'docling_hierarchical_v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_document_files_relative_path
    ON document_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_document_files_format
    ON document_files(file_format);

CREATE INDEX IF NOT EXISTS idx_document_files_status
    ON document_files(status);


CREATE TABLE IF NOT EXISTS document_sections (
    section_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES document_files(source_id) ON DELETE CASCADE,
    section_type TEXT NOT NULL DEFAULT 'document_chunk',
    chunk_index INTEGER NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    char_count INTEGER NOT NULL DEFAULT 0,
    search_text TEXT,
    image_id TEXT,
    image_index INTEGER,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE document_sections
    ADD COLUMN IF NOT EXISTS section_type TEXT NOT NULL DEFAULT 'document_chunk';

ALTER TABLE document_sections
    ADD COLUMN IF NOT EXISTS image_id TEXT;

ALTER TABLE document_sections
    ADD COLUMN IF NOT EXISTS image_index INTEGER;

CREATE INDEX IF NOT EXISTS idx_document_sections_source_id
    ON document_sections(source_id);

CREATE INDEX IF NOT EXISTS idx_document_sections_chunk_index
    ON document_sections(chunk_index);


CREATE TABLE IF NOT EXISTS json_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    parser_version TEXT NOT NULL DEFAULT 'flattened_json_v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    top_level_type TEXT,
    entry_count INTEGER NOT NULL DEFAULT 0,
    max_depth INTEGER NOT NULL DEFAULT 0,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_json_files_relative_path
    ON json_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_json_files_format
    ON json_files(file_format);

CREATE INDEX IF NOT EXISTS idx_json_files_status
    ON json_files(status);


CREATE TABLE IF NOT EXISTS json_sections (
    section_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES json_files(source_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    path_start TEXT,
    path_end TEXT,
    entry_count INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    char_count INTEGER NOT NULL DEFAULT 0,
    search_text TEXT,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_json_sections_source_id
    ON json_sections(source_id);

CREATE INDEX IF NOT EXISTS idx_json_sections_chunk_index
    ON json_sections(chunk_index);


CREATE TABLE IF NOT EXISTS slideshow_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    parser_version TEXT NOT NULL DEFAULT 'docling_hierarchical_v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_slideshow_files_relative_path
    ON slideshow_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_slideshow_files_format
    ON slideshow_files(file_format);

CREATE INDEX IF NOT EXISTS idx_slideshow_files_status
    ON slideshow_files(status);


CREATE TABLE IF NOT EXISTS slideshow_sections (
    section_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES slideshow_files(source_id) ON DELETE CASCADE,
    section_type TEXT NOT NULL DEFAULT 'slide_chunk',
    chunk_index INTEGER NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    slide_start INTEGER,
    slide_end INTEGER,
    char_count INTEGER NOT NULL DEFAULT 0,
    search_text TEXT,
    image_id TEXT,
    image_index INTEGER,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE slideshow_sections
    ADD COLUMN IF NOT EXISTS section_type TEXT NOT NULL DEFAULT 'slide_chunk';

ALTER TABLE slideshow_sections
    ADD COLUMN IF NOT EXISTS image_id TEXT;

ALTER TABLE slideshow_sections
    ADD COLUMN IF NOT EXISTS image_index INTEGER;

CREATE INDEX IF NOT EXISTS idx_slideshow_sections_source_id
    ON slideshow_sections(source_id);

CREATE INDEX IF NOT EXISTS idx_slideshow_sections_chunk_index
    ON slideshow_sections(chunk_index);


CREATE TABLE IF NOT EXISTS image_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    color_mode TEXT NOT NULL,
    has_alpha BOOLEAN NOT NULL DEFAULT FALSE,
    is_animated BOOLEAN NOT NULL DEFAULT FALSE,
    frame_count INTEGER NOT NULL DEFAULT 1,
    parser_version TEXT NOT NULL DEFAULT 'v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_image_files_relative_path
    ON image_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_image_files_format
    ON image_files(file_format);

CREATE INDEX IF NOT EXISTS idx_image_files_status
    ON image_files(status);


CREATE TABLE IF NOT EXISTS image_sections (
    section_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES image_files(source_id) ON DELETE CASCADE,
    section_type TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    heading TEXT,
    content TEXT NOT NULL,
    line_start INTEGER,
    line_end INTEGER,
    char_count INTEGER NOT NULL DEFAULT 0,
    search_text TEXT,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_image_sections_source_id
    ON image_sections(source_id);

CREATE INDEX IF NOT EXISTS idx_image_sections_chunk_index
    ON image_sections(chunk_index);

CREATE TABLE IF NOT EXISTS sql_script_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    parser_version TEXT NOT NULL DEFAULT 'v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sql_script_files_relative_path
    ON sql_script_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_sql_script_files_format
    ON sql_script_files(file_format);

CREATE INDEX IF NOT EXISTS idx_sql_script_files_status
    ON sql_script_files(status);

CREATE TABLE IF NOT EXISTS database_files (
    source_id TEXT PRIMARY KEY,
    relative_path TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    file_format TEXT NOT NULL,
    size_bytes BIGINT NOT NULL CHECK (size_bytes >= 0),
    last_modified TIMESTAMPTZ,
    parser_version TEXT NOT NULL DEFAULT 'v1',
    parse_warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_summary TEXT,
    file_keywords JSONB NOT NULL DEFAULT '[]'::jsonb,
    file_search_text TEXT,
    status TEXT NOT NULL DEFAULT 'indexed',
    error_message TEXT,
    first_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_indexed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_database_files_relative_path
    ON database_files(relative_path);

CREATE INDEX IF NOT EXISTS idx_database_files_format
    ON database_files(file_format);

CREATE INDEX IF NOT EXISTS idx_database_files_status
    ON database_files(status);
