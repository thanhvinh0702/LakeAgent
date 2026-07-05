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
