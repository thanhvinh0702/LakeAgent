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
