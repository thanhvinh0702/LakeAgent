CREATE TABLE IF NOT EXISTS inventory_runs (
    run_id UUID PRIMARY KEY,
    bucket TEXT NOT NULL,
    prefix TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL,
    listing_completed BOOLEAN NOT NULL DEFAULT FALSE,
    discovered_count BIGINT NOT NULL DEFAULT 0,
    identified_count BIGINT NOT NULL DEFAULT 0,
    unchanged_count BIGINT NOT NULL DEFAULT 0,
    error_count BIGINT NOT NULL DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS storage_objects (
    object_id TEXT PRIMARY KEY,
    object_identity TEXT NOT NULL UNIQUE,
    bucket TEXT NOT NULL,
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
    sha256 TEXT,
    user_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    warnings JSONB NOT NULL DEFAULT '[]'::jsonb,
    status TEXT NOT NULL,
    first_seen_run_id UUID NOT NULL REFERENCES inventory_runs(run_id),
    last_seen_run_id UUID NOT NULL REFERENCES inventory_runs(run_id),
    is_present BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_storage_objects_bucket_key
    ON storage_objects(bucket, object_key);

CREATE INDEX IF NOT EXISTS idx_storage_objects_modality
    ON storage_objects(modality);

CREATE INDEX IF NOT EXISTS idx_storage_objects_format
    ON storage_objects(detected_format);

CREATE INDEX IF NOT EXISTS idx_storage_objects_sha256
    ON storage_objects(sha256)
    WHERE sha256 IS NOT NULL;

CREATE TABLE IF NOT EXISTS inventory_errors (
    error_id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES inventory_runs(run_id),
    bucket TEXT NOT NULL,
    object_key TEXT,
    version_id TEXT,
    stage TEXT NOT NULL,
    error_type TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inventory_errors_run
    ON inventory_errors(run_id);
