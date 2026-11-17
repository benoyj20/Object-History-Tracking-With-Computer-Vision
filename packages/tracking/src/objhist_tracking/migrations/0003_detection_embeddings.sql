-- Appearance embeddings of detection crops (ADR 0003), computed once and reused by every
-- tracker run that compares appearance. Like detections, rows are evidence and are never
-- updated: a different model or model revision is a different `embedding_model`.

CREATE TABLE detection_embeddings (
    detection_id INTEGER NOT NULL REFERENCES detections (detection_id),
    -- Model ID and the exact revision that produced the vector, e.g.
    -- 'facebook/dinov2-small@<commit hash>'.
    embedding_model TEXT NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    -- Little-endian float32 values with unit length, so a dot product is cosine similarity.
    embedding BLOB NOT NULL CHECK (length(embedding) = 4 * dimensions),
    created_at TEXT NOT NULL,
    PRIMARY KEY (detection_id, embedding_model)
) STRICT;
