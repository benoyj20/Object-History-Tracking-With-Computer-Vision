-- Tracker output: which detections are the same physical object, and the added, removed, and
-- moved events derived from that. Every row belongs to one tracker run, so rerunning the
-- tracker with other settings adds a new run beside the old ones instead of replacing them.

-- One row per run of the tracker over one session's detections from one detector version.
-- The composite foreign key requires that detector's file for that session to be imported.
CREATE TABLE tracker_runs (
    tracker_run_id INTEGER PRIMARY KEY,
    tracker_version TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    session_id TEXT NOT NULL,
    -- The embedding model used for appearance, e.g. 'dinov2-vits14'; NULL for the
    -- position-only (IoU) baseline.
    appearance_model TEXT,
    -- Detections below this were ignored; at least the detector's own threshold.
    confidence_threshold REAL NOT NULL CHECK (confidence_threshold BETWEEN 0 AND 1),
    -- Named association thresholds and weights, and the rule that confirms a removal, as
    -- sorted JSON, so every history records the configuration that produced it.
    settings_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (detector_version, session_id)
        REFERENCES session_imports (detector_version, session_id)
) STRICT;

-- One row per physical item the tracker believes exists in a run, e.g. "cup #2". An object
-- has one class for its whole life: tracks of different classes are never merged.
CREATE TABLE objects (
    object_id INTEGER PRIMARY KEY,
    tracker_run_id INTEGER NOT NULL REFERENCES tracker_runs (tracker_run_id),
    class_name TEXT NOT NULL,
    -- Lets observations and events check that their object belongs to their run.
    UNIQUE (object_id, tracker_run_id)
) STRICT;

-- One row per detection assigned to an object: "this box in this photo is cup #2". A photo
-- where an object has no observation is a photo where it was not seen, which on its own is
-- not a removal. The match evidence is NULL on the observation that created the object,
-- because there was nothing earlier to match against.
CREATE TABLE observations (
    tracker_run_id INTEGER NOT NULL,
    object_id INTEGER NOT NULL,
    detection_id INTEGER NOT NULL REFERENCES detections (detection_id),
    -- Distance between the object's expected and observed position (lower is closer); the
    -- unit depends on the position method and is named in the run's settings.
    position_distance REAL CHECK (position_distance >= 0),
    -- Cosine similarity of appearance embeddings; NULL when the run uses no appearance model.
    appearance_similarity REAL CHECK (appearance_similarity BETWEEN -1 AND 1),
    -- The combined cost the assignment minimized; lower is a better match.
    match_cost REAL,
    PRIMARY KEY (tracker_run_id, detection_id),
    FOREIGN KEY (object_id, tracker_run_id) REFERENCES objects (object_id, tracker_run_id)
) STRICT;

CREATE INDEX observations_by_object ON observations (object_id);

-- The history the assistant answers from. Snapshots are sparse, so a change is placed
-- between two photos, never at one instant:
-- - added: absent in frame_before, present in frame_after. frame_before is NULL when the
--   object was already there in the session's first photo, so its placement time is unknown.
-- - removed: last seen in frame_before, missing from frame_after onward. Written only once
--   the run's removal rule is met, so a missed detection or occlusion is not a removal.
-- - moved: at one place in frame_before, at another in frame_after.
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY,
    tracker_run_id INTEGER NOT NULL,
    object_id INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('added', 'removed', 'moved')),
    frame_before_id TEXT REFERENCES frames (frame_id),
    frame_after_id TEXT NOT NULL REFERENCES frames (frame_id),
    CHECK (frame_before_id IS NOT NULL OR event_type = 'added'),
    CHECK (frame_before_id IS NULL OR frame_before_id <> frame_after_id),
    FOREIGN KEY (object_id, tracker_run_id) REFERENCES objects (object_id, tracker_run_id)
) STRICT;

CREATE INDEX events_by_object ON events (object_id);
CREATE INDEX events_by_run ON events (tracker_run_id);
