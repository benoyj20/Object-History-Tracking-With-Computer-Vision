-- Detector output imported from `SessionDetections` files (ADR 0005). These rows are the
-- evidence the tracker works from and are never updated in place: a detector with different
-- weights or settings has a different version and gets its own rows.

CREATE TABLE detector_runs (
    version TEXT PRIMARY KEY,
    weights_sha256 TEXT NOT NULL,
    class_names_json TEXT NOT NULL,
    confidence_threshold REAL NOT NULL CHECK (confidence_threshold BETWEEN 0 AND 1),
    settings_json TEXT NOT NULL,
    framework_version TEXT NOT NULL
) STRICT;

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    capture_time_source TEXT NOT NULL CHECK (capture_time_source IN ('manifest', 'exif'))
) STRICT;

-- One row per photo, shared by every detector that processed it. `captured_at_utc` is
-- ISO 8601 in UTC with microseconds, so text order is time order; `utc_offset_seconds` keeps
-- the photo's local offset for display.
CREATE TABLE frames (
    frame_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions (session_id),
    captured_at_utc TEXT NOT NULL,
    utc_offset_seconds INTEGER NOT NULL,
    image_width_px INTEGER NOT NULL CHECK (image_width_px > 0),
    image_height_px INTEGER NOT NULL CHECK (image_height_px > 0)
) STRICT;

CREATE INDEX frames_by_session_time ON frames (session_id, captured_at_utc);

-- One row per imported detection file. `content_sha256` covers the session, detector record,
-- and frames but not the file's `created_at`, so rerunning the same detector is recognized.
CREATE TABLE session_imports (
    detector_version TEXT NOT NULL REFERENCES detector_runs (version),
    session_id TEXT NOT NULL REFERENCES sessions (session_id),
    content_sha256 TEXT NOT NULL,
    file_created_at TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    PRIMARY KEY (detector_version, session_id)
) STRICT;

-- The frames each detector processed. A processed frame with no detections means the
-- detector ran and found nothing; a frame missing here was never processed by that detector.
CREATE TABLE processed_frames (
    detector_version TEXT NOT NULL REFERENCES detector_runs (version),
    frame_id TEXT NOT NULL REFERENCES frames (frame_id),
    PRIMARY KEY (detector_version, frame_id)
) STRICT;

-- `detection_index` is the detection's position in its frame's list in the file, which is
-- the only identity a detection has before tracking.
CREATE TABLE detections (
    detection_id INTEGER PRIMARY KEY,
    detector_version TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    detection_index INTEGER NOT NULL CHECK (detection_index >= 0),
    class_name TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    x_min_px REAL NOT NULL CHECK (x_min_px >= 0),
    y_min_px REAL NOT NULL CHECK (y_min_px >= 0),
    x_max_px REAL NOT NULL,
    y_max_px REAL NOT NULL,
    CHECK (x_max_px > x_min_px AND y_max_px > y_min_px),
    UNIQUE (detector_version, frame_id, detection_index),
    FOREIGN KEY (detector_version, frame_id)
        REFERENCES processed_frames (detector_version, frame_id)
) STRICT;
