# 0006: History store schema and detection import

Date: 2026-11-17. Status: proposed by Justin; needs Benoy's agreement, because the history
service and the assistant will read this schema.

## Context

[ADR 0002](0002-assistant-claude-sqlite-web.md) chose a single SQLite file for the history,
written by the tracker, opened read-only by the assistant, and changed only through ordered
migrations. [ADR 0005](0005-detector-output-file-for-the-tracker.md) defines the detector's
`SessionDetections` files. The tracker needs those detections in the history so it can
associate them across photos, and so each history traces back to the detector that produced
it (AGENTS.md).

Three facts shape the schema. Benoy keeps retraining, so the same photo can have detections
from several detector versions. A photo with no detections is evidence that nothing was found,
which tracking must keep distinct from a photo that was never processed. Association
thresholds will be tuned, so tracker output will be recomputed many times from the same
detections.

## Decision

- **Location:** `$OBJHIST_DATA_DIR/histories/history.sqlite` by default, as in
  [data/README.md](../../data/README.md).
- **Migrations:** numbered SQL files in `packages/tracking/src/objhist_tracking/migrations/`
  (`0001_<name>.sql`, `0002_<name>.sql`, ...), applied in order by
  `objhist_tracking.history_store.open_history_store`, each in its own transaction. Each
  applied migration is recorded in `schema_migrations` with its SHA-256. Opening a history
  whose recorded hash no longer matches the file, or that has a migration the code does not
  know, fails. Schema changes always go in a new migration; applied ones are never edited.
- **Two layers.** Detector output (migration `0001`) is imported once and never updated.
  Tracker output (migration `0002`) is stored separately, and every row belongs to one tracker
  run, so rerunning the tracker with other settings adds a run beside the old ones.
- **Detector-output tables** (migration `0001_detector_output`), all `STRICT` with foreign keys
  enforced:
  - `detector_runs`: one row per detector version, with weights SHA-256, class names,
    confidence threshold, inference settings, and framework version.
  - `sessions`: session ID and capture-time source (`manifest` or `exif`).
  - `frames`: one row per photo, shared by every detector that processed it. Capture time is
    stored as fixed-width ISO 8601 UTC text (`captured_at_utc`, so text order is time order)
    plus the photo's original `utc_offset_seconds` for display.
  - `processed_frames`: which detector version processed which photo. A processed frame with
    no detections means the detector ran and found nothing.
  - `detections`: one row per box, in pixels of the upright photo, keyed by
    `(detector_version, frame_id, detection_index)`, where the index is the detection's
    position in its frame's list in the file.
  - `session_imports`: one row per imported file, with a SHA-256 of its contents (excluding
    the file's `created_at`) and the import time.

  The tables after migration `0001`. `detections` links to `processed_frames` through the
  pair `(detector_version, frame_id)`; that pair plus `detection_index` is unique (`UK`).

  ```mermaid
  erDiagram
      detector_runs {
          TEXT version PK "e.g. yolo26s-...-57f3572d....-single1280"
          TEXT weights_sha256 "hash of the model weights"
          TEXT class_names_json "JSON list of class names"
          REAL confidence_threshold "0 to 1; lower detections not in file"
          TEXT settings_json "inference settings, sorted JSON"
          TEXT framework_version "Ultralytics version"
      }

      sessions {
          TEXT session_id PK "e.g. 2026-10-18-0930"
          TEXT capture_time_source "manifest or exif"
      }

      frames {
          TEXT frame_id PK "session_id/photo filename"
          TEXT session_id FK "links to sessions"
          TEXT captured_at_utc "fixed-width ISO 8601 UTC, sorts by time"
          INTEGER utc_offset_seconds "original offset, e.g. -14400"
          INTEGER image_width_px "greater than 0"
          INTEGER image_height_px "greater than 0"
      }

      processed_frames {
          TEXT detector_version PK, FK "links to detector_runs"
          TEXT frame_id PK, FK "links to frames"
      }

      detections {
          INTEGER detection_id PK "auto-numbered"
          TEXT detector_version FK, UK "with frame_id: links to processed_frames"
          TEXT frame_id FK, UK "with detector_version: links to processed_frames"
          INTEGER detection_index UK "position in the frame's list in the file"
          TEXT class_name "e.g. cup"
          REAL confidence "0 to 1"
          REAL x_min_px "at least 0"
          REAL y_min_px "at least 0"
          REAL x_max_px "greater than x_min_px"
          REAL y_max_px "greater than y_min_px"
      }

      session_imports {
          TEXT detector_version PK, FK "links to detector_runs"
          TEXT session_id PK, FK "links to sessions"
          TEXT content_sha256 "file contents without created_at"
          TEXT file_created_at "when the detector wrote the file"
          TEXT imported_at "when it was imported, UTC"
      }

      schema_migrations {
          INTEGER version PK "1, 2, 3, ..."
          TEXT name "e.g. detector_output"
          TEXT sha256 "hash of the migration file"
          TEXT applied_at "UTC"
      }

      sessions ||--|{ frames : "contains"
      detector_runs ||--o{ processed_frames : "processed"
      frames ||--o{ processed_frames : "processed by"
      processed_frames ||--o{ detections : "found (zero or more)"
      detector_runs ||--o{ session_imports : "imported from"
      sessions ||--o{ session_imports : "imported as"
  ```

- **Tracker-output tables** (migration `0002_tracker_output`), also `STRICT` with foreign keys
  enforced:
  - `tracker_runs`: one row per run of the tracker over one session's detections from one
    detector version. A composite foreign key to `session_imports` requires that detector's
    file for that session to be imported. It records the tracker version
    (`position-centre-v1` for the position-only baseline, `appearance-gated-v1` for
    appearance-first matching), the appearance model revision (`NULL` for the baseline), the
    confidence threshold applied, and the named association thresholds, weights, and removal
    rule as sorted JSON (AGENTS.md).
  - `objects`: one row per physical item the tracker believes exists in a run, with one class
    for its whole life; tracks of different classes are never merged.
  - `observations`: one row per detection assigned to an object, with the match evidence:
    position distance, appearance cosine similarity, and the combined match cost. The evidence
    is `NULL` on the observation that created the object. The key
    `(tracker_run_id, detection_id)` gives each detection at most one object per run. A photo
    with no observation of an object means it was not seen there, which alone is not a
    removal.
  - `events`: `added`, `removed`, or `moved`, each placed between two photos
    (`frame_before_id`, `frame_after_id`), never at one instant. For `added`, the before photo
    is the last one without the object, or `NULL` when it was already in the session's first
    photo and its placement time is unknown. For `removed`, it is the last photo the object
    was seen in, and the event is written only once the run's removal rule is met, so a missed
    detection or occlusion is not a removal. `removed` and `moved` always have both photos.
  - Observations and events reference their object through `(object_id, tracker_run_id)`, so
    neither can point at an object from another run.

  The tracker tables after migration `0002`, with the detector tables they reference shown by
  their keys only:

  ```mermaid
  erDiagram
      session_imports {
          TEXT detector_version PK, FK "see detector tables"
          TEXT session_id PK, FK "see detector tables"
      }

      tracker_runs {
          INTEGER tracker_run_id PK "auto-numbered"
          TEXT tracker_version "e.g. position-dinov2-v1"
          TEXT detector_version FK "with session_id: links to session_imports"
          TEXT session_id FK "with detector_version: links to session_imports"
          TEXT appearance_model "e.g. dinov2-vits14; NULL for position only"
          REAL confidence_threshold "0 to 1; at least the detector's"
          TEXT settings_json "thresholds, weights, removal rule"
          TEXT created_at "UTC"
      }

      objects {
          INTEGER object_id PK "auto-numbered"
          INTEGER tracker_run_id FK "links to tracker_runs"
          TEXT class_name "fixed for the object's life"
      }

      observations {
          INTEGER tracker_run_id PK, FK "with object_id: links to objects"
          INTEGER object_id FK "with tracker_run_id: links to objects"
          INTEGER detection_id PK, FK "links to detections"
          REAL position_distance "0 or more; NULL on first sighting"
          REAL appearance_similarity "-1 to 1; NULL without a model"
          REAL match_cost "lower is a better match"
      }

      events {
          INTEGER event_id PK "auto-numbered"
          INTEGER tracker_run_id FK "with object_id: links to objects"
          INTEGER object_id FK "with tracker_run_id: links to objects"
          TEXT event_type "added, removed, or moved"
          TEXT frame_before_id FK "NULL only for added: present from first photo"
          TEXT frame_after_id FK "differs from frame_before_id"
      }

      detections {
          INTEGER detection_id PK "see detector tables"
      }

      frames {
          TEXT frame_id PK "see detector tables"
      }

      session_imports ||--o{ tracker_runs : "tracked by"
      tracker_runs ||--o{ objects : "found"
      objects ||--|{ observations : "seen as"
      detections ||--o{ observations : "assigned in"
      objects ||--o{ events : "changed by"
      frames |o--o{ events : "before"
      frames ||--o{ events : "after"
  ```
- **Appearance embeddings** (migration `0003_detection_embeddings`), the cache that
  [ADR 0003](0003-appearance-embeddings-and-provided-dataset.md) calls for:
  `detection_embeddings` holds one unit-length float32 vector per detection and model
  revision (`embedding_model`, e.g. `facebook/dinov2-small@<commit>`), keyed by
  `(detection_id, embedding_model)`. Rows are never updated; another model or revision gets
  its own rows. `python -m objhist_tracking.embed_detections` fills it, and
  `track_session --appearance <model ID>` reads it. A tracker run records the exact revision
  in `tracker_runs.appearance_model`.

  ```mermaid
  erDiagram
      detections {
          INTEGER detection_id PK "see detector tables"
      }

      detection_embeddings {
          INTEGER detection_id PK, FK "links to detections"
          TEXT embedding_model PK "model ID @ exact revision"
          INTEGER dimensions "e.g. 384 for dinov2-small"
          BLOB embedding "little-endian float32, unit length"
          TEXT created_at "UTC"
      }

      detections ||--o{ detection_embeddings : "described by"
  ```

- **Import:** `python -m objhist_tracking.import_detections <files or folders>` imports each
  file in one transaction. Importing identical detections again changes nothing. The import
  stops with an error, leaving that file's data out, when:
  - a detector version and session already in the history arrive with different detections;
  - a detector version arrives with different weights, classes, threshold, or settings;
  - a photo arrives with a different capture time or image size;
  - a session arrives with a different capture-time source.

## Consequences

- Detections from several detector versions can live in one history, and every result can
  name the detector version it came from. Tracker runs must say which detector version they
  read.
- A detector rerun that produces different boxes must use a new version name; the history
  never silently replaces evidence. Benoy's version string (ADR 0005) already changes with the
  weights and inference mode.
- Readers find photos with no detections through `processed_frames`, not through
  `detections`.
- The tracker code must enforce what SQLite cannot check across tables: an event's before
  photo is earlier than its after photo, both photos and every observed detection belong to
  the run's session and detector version, and an object has at most one observation per
  photo. Tests cover these with the association code.
- The tracker tables are written before the association code exists, so their evidence
  columns may change. Until migration `0002` is merged it can be edited; after that, changes
  go in a new migration.
- The assistant and the history service depend on this schema. Changes to existing tables are
  a shared-interface change: add a migration, update the readers and their tests, and record
  it in a new ADR when the meaning of a table changes.
- `detection_index` is stable only within one imported file. If tracker events need to point
  at detections across re-exports, the detector file needs a stable detection ID, which would
  be a `format_version` 2 change to ADR 0005.
