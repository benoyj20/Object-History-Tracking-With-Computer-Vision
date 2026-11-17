"""Run the tracker on sessions in the SQLite history and store each run's results.

    uv run python -m objhist_tracking.track_session 2026-10-18-0930 [more sessions] \\
        [--appearance facebook/dinov2-small] [--align] [--detector <detector version>] \\
        [--min-confidence 0.4] [--history PATH]

Each session's processed photos and detections are read from the history (imported with
`objhist_tracking.import_detections`), tracked with `objhist_tracking.tracker`, and stored as
one tracker run: `tracker_runs`, `objects`, `observations`, and `events`, in one transaction.
Running again adds a new run beside the earlier ones; nothing is replaced.

`--detector` may be left out when a session was imported from exactly one detector version.
`--min-confidence` defaults to the detector's own threshold and cannot be lower, because
detections below it are not in the history. Without `--appearance` the tracker matches by
position only (the baseline); with it, by the embeddings cached for that model by
`objhist_tracking.embed_detections`, which must exist for every detection used. `--align`
compensates for camera movement between photos before matching, in either mode.
"""

import argparse
import json
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from objhist_core import BoundingBox
from objhist_tracking.embedding_store import (
    EmbeddingError,
    embedding_model_for,
    load_embeddings,
)
from objhist_tracking.history_store import (
    HistoryStoreError,
    default_history_path,
    open_history_store,
)
from objhist_tracking.tracker import (
    Embedding,
    TrackerDetection,
    TrackerFrame,
    TrackerInputError,
    TrackerSettings,
    TrackingResult,
    default_settings,
    track_session,
)


class TrackSessionError(ValueError):
    """A session cannot be tracked as asked."""


@dataclass(frozen=True)
class TrackerRunSummary:
    tracker_run_id: int
    session_id: str
    detector_version: str
    frame_count: int
    object_count: int
    observation_count: int
    added_count: int
    removed_count: int
    aligned_photo_count: int | None
    """Photos whose camera movement was compensated; None without alignment."""


def imported_detector_versions(connection: sqlite3.Connection, session_id: str) -> list[str]:
    rows = connection.execute(
        "SELECT detector_version FROM session_imports WHERE session_id = ? "
        "ORDER BY detector_version",
        (session_id,),
    )
    return [version for (version,) in rows]


def resolve_detector_version(
    connection: sqlite3.Connection, session_id: str, detector_version: str | None
) -> str:
    """The detector to track with: the one asked for, or the session's only one."""
    versions = imported_detector_versions(connection, session_id)
    if not versions:
        raise TrackSessionError(
            f"session {session_id} has no imported detections; import them first with "
            "python -m objhist_tracking.import_detections"
        )
    if detector_version is not None:
        if detector_version not in versions:
            raise TrackSessionError(
                f"session {session_id} has no detections from {detector_version}; "
                f"imported: {', '.join(versions)}"
            )
        return detector_version
    if len(versions) > 1:
        raise TrackSessionError(
            f"session {session_id} was imported from several detectors; choose one with "
            f"--detector: {', '.join(versions)}"
        )
    return versions[0]


def detector_confidence_threshold(connection: sqlite3.Connection, detector_version: str) -> float:
    row = connection.execute(
        "SELECT confidence_threshold FROM detector_runs WHERE version = ?", (detector_version,)
    ).fetchone()
    if row is None:
        raise TrackSessionError(f"detector {detector_version} is not in the history")
    return float(row[0])


def load_session_frames(
    connection: sqlite3.Connection,
    session_id: str,
    detector_version: str,
    embeddings: dict[int, Embedding] | None = None,
) -> tuple[TrackerFrame, ...]:
    """Every photo of the session the detector processed, in capture order, with its boxes
    and, when given, each detection's embedding."""
    embeddings = embeddings or {}
    frame_rows = connection.execute(
        "SELECT f.frame_id, f.captured_at_utc, f.image_width_px, f.image_height_px "
        "FROM frames f JOIN processed_frames p ON p.frame_id = f.frame_id "
        "WHERE f.session_id = ? AND p.detector_version = ? "
        "ORDER BY f.captured_at_utc",
        (session_id, detector_version),
    ).fetchall()
    detections: dict[str, list[TrackerDetection]] = {row[0]: [] for row in frame_rows}
    detection_rows = connection.execute(
        "SELECT d.frame_id, d.detection_id, d.class_name, d.confidence, "
        "d.x_min_px, d.y_min_px, d.x_max_px, d.y_max_px "
        "FROM detections d JOIN frames f ON f.frame_id = d.frame_id "
        "WHERE f.session_id = ? AND d.detector_version = ? "
        "ORDER BY d.frame_id, d.detection_index",
        (session_id, detector_version),
    )
    for (
        frame_id,
        detection_id,
        class_name,
        confidence,
        x_min,
        y_min,
        x_max,
        y_max,
    ) in detection_rows:
        detections[frame_id].append(
            TrackerDetection(
                detection_id=detection_id,
                class_name=class_name,
                confidence=confidence,
                box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
                embedding=embeddings.get(detection_id),
            )
        )
    return tuple(
        TrackerFrame(
            frame_id=frame_id,
            captured_at=datetime.fromisoformat(captured_at_utc),
            image_width_px=width_px,
            image_height_px=height_px,
            detections=tuple(detections[frame_id]),
        )
        for frame_id, captured_at_utc, width_px, height_px in frame_rows
    )


def run_tracker(
    connection: sqlite3.Connection,
    session_id: str,
    detector_version: str | None = None,
    min_confidence: float | None = None,
    appearance_model_id: str | None = None,
    align: bool = False,
) -> TrackerRunSummary:
    """Track one session and store the run; `min_confidence` defaults to the detector's.

    With `appearance_model_id` (e.g. 'facebook/dinov2-small'), match by the session's cached
    embeddings from that model; without it, by position only. With `align`, compensate for
    camera movement before matching."""
    detector_version = resolve_detector_version(connection, session_id, detector_version)
    detector_threshold = detector_confidence_threshold(connection, detector_version)
    if min_confidence is None:
        min_confidence = detector_threshold
    elif min_confidence < detector_threshold:
        raise TrackSessionError(
            f"detections from {detector_version} were kept only at confidence "
            f"{detector_threshold} or above, so the tracker cannot use {min_confidence}"
        )
    embedding_model: str | None = None
    embeddings: dict[int, Embedding] | None = None
    if appearance_model_id is not None:
        embedding_model = embedding_model_for(connection, appearance_model_id, session_id)
        embeddings = load_embeddings(connection, session_id, detector_version, embedding_model)
    settings = default_settings(min_confidence, embedding_model, align)
    frames = load_session_frames(connection, session_id, detector_version, embeddings)
    result = track_session(frames, settings)
    tracker_run_id = _save_run(connection, session_id, detector_version, settings, result)
    return TrackerRunSummary(
        tracker_run_id=tracker_run_id,
        session_id=session_id,
        detector_version=detector_version,
        frame_count=len(frames),
        object_count=len(result.objects),
        observation_count=len(result.observations),
        added_count=sum(event.event_type == "added" for event in result.events),
        removed_count=sum(event.event_type == "removed" for event in result.events),
        aligned_photo_count=(
            None
            if settings.alignment is None
            else sum(a.transform is not None for a in result.alignments)
        ),
    )


def _save_run(
    connection: sqlite3.Connection,
    session_id: str,
    detector_version: str,
    settings: TrackerSettings,
    result: TrackingResult,
) -> int:
    with connection:
        cursor = connection.execute(
            "INSERT INTO tracker_runs (tracker_version, detector_version, session_id, "
            "appearance_model, confidence_threshold, settings_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                settings.tracker_version,
                detector_version,
                session_id,
                None if settings.appearance is None else settings.appearance.embedding_model,
                settings.min_confidence,
                json.dumps(settings.as_record(), sort_keys=True),
                datetime.now(UTC).isoformat(timespec="microseconds"),
            ),
        )
        tracker_run_id = _row_id(cursor)
        object_ids: dict[int, int] = {}
        for tracked in result.objects:
            cursor = connection.execute(
                "INSERT INTO objects (tracker_run_id, class_name) VALUES (?, ?)",
                (tracker_run_id, tracked.class_name),
            )
            object_ids[tracked.object_key] = _row_id(cursor)
        connection.executemany(
            "INSERT INTO observations (tracker_run_id, object_id, detection_id, "
            "position_distance, appearance_similarity, match_cost) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    tracker_run_id,
                    object_ids[observation.object_key],
                    observation.detection_id,
                    observation.position_distance,
                    observation.appearance_similarity,
                    observation.match_cost,
                )
                for observation in result.observations
            ],
        )
        connection.executemany(
            "INSERT INTO events (tracker_run_id, object_id, event_type, frame_before_id, "
            "frame_after_id) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    tracker_run_id,
                    object_ids[event.object_key],
                    event.event_type,
                    event.frame_before_id,
                    event.frame_after_id,
                )
                for event in result.events
            ],
        )
    return tracker_run_id


def _row_id(cursor: sqlite3.Cursor) -> int:
    if cursor.lastrowid is None:
        raise sqlite3.DatabaseError("SQLite did not return the new row's ID")
    return cursor.lastrowid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="+", help="session IDs to track")
    parser.add_argument(
        "--detector", help="detector version (default: the session's only imported one)"
    )
    parser.add_argument(
        "--appearance",
        metavar="MODEL_ID",
        help="match by cached embeddings from this model, e.g. facebook/dinov2-small "
        "(default: position only)",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="compensate for camera movement between photos before matching",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        help="ignore detections below this (default: the detector's own threshold)",
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="SQLite history file (default: $OBJHIST_DATA_DIR/histories/history.sqlite)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    history_path: Path = args.history or default_history_path()
    if not history_path.is_file():
        print(
            f"error: no history at {history_path}; import detections first with "
            "python -m objhist_tracking.import_detections",
            file=sys.stderr,
        )
        return 1
    try:
        connection = open_history_store(history_path)
    except (HistoryStoreError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        for session_id in args.sessions:
            summary = run_tracker(
                connection,
                session_id,
                args.detector,
                args.min_confidence,
                args.appearance,
                args.align,
            )
            aligned = (
                ""
                if summary.aligned_photo_count is None
                else f", camera movement compensated in {summary.aligned_photo_count} photos"
            )
            print(
                f"tracker run {summary.tracker_run_id}: {summary.session_id} from "
                f"{summary.detector_version}, {summary.frame_count} photos, "
                f"{summary.object_count} objects, {summary.added_count} added, "
                f"{summary.removed_count} removed{aligned}"
            )
    except (TrackSessionError, TrackerInputError, EmbeddingError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        print("runs listed above were stored; the failing session stored nothing", file=sys.stderr)
        return 1
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
