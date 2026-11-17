"""Import detector output files into the SQLite object history.

    uv run python -m objhist_tracking.import_detections \\
        data/interim/detections/<detector version> [more folders or files]

Each argument is a `SessionDetections` file from `python -m objhist_detection.detect`
(ADR 0005) or a folder of them. The history is `$OBJHIST_DATA_DIR/histories/history.sqlite`
unless `--history` names another file.

Every frame is stored, including frames without detections, so "the detector ran and found
nothing" stays distinct from "never processed". Each file is imported in one transaction.
Importing the same detections again changes nothing. Different detections for a detector
version and session that are already in the history are an error: imported detections are
evidence and are never replaced; a changed detector must have a new version.
"""

import argparse
import hashlib
import json
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from objhist_core import DetectorRun, FrameDetections, SessionDetections
from objhist_tracking.history_store import (
    HistoryStoreError,
    default_history_path,
    open_history_store,
)

ImportOutcome = Literal["imported", "already imported"]


class DetectionImportError(ValueError):
    """A detection file cannot be imported without contradicting the history."""


@dataclass(frozen=True)
class SessionImport:
    session_id: str
    detector_version: str
    outcome: ImportOutcome
    frame_count: int
    detection_count: int


def read_session_file(path: Path) -> SessionDetections:
    try:
        return SessionDetections.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise DetectionImportError(f"{path} is not a valid detection file: {error}") from error


def detection_files(sources: Sequence[Path]) -> list[Path]:
    """Expand folders to their `*.json` files; a folder without any is an error."""
    paths: list[Path] = []
    for source in sources:
        if source.is_dir():
            found = sorted(source.glob("*.json"))
            if not found:
                raise DetectionImportError(f"no detection files in {source}")
            paths.extend(found)
        elif source.is_file():
            paths.append(source)
        else:
            raise DetectionImportError(f"{source} does not exist")
    return paths


def content_sha256(session: SessionDetections) -> str:
    """Hash of everything imported from the file, leaving out when the file was written."""
    content = session.model_dump(mode="json", exclude={"created_at"})
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def import_session(connection: sqlite3.Connection, session: SessionDetections) -> SessionImport:
    """Store one session's detections in a single transaction."""
    detector_version = session.detector.version
    detection_count = sum(len(frame.detections) for frame in session.frames)
    digest = content_sha256(session)
    with connection:
        previous = connection.execute(
            "SELECT content_sha256 FROM session_imports "
            "WHERE detector_version = ? AND session_id = ?",
            (detector_version, session.session_id),
        ).fetchone()
        if previous is not None:
            if previous[0] != digest:
                raise DetectionImportError(
                    f"session {session.session_id} from detector {detector_version} is already "
                    "in the history with different detections; imported detections are not "
                    "replaced, so give a changed detector a new version"
                )
            outcome: ImportOutcome = "already imported"
        else:
            _store_detector_run(connection, session.detector)
            _store_session(connection, session)
            for frame in session.frames:
                _store_frame(connection, session.session_id, frame)
                _store_frame_detections(connection, frame)
            connection.execute(
                "INSERT INTO session_imports (detector_version, session_id, content_sha256, "
                "file_created_at, imported_at) VALUES (?, ?, ?, ?, ?)",
                (
                    detector_version,
                    session.session_id,
                    digest,
                    session.created_at.isoformat(),
                    datetime.now(UTC).isoformat(timespec="microseconds"),
                ),
            )
            outcome = "imported"
    return SessionImport(
        session.session_id, detector_version, outcome, len(session.frames), detection_count
    )


def utc_timestamp(moment: datetime) -> str:
    """Fixed-width UTC ISO 8601 text, so sorting the text sorts by time."""
    return moment.astimezone(UTC).isoformat(timespec="microseconds")


def utc_offset_seconds(moment: datetime) -> int:
    offset = moment.utcoffset()
    if offset is None:
        # SessionDetections only accepts timezone-aware times; this guards direct callers.
        raise DetectionImportError(f"capture time {moment} has no UTC offset")
    return int(offset.total_seconds())


def _store_detector_run(connection: sqlite3.Connection, detector: DetectorRun) -> None:
    row = (
        detector.weights_sha256,
        json.dumps(list(detector.class_names)),
        detector.confidence_threshold,
        json.dumps(detector.settings, sort_keys=True),
        detector.framework_version,
    )
    existing = connection.execute(
        "SELECT weights_sha256, class_names_json, confidence_threshold, settings_json, "
        "framework_version FROM detector_runs WHERE version = ?",
        (detector.version,),
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO detector_runs (version, weights_sha256, class_names_json, "
            "confidence_threshold, settings_json, framework_version) VALUES (?, ?, ?, ?, ?, ?)",
            (detector.version, *row),
        )
    elif tuple(existing) != row:
        raise DetectionImportError(
            f"detector {detector.version} is already in the history with different weights, "
            "classes, threshold, or settings; one version must mean one configuration"
        )


def _store_session(connection: sqlite3.Connection, session: SessionDetections) -> None:
    existing = connection.execute(
        "SELECT capture_time_source FROM sessions WHERE session_id = ?", (session.session_id,)
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO sessions (session_id, capture_time_source) VALUES (?, ?)",
            (session.session_id, session.capture_time_source),
        )
    elif existing[0] != session.capture_time_source:
        raise DetectionImportError(
            f"session {session.session_id} is already in the history with capture times from "
            f"{existing[0]}, but this file uses {session.capture_time_source}; one session "
            "must use one clock"
        )


def _store_frame(connection: sqlite3.Connection, session_id: str, frame: FrameDetections) -> None:
    row = (
        session_id,
        utc_timestamp(frame.captured_at),
        utc_offset_seconds(frame.captured_at),
        frame.image_width_px,
        frame.image_height_px,
    )
    existing = connection.execute(
        "SELECT session_id, captured_at_utc, utc_offset_seconds, image_width_px, "
        "image_height_px FROM frames WHERE frame_id = ?",
        (frame.frame_id,),
    ).fetchone()
    if existing is None:
        connection.execute(
            "INSERT INTO frames (frame_id, session_id, captured_at_utc, utc_offset_seconds, "
            "image_width_px, image_height_px) VALUES (?, ?, ?, ?, ?, ?)",
            (frame.frame_id, *row),
        )
    elif tuple(existing) != row:
        raise DetectionImportError(
            f"frame {frame.frame_id} is already in the history with a different capture time "
            "or image size, from another detection file; check both files' capture times"
        )


def _store_frame_detections(connection: sqlite3.Connection, frame: FrameDetections) -> None:
    connection.execute(
        "INSERT INTO processed_frames (detector_version, frame_id) VALUES (?, ?)",
        (frame.detector_version, frame.frame_id),
    )
    connection.executemany(
        "INSERT INTO detections (detector_version, frame_id, detection_index, class_name, "
        "confidence, x_min_px, y_min_px, x_max_px, y_max_px) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                frame.detector_version,
                frame.frame_id,
                index,
                detection.class_name,
                detection.confidence,
                detection.box.x_min,
                detection.box.y_min,
                detection.box.x_max,
                detection.box.y_max,
            )
            for index, detection in enumerate(frame.detections)
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "detections", nargs="+", type=Path, help="detection files or folders of them"
    )
    parser.add_argument(
        "--history",
        type=Path,
        help="SQLite history file (default: $OBJHIST_DATA_DIR/histories/history.sqlite)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    history_path: Path = args.history or default_history_path()
    try:
        paths = detection_files(args.detections)
        connection = open_history_store(history_path)
    except (DetectionImportError, HistoryStoreError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        for path in paths:
            result = import_session(connection, read_session_file(path))
            print(
                f"{result.outcome}: {result.session_id} from {result.detector_version} "
                f"({result.frame_count} frames, {result.detection_count} detections)"
            )
    except (DetectionImportError, OSError, sqlite3.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        print("files listed above were imported; the failing file changed nothing", file=sys.stderr)
        return 1
    finally:
        connection.close()
    print(f"history: {history_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
