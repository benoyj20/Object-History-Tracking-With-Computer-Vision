"""Constraints of the tracker-output tables (migration 0002), on a tiny hand-built history."""

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from objhist_tracking.history_store import open_history_store

DETECTOR = "test-detector-000000000000-single1280"
SESSION = "2026-10-18-0930"
FRAME_A = f"{SESSION}/a.jpg"
FRAME_B = f"{SESSION}/b.jpg"


@pytest.fixture
def history(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    """One imported session: a cup in photo a, a cup in photo b, and tracker run 1."""
    connection = open_history_store(tmp_path / "history.sqlite")
    with connection:
        connection.execute(
            "INSERT INTO detector_runs VALUES (?, ?, '[\"cup\"]', 0.25, '{}', 'test')",
            (DETECTOR, "0" * 64),
        )
        connection.execute("INSERT INTO sessions VALUES (?, 'exif')", (SESSION,))
        for frame_id, time in (
            (FRAME_A, "2026-10-18T13:30:00.000000+00:00"),
            (FRAME_B, "2026-10-18T13:45:00.000000+00:00"),
        ):
            connection.execute(
                "INSERT INTO frames VALUES (?, ?, ?, -14400, 400, 300)", (frame_id, SESSION, time)
            )
            connection.execute("INSERT INTO processed_frames VALUES (?, ?)", (DETECTOR, frame_id))
            connection.execute(
                "INSERT INTO detections (detector_version, frame_id, detection_index, class_name, "
                "confidence, x_min_px, y_min_px, x_max_px, y_max_px) "
                "VALUES (?, ?, 0, 'cup', 0.9, 10, 20, 110, 140)",
                (DETECTOR, frame_id),
            )
        connection.execute(
            "INSERT INTO session_imports VALUES (?, ?, 'x', '2026-10-28T14:00:00+00:00', "
            "'2026-11-17T00:00:00+00:00')",
            (DETECTOR, SESSION),
        )
        connection.execute(
            "INSERT INTO tracker_runs (tracker_run_id, tracker_version, detector_version, "
            "session_id, appearance_model, confidence_threshold, settings_json, created_at) "
            "VALUES (1, 'test-tracker-v1', ?, ?, NULL, 0.25, '{}', '2026-11-17T00:00:00+00:00')",
            (DETECTOR, SESSION),
        )
        connection.execute("INSERT INTO objects VALUES (1, 1, 'cup')")
    yield connection
    connection.close()


def detection_id(connection: sqlite3.Connection, frame_id: str) -> int:
    return connection.execute(
        "SELECT detection_id FROM detections WHERE frame_id = ?", (frame_id,)
    ).fetchone()[0]


def add_event(
    connection: sqlite3.Connection, event_type: str, before: str | None, after: str
) -> None:
    with connection:
        connection.execute(
            "INSERT INTO events (tracker_run_id, object_id, event_type, frame_before_id, "
            "frame_after_id) VALUES (1, 1, ?, ?, ?)",
            (event_type, before, after),
        )


def test_tracker_run_needs_an_imported_session(history: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError), history:
        history.execute(
            "INSERT INTO tracker_runs (tracker_version, detector_version, session_id, "
            "confidence_threshold, settings_json, created_at) "
            "VALUES ('v1', ?, 'never-imported', 0.25, '{}', 'now')",
            (DETECTOR,),
        )


def test_one_object_is_observed_across_photos(history: sqlite3.Connection) -> None:
    with history:
        history.execute(
            "INSERT INTO observations (tracker_run_id, object_id, detection_id) VALUES (1, 1, ?)",
            (detection_id(history, FRAME_A),),
        )
        history.execute(
            "INSERT INTO observations VALUES (1, 1, ?, 4.0, 0.93, 0.12)",
            (detection_id(history, FRAME_B),),
        )
    rows = history.execute(
        "SELECT d.frame_id FROM observations o JOIN detections d USING (detection_id) "
        "WHERE o.object_id = 1 ORDER BY d.frame_id"
    ).fetchall()
    assert rows == [(FRAME_A,), (FRAME_B,)]


def test_a_detection_belongs_to_one_object_per_run(history: sqlite3.Connection) -> None:
    detection = detection_id(history, FRAME_A)
    with history:
        history.execute("INSERT INTO objects VALUES (2, 1, 'cup')")
        history.execute(
            "INSERT INTO observations (tracker_run_id, object_id, detection_id) VALUES (1, 1, ?)",
            (detection,),
        )
    with pytest.raises(sqlite3.IntegrityError), history:
        history.execute(
            "INSERT INTO observations (tracker_run_id, object_id, detection_id) VALUES (1, 2, ?)",
            (detection,),
        )


def test_observation_object_must_belong_to_its_run(history: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError), history:
        history.execute(
            "INSERT INTO observations (tracker_run_id, object_id, detection_id) VALUES (2, 1, ?)",
            (detection_id(history, FRAME_A),),
        )


def test_added_event_may_have_an_unknown_start(history: sqlite3.Connection) -> None:
    add_event(history, "added", None, FRAME_A)
    add_event(history, "added", FRAME_A, FRAME_B)

    assert history.execute("SELECT count(*) FROM events").fetchone()[0] == 2


@pytest.mark.parametrize("event_type", ["removed", "moved"])
def test_removed_and_moved_need_both_photos(history: sqlite3.Connection, event_type: str) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        add_event(history, event_type, None, FRAME_B)


def test_event_interval_needs_two_different_photos(history: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        add_event(history, "moved", FRAME_A, FRAME_A)


def test_unknown_event_type_is_refused(history: sqlite3.Connection) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        add_event(history, "teleported", FRAME_A, FRAME_B)
