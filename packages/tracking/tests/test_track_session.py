"""Tracking sessions stored in a real (temporary) history, built with the import code."""

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from objhist_core import BoundingBox, Detection, DetectorRun, FrameDetections, SessionDetections
from objhist_tracking.history_store import open_history_store
from objhist_tracking.import_detections import import_session
from objhist_tracking.track_session import TrackSessionError, main, run_tracker

SESSION_ID = "2026-10-18-0930"
DETECTOR = "test-detector-000000000000-single1280"
START = datetime(2026, 10, 18, 13, 30, tzinfo=UTC)


def detector(version: str = DETECTOR) -> DetectorRun:
    return DetectorRun(
        version=version,
        weights_sha256="0" * 64,
        class_names=("cup", "fork"),
        confidence_threshold=0.25,
        settings={"imgsz": 1280},
        framework_version="test",
    )


def cup(x_px: float, confidence: float = 0.9) -> Detection:
    box = BoundingBox(x_min=x_px - 20, y_min=480, x_max=x_px + 20, y_max=520)
    return Detection(class_name="cup", confidence=confidence, box=box)


def session(photos: list[list[Detection]], run: DetectorRun | None = None) -> SessionDetections:
    """One photo per list, 15 minutes apart, in a 1000 x 1000 px frame."""
    run = run or detector()
    return SessionDetections(
        session_id=SESSION_ID,
        created_at=START,
        capture_time_source="exif",
        detector=run,
        frames=tuple(
            FrameDetections(
                frame_id=f"{SESSION_ID}/f{index}.jpg",
                captured_at=START + timedelta(minutes=15 * index),
                image_width_px=1000,
                image_height_px=1000,
                detector_version=run.version,
                detections=tuple(detections),
            )
            for index, detections in enumerate(photos)
        ),
    )


# A cup there from the start and taken away after f1; a second cup put down in f1 that stays.
CUP_TAKEN_AND_CUP_ADDED = [[cup(300)], [cup(305), cup(700)], [cup(700)], [cup(700)]]


@pytest.fixture
def history(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = open_history_store(tmp_path / "history.sqlite")
    yield connection
    connection.close()


def frame(index: int) -> str:
    return f"{SESSION_ID}/f{index}.jpg"


def test_run_stores_objects_observations_and_events(history: sqlite3.Connection) -> None:
    import_session(history, session(CUP_TAKEN_AND_CUP_ADDED))

    summary = run_tracker(history, SESSION_ID)

    assert (summary.frame_count, summary.object_count, summary.observation_count) == (4, 2, 5)
    assert (summary.added_count, summary.removed_count) == (2, 1)
    events = history.execute(
        "SELECT o.object_id, e.event_type, e.frame_before_id, e.frame_after_id "
        "FROM events e JOIN objects o USING (object_id) ORDER BY e.event_id"
    ).fetchall()
    first, second = (row[0] for row in history.execute("SELECT object_id FROM objects"))
    assert events == [
        (first, "added", None, frame(0)),
        (second, "added", frame(0), frame(1)),
        (first, "removed", frame(1), frame(2)),
    ]


def test_run_records_its_configuration(history: sqlite3.Connection) -> None:
    import_session(history, session(CUP_TAKEN_AND_CUP_ADDED))

    summary = run_tracker(history, SESSION_ID)

    row = history.execute(
        "SELECT tracker_version, detector_version, appearance_model, confidence_threshold, "
        "settings_json FROM tracker_runs WHERE tracker_run_id = ?",
        (summary.tracker_run_id,),
    ).fetchone()
    assert row[:4] == ("position-centre-v1", DETECTOR, None, 0.25)
    assert json.loads(row[4])["removal_missing_frames"] == 2


def test_observations_carry_position_evidence(history: sqlite3.Connection) -> None:
    import_session(history, session([[cup(300)], [cup(340)]]))

    run_tracker(history, SESSION_ID)

    evidence = history.execute(
        "SELECT position_distance, appearance_similarity, match_cost FROM observations "
        "JOIN detections USING (detection_id) ORDER BY frame_id"
    ).fetchall()
    assert evidence[0] == (None, None, None)
    assert evidence[1] == (pytest.approx(0.04), None, pytest.approx(0.04))


def test_rerunning_adds_a_run_beside_the_first(history: sqlite3.Connection) -> None:
    import_session(history, session(CUP_TAKEN_AND_CUP_ADDED))

    first = run_tracker(history, SESSION_ID)
    second = run_tracker(history, SESSION_ID)

    assert first.tracker_run_id != second.tracker_run_id
    counts = history.execute(
        "SELECT tracker_run_id, count(*) FROM events GROUP BY tracker_run_id"
    ).fetchall()
    assert counts == [(first.tracker_run_id, 3), (second.tracker_run_id, 3)]


def test_photo_with_no_detections_counts_as_a_missing_photo(history: sqlite3.Connection) -> None:
    import_session(history, session([[cup(300)], [], [cup(300)]]))

    summary = run_tracker(history, SESSION_ID)

    assert (summary.frame_count, summary.object_count, summary.removed_count) == (3, 1, 0)


def test_higher_confidence_threshold_ignores_weak_detections(history: sqlite3.Connection) -> None:
    import_session(history, session([[cup(300), cup(700, confidence=0.3)]]))

    summary = run_tracker(history, SESSION_ID, min_confidence=0.5)

    assert summary.object_count == 1
    stored = history.execute("SELECT confidence_threshold FROM tracker_runs").fetchone()
    assert stored == (0.5,)


def test_threshold_below_the_detectors_is_refused(history: sqlite3.Connection) -> None:
    import_session(history, session([[cup(300)]]))

    with pytest.raises(TrackSessionError, match="cannot use 0.1"):
        run_tracker(history, SESSION_ID, min_confidence=0.1)


def test_session_never_imported_is_refused(history: sqlite3.Connection) -> None:
    with pytest.raises(TrackSessionError, match="no imported detections"):
        run_tracker(history, SESSION_ID)


def test_two_imported_detectors_need_a_choice(history: sqlite3.Connection) -> None:
    import_session(history, session([[cup(300)]]))
    other = "test-detector-111111111111-tiled"
    import_session(history, session([[cup(300), cup(700)]], detector(other)))

    with pytest.raises(TrackSessionError, match="choose one with --detector"):
        run_tracker(history, SESSION_ID)
    assert run_tracker(history, SESSION_ID, other).object_count == 2


def test_unknown_detector_is_refused(history: sqlite3.Connection) -> None:
    import_session(history, session([[cup(300)]]))

    with pytest.raises(TrackSessionError, match="no detections from"):
        run_tracker(history, SESSION_ID, "never-imported")


def test_command_tracks_a_session(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    history_path = tmp_path / "history.sqlite"
    connection = open_history_store(history_path)
    import_session(connection, session(CUP_TAKEN_AND_CUP_ADDED))
    connection.close()

    assert main([SESSION_ID, "--history", str(history_path)]) == 0
    assert "2 objects, 2 added, 1 removed" in capsys.readouterr().out


def test_command_without_a_history_does_not_create_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    history_path = tmp_path / "missing.sqlite"

    assert main([SESSION_ID, "--history", str(history_path)]) == 1
    assert "no history" in capsys.readouterr().err
    assert not history_path.exists()


def test_command_reports_an_unknown_session(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    history_path = tmp_path / "history.sqlite"
    open_history_store(history_path).close()

    assert main(["no-such-session", "--history", str(history_path)]) == 1
    assert "no imported detections" in capsys.readouterr().err
