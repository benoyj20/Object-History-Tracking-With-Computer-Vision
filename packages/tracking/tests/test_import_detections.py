import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from objhist_core import BoundingBox, Detection, DetectorRun, FrameDetections, SessionDetections
from objhist_tracking.history_store import open_history_store
from objhist_tracking.import_detections import (
    DetectionImportError,
    import_session,
    main,
)

SESSION_ID = "2026-10-18-0930"
EASTERN_DAYLIGHT = timezone(timedelta(hours=-4))
START = datetime(2026, 10, 18, 9, 30, tzinfo=EASTERN_DAYLIGHT)
CREATED_AT = datetime(2026, 10, 28, 14, 0, tzinfo=UTC)


def detector(
    version: str = "test-detector-000000000000-single1280", confidence_threshold: float = 0.25
) -> DetectorRun:
    return DetectorRun(
        version=version,
        weights_sha256="0" * 64,
        class_names=("cup", "fork"),
        confidence_threshold=confidence_threshold,
        settings={"imgsz": 1280, "tiled": False},
        framework_version="test",
    )


def cup(confidence: float = 0.9) -> Detection:
    box = BoundingBox(x_min=10, y_min=20, x_max=110, y_max=140)
    return Detection(class_name="cup", confidence=confidence, box=box)


def session(
    run: DetectorRun | None = None,
    *,
    session_id: str = SESSION_ID,
    cup_confidence: float = 0.9,
    created_at: datetime = CREATED_AT,
    start: datetime = START,
) -> SessionDetections:
    """Two photos: a cup on the counter, then an empty counter."""
    run = run or detector()
    frames = (
        FrameDetections(
            frame_id=f"{session_id}/a.jpg",
            captured_at=start,
            image_width_px=400,
            image_height_px=300,
            detector_version=run.version,
            detections=(cup(cup_confidence),),
        ),
        FrameDetections(
            frame_id=f"{session_id}/b.jpg",
            captured_at=start + timedelta(minutes=15),
            image_width_px=400,
            image_height_px=300,
            detector_version=run.version,
        ),
    )
    return SessionDetections(
        session_id=session_id,
        created_at=created_at,
        capture_time_source="exif",
        detector=run,
        frames=frames,
    )


@pytest.fixture
def history(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = open_history_store(tmp_path / "history.sqlite")
    yield connection
    connection.close()


def count(connection: sqlite3.Connection, table: str) -> int:
    # Table names come from the tests themselves, never from input.
    return connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_import_stores_frames_detections_and_detector(history: sqlite3.Connection) -> None:
    result = import_session(history, session())

    assert result.outcome == "imported"
    assert (result.frame_count, result.detection_count) == (2, 1)
    frames = history.execute(
        "SELECT frame_id, captured_at_utc, utc_offset_seconds FROM frames ORDER BY captured_at_utc"
    ).fetchall()
    assert frames == [
        (f"{SESSION_ID}/a.jpg", "2026-10-18T13:30:00.000000+00:00", -4 * 3600),
        (f"{SESSION_ID}/b.jpg", "2026-10-18T13:45:00.000000+00:00", -4 * 3600),
    ]
    detection = history.execute(
        "SELECT frame_id, detection_index, class_name, confidence, "
        "x_min_px, y_min_px, x_max_px, y_max_px FROM detections"
    ).fetchall()
    assert detection == [(f"{SESSION_ID}/a.jpg", 0, "cup", 0.9, 10.0, 20.0, 110.0, 140.0)]
    stored_run = history.execute(
        "SELECT confidence_threshold, class_names_json FROM detector_runs"
    ).fetchone()
    assert stored_run == (0.25, '["cup", "fork"]')


def test_frame_without_detections_is_recorded_as_processed(history: sqlite3.Connection) -> None:
    import_session(history, session())

    processed = history.execute(
        "SELECT p.frame_id, count(d.detection_id) FROM processed_frames p "
        "LEFT JOIN detections d USING (detector_version, frame_id) "
        "GROUP BY p.frame_id ORDER BY p.frame_id"
    ).fetchall()
    assert processed == [(f"{SESSION_ID}/a.jpg", 1), (f"{SESSION_ID}/b.jpg", 0)]


def test_reimporting_the_same_detections_changes_nothing(history: sqlite3.Connection) -> None:
    import_session(history, session())
    rerun = session(created_at=CREATED_AT + timedelta(days=1))

    result = import_session(history, rerun)

    assert result.outcome == "already imported"
    assert count(history, "detections") == 1
    assert count(history, "session_imports") == 1


def test_different_detections_for_an_imported_session_are_refused(
    history: sqlite3.Connection,
) -> None:
    import_session(history, session())

    with pytest.raises(DetectionImportError, match="different detections"):
        import_session(history, session(cup_confidence=0.5))
    assert history.execute("SELECT confidence FROM detections").fetchall() == [(0.9,)]


def test_a_second_detector_version_is_stored_alongside(history: sqlite3.Connection) -> None:
    import_session(history, session())
    import_session(history, session(detector("test-detector-111111111111-tiled")))

    assert count(history, "frames") == 2
    assert count(history, "processed_frames") == 4
    assert count(history, "detections") == 2


def test_one_detector_version_cannot_mean_two_configurations(
    history: sqlite3.Connection,
) -> None:
    import_session(history, session())
    other_session = session(detector(confidence_threshold=0.5), session_id="2026-10-18-1200")

    with pytest.raises(DetectionImportError, match="one version must mean one configuration"):
        import_session(history, other_session)
    assert count(history, "sessions") == 1


def test_conflicting_capture_times_roll_back_the_whole_file(history: sqlite3.Connection) -> None:
    import_session(history, session())
    shifted = session(
        detector("test-detector-111111111111-tiled"), start=START + timedelta(seconds=1)
    )

    with pytest.raises(DetectionImportError, match="different capture time"):
        import_session(history, shifted)
    assert count(history, "detector_runs") == 1
    assert count(history, "processed_frames") == 2


def test_command_imports_a_folder_and_reports_each_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = tmp_path / "detections"
    folder.mkdir()
    (folder / f"{SESSION_ID}.json").write_text(session().model_dump_json(), encoding="utf-8")
    history_path = tmp_path / "histories" / "history.sqlite"

    assert main([str(folder), "--history", str(history_path)]) == 0
    assert main([str(folder), "--history", str(history_path)]) == 0

    output = capsys.readouterr().out
    assert f"imported: {SESSION_ID}" in output
    assert f"already imported: {SESSION_ID}" in output


def test_command_rejects_a_folder_without_detection_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "empty").mkdir()

    assert main([str(tmp_path / "empty"), "--history", str(tmp_path / "history.sqlite")]) == 1
    assert "no detection files" in capsys.readouterr().err
    assert not (tmp_path / "history.sqlite").exists()


def test_command_rejects_a_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing.json"

    assert main([str(missing), "--history", str(tmp_path / "history.sqlite")]) == 1
    assert "does not exist" in capsys.readouterr().err


def test_command_rejects_an_invalid_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"format_version": 1}', encoding="utf-8")

    assert main([str(bad), "--history", str(tmp_path / "history.sqlite")]) == 1
    assert "is not a valid detection file" in capsys.readouterr().err
