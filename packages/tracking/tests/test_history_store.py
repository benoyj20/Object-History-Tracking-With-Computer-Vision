import sqlite3
from pathlib import Path

import pytest

from objhist_tracking.history_store import (
    HistoryStoreError,
    Migration,
    apply_migrations,
    load_migrations,
    open_history_store,
)


def table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    return {name for (name,) in rows}


def test_packaged_migrations_are_numbered_from_one() -> None:
    migrations = load_migrations()
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))
    assert migrations[0].name == "detector_output"


def test_new_history_gets_the_detector_output_tables(tmp_path: Path) -> None:
    connection = open_history_store(tmp_path / "histories" / "history.sqlite")
    with connection:
        assert {
            "schema_migrations",
            "detector_runs",
            "sessions",
            "frames",
            "session_imports",
            "processed_frames",
            "detections",
            "tracker_runs",
            "objects",
            "observations",
            "events",
            "detection_embeddings",
        } <= table_names(connection)
        recorded = connection.execute("SELECT version, name FROM schema_migrations").fetchall()
    connection.close()
    assert recorded == [(m.version, m.name) for m in load_migrations()]


def test_reopening_applies_nothing_twice(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite"
    open_history_store(path).close()
    connection = open_history_store(path)
    with connection:
        count = connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0]
    connection.close()
    assert count == len(load_migrations())


def test_foreign_keys_are_enforced(tmp_path: Path) -> None:
    connection = open_history_store(tmp_path / "history.sqlite")
    with pytest.raises(sqlite3.IntegrityError), connection:
        connection.execute(
            "INSERT INTO frames VALUES ('missing-session/a.jpg', 'missing-session', "
            "'2026-10-18T13:30:00.000000+00:00', 0, 10, 10)"
        )
    connection.close()


def test_edited_applied_migration_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite"
    connection = open_history_store(path)
    with connection:
        connection.execute("UPDATE schema_migrations SET sha256 = 'edited' WHERE version = 1")
    connection.close()
    with pytest.raises(HistoryStoreError, match="changed after it was applied"):
        open_history_store(path)


def test_history_from_newer_code_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite"
    connection = open_history_store(path)
    with connection:
        connection.execute(
            "INSERT INTO schema_migrations VALUES (999, 'future', 'x', '2026-11-17T00:00:00Z')"
        )
    connection.close()
    with pytest.raises(HistoryStoreError, match="does not have"):
        open_history_store(path)


def test_failing_migration_leaves_no_partial_schema(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "history.sqlite", autocommit=False)
    broken = Migration(1, "broken", "CREATE TABLE half_done (id INTEGER);\nNOT VALID SQL;")
    with pytest.raises(sqlite3.OperationalError):
        apply_migrations(connection, [broken])
    assert "half_done" not in table_names(connection)
    assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 0
    connection.close()


def test_file_that_is_not_sqlite_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite"
    path.write_text("not a database, just text that is long enough to have a header", "utf-8")
    with pytest.raises(HistoryStoreError, match="not a usable SQLite history"):
        open_history_store(path)
