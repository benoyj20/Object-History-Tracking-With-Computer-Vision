"""Opening the SQLite object history and applying its schema migrations.

The history is one SQLite file, by default `$OBJHIST_DATA_DIR/histories/history.sqlite`
(ADR 0002). Its schema is built by the numbered SQL files in `migrations/`
(`0001_<name>.sql`, `0002_<name>.sql`, ...), applied in order, each in its own transaction,
and recorded with its SHA-256 in `schema_migrations`. An applied migration must never be
edited: opening a history whose recorded hash no longer matches the file fails, so schema
changes always go in a new migration.
"""

import hashlib
import os
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from pathlib import Path

MIGRATION_FILE_PATTERN = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
HISTORY_FILENAME = "history.sqlite"


class HistoryStoreError(RuntimeError):
    """The history file cannot be opened or brought up to date safely."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def default_history_path() -> Path:
    return Path(os.environ.get("OBJHIST_DATA_DIR", "data")) / "histories" / HISTORY_FILENAME


def load_migrations() -> tuple[Migration, ...]:
    """The packaged migrations in order. Versions must run 1, 2, 3, ... without gaps."""
    migrations: list[Migration] = []
    for entry in resources.files("objhist_tracking").joinpath("migrations").iterdir():
        if not entry.name.endswith(".sql"):
            continue
        match = MIGRATION_FILE_PATTERN.match(entry.name)
        if match is None:
            raise HistoryStoreError(
                f"migration file {entry.name} is not named like 0001_lowercase_name.sql"
            )
        # Normalize line endings so a checkout with CRLF endings hashes the same.
        sql = entry.read_text(encoding="utf-8").replace("\r\n", "\n")
        migrations.append(Migration(int(match.group(1)), match.group(2), sql))
    migrations.sort(key=lambda migration: migration.version)
    expected_versions = list(range(1, len(migrations) + 1))
    if [migration.version for migration in migrations] != expected_versions:
        found = [migration.version for migration in migrations]
        raise HistoryStoreError(f"migration versions must be 1..{len(migrations)}, found {found}")
    return tuple(migrations)


def open_history_store(path: Path) -> sqlite3.Connection:
    """Open the history at `path`, creating it if needed, and apply pending migrations.

    The connection has foreign keys enforced and `autocommit=False`; use `with connection:`
    around each unit of work so it commits on success and rolls back on error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, autocommit=True)
    try:
        # Foreign-key enforcement can only be switched on outside a transaction.
        connection.execute("PRAGMA foreign_keys = ON")
        connection.autocommit = False
        apply_migrations(connection, load_migrations())
    except sqlite3.DatabaseError as error:
        connection.close()
        raise HistoryStoreError(f"{path} is not a usable SQLite history: {error}") from error
    except Exception:
        connection.close()
        raise
    return connection


def apply_migrations(connection: sqlite3.Connection, migrations: Sequence[Migration]) -> None:
    """Check the migrations already applied, then apply the missing ones in order."""
    with connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version INTEGER PRIMARY KEY, name TEXT NOT NULL, "
            "sha256 TEXT NOT NULL, applied_at TEXT NOT NULL) STRICT"
        )
    applied = {
        version: (name, sha256)
        for version, name, sha256 in connection.execute(
            "SELECT version, name, sha256 FROM schema_migrations"
        )
    }
    known = {migration.version: migration for migration in migrations}
    for version, (name, sha256) in sorted(applied.items()):
        migration = known.get(version)
        if migration is None:
            raise HistoryStoreError(
                f"the history has migration {version} ({name}), which this code does not "
                "have; update the code before opening this history"
            )
        if migration.name != name or migration.sha256 != sha256:
            raise HistoryStoreError(
                f"migration {version} ({name}) was changed after it was applied to this "
                "history; restore the original file and put the change in a new migration"
            )
    for migration in migrations:
        if migration.version in applied:
            continue
        with connection:
            # With autocommit=False, executescript runs inside the open transaction, so a
            # failing migration leaves no partial schema behind.
            connection.executescript(migration.sql)
            connection.execute(
                "INSERT INTO schema_migrations (version, name, sha256, applied_at) "
                "VALUES (?, ?, ?, ?)",
                (migration.version, migration.name, migration.sha256, _now_utc()),
            )


def _now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")
