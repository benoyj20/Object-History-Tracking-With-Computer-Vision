"""Reading and writing appearance embeddings in the history's `detection_embeddings` table.

Embeddings are stored as little-endian float32 bytes with unit length, keyed by detection and
by the model revision that produced them (`<model_id>@<revision>`).
"""

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt

EMBEDDING_DTYPE = np.dtype("<f4")


class EmbeddingError(ValueError):
    """Embeddings cannot be computed or read as asked."""


def embedding_model_for(connection: sqlite3.Connection, model_id: str, session_id: str) -> str:
    """The one stored model revision (`<model_id>@<revision>`) of a session's embeddings."""
    prefix = f"{model_id}@"
    rows = connection.execute(
        "SELECT DISTINCT e.embedding_model FROM detection_embeddings e "
        "JOIN detections d USING (detection_id) JOIN frames f USING (frame_id) "
        "WHERE f.session_id = ? AND substr(e.embedding_model, 1, ?) = ? "
        "ORDER BY e.embedding_model",
        (session_id, len(prefix), prefix),
    ).fetchall()
    if not rows:
        raise EmbeddingError(
            f"session {session_id} has no {model_id} embeddings; compute them with "
            "python -m objhist_tracking.embed_detections"
        )
    if len(rows) > 1:
        found = ", ".join(row[0] for row in rows)
        raise EmbeddingError(
            f"session {session_id} has embeddings from several {model_id} revisions: {found}"
        )
    return str(rows[0][0])


def load_embeddings(
    connection: sqlite3.Connection, session_id: str, detector_version: str, embedding_model: str
) -> dict[int, npt.NDArray[np.float32]]:
    """Stored embeddings of the session's detections, by detection ID."""
    rows = connection.execute(
        "SELECT e.detection_id, e.dimensions, e.embedding FROM detection_embeddings e "
        "JOIN detections d USING (detection_id) JOIN frames f USING (frame_id) "
        "WHERE f.session_id = ? AND d.detector_version = ? AND e.embedding_model = ?",
        (session_id, detector_version, embedding_model),
    )
    embeddings: dict[int, npt.NDArray[np.float32]] = {}
    for detection_id, dimensions, blob in rows:
        vector = np.frombuffer(blob, dtype=EMBEDDING_DTYPE)
        if vector.shape != (dimensions,):
            raise EmbeddingError(f"embedding of detection {detection_id} is damaged")
        embeddings[detection_id] = vector.astype(np.float32)
    return embeddings


def store_embeddings(
    connection: sqlite3.Connection,
    detection_ids: Sequence[int],
    embeddings: npt.NDArray[np.float32],
    embedding_model: str,
) -> None:
    created_at = datetime.now(UTC).isoformat(timespec="microseconds")
    rows = []
    for detection_id, embedding in zip(detection_ids, embeddings, strict=True):
        if not np.all(np.isfinite(embedding)):
            raise EmbeddingError(f"embedding of detection {detection_id} is not finite")
        vector = embedding.astype(EMBEDDING_DTYPE)
        rows.append((detection_id, embedding_model, vector.shape[0], vector.tobytes(), created_at))
    with connection:
        connection.executemany(
            "INSERT INTO detection_embeddings (detection_id, embedding_model, dimensions, "
            "embedding, created_at) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
