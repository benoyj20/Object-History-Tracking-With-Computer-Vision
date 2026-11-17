"""Compute DINOv2 embeddings for the detections in the history and cache them there.

    uv run python -m objhist_tracking.embed_detections 2026-10-18-0930 [more sessions] \\
        [--detector <detector version>] [--device cpu|mps|cuda] [--history PATH]

For each session, every detection from the detector that has no embedding from the loaded
model revision yet is cropped from its photo in `$OBJHIST_DATA_DIR/raw/sequences/<frame_id>`
and embedded; existing embeddings are kept, so rerunning only fills gaps. Each photo's
embeddings are stored in one transaction. A photo whose upright size differs from the size
recorded at detection time stops the run, because its boxes would point at the wrong pixels.
Photos stay on this machine.
"""

import argparse
import os
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from objhist_core import BoundingBox
from objhist_tracking.appearance import (
    DINOV2_MODEL_ID,
    AppearanceError,
    Dinov2Embedder,
    Embedder,
    load_upright_photo,
    square_crop,
)
from objhist_tracking.embedding_store import EmbeddingError, store_embeddings
from objhist_tracking.history_store import (
    HistoryStoreError,
    default_history_path,
    open_history_store,
)
from objhist_tracking.track_session import TrackSessionError, resolve_detector_version


@dataclass(frozen=True)
class EmbeddingSummary:
    session_id: str
    computed: int
    already_cached: int


@dataclass(frozen=True)
class _PendingDetection:
    detection_id: int
    box: BoundingBox


def default_sequences_dir() -> Path:
    return Path(os.environ.get("OBJHIST_DATA_DIR", "data")) / "raw" / "sequences"


def photo_path(sequences_dir: Path, frame_id: str) -> Path:
    """The photo for a frame ID (`<session>/<file>`), refusing paths outside the folder."""
    root = sequences_dir.resolve()
    path = (root / frame_id).resolve()
    if not path.is_relative_to(root) or path == root:
        raise EmbeddingError(f"frame ID {frame_id!r} does not name a photo in {sequences_dir}")
    return path


def embed_session(
    connection: sqlite3.Connection,
    session_id: str,
    embedder: Embedder,
    sequences_dir: Path,
    detector_version: str | None = None,
) -> EmbeddingSummary:
    """Embed every detection of the session that has no embedding from this model yet."""
    detector_version = resolve_detector_version(connection, session_id, detector_version)
    total = connection.execute(
        "SELECT count(*) FROM detections d JOIN frames f USING (frame_id) "
        "WHERE f.session_id = ? AND d.detector_version = ?",
        (session_id, detector_version),
    ).fetchone()[0]
    pending = _pending_by_frame(connection, session_id, detector_version, embedder.model_name)
    computed = 0
    for (frame_id, width_px, height_px), detections in pending.items():
        photo = load_upright_photo(photo_path(sequences_dir, frame_id))
        if (photo.width, photo.height) != (width_px, height_px):
            raise EmbeddingError(
                f"{frame_id} is {photo.width}x{photo.height} px upright, but its detections "
                f"were made on {width_px}x{height_px} px; the photo has changed since detection"
            )
        crops = [square_crop(photo, detection.box) for detection in detections]
        embeddings = embedder.embed(crops)
        if embeddings.shape[0] != len(crops) or embeddings.ndim != 2:
            raise AppearanceError(
                f"{embedder.model_name} returned {embeddings.shape} for {len(crops)} crops"
            )
        store_embeddings(
            connection,
            [d.detection_id for d in detections],
            embeddings,
            embedder.model_name,
        )
        computed += len(detections)
    return EmbeddingSummary(session_id, computed, total - computed)


def _pending_by_frame(
    connection: sqlite3.Connection, session_id: str, detector_version: str, embedding_model: str
) -> dict[tuple[str, int, int], list[_PendingDetection]]:
    rows = connection.execute(
        "SELECT f.frame_id, f.image_width_px, f.image_height_px, d.detection_id, "
        "d.x_min_px, d.y_min_px, d.x_max_px, d.y_max_px "
        "FROM detections d JOIN frames f USING (frame_id) "
        "WHERE f.session_id = ? AND d.detector_version = ? AND NOT EXISTS ("
        "  SELECT 1 FROM detection_embeddings e "
        "  WHERE e.detection_id = d.detection_id AND e.embedding_model = ?) "
        "ORDER BY f.captured_at_utc, d.detection_index",
        (session_id, detector_version, embedding_model),
    )
    pending: dict[tuple[str, int, int], list[_PendingDetection]] = {}
    for frame_id, width_px, height_px, detection_id, x_min, y_min, x_max, y_max in rows:
        box = BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max)
        pending.setdefault((frame_id, width_px, height_px), []).append(
            _PendingDetection(detection_id, box)
        )
    return pending


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="+", help="session IDs to embed")
    parser.add_argument(
        "--detector", help="detector version (default: the session's only imported one)"
    )
    parser.add_argument("--model", default=DINOV2_MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--device", default="cpu", help="PyTorch device: cpu, mps, or cuda")
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        help="photo folder (default: $OBJHIST_DATA_DIR/raw/sequences)",
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
    sequences_dir: Path = args.sequences_dir or default_sequences_dir()
    if not history_path.is_file():
        print(f"error: no history at {history_path}; import detections first", file=sys.stderr)
        return 1
    try:
        connection = open_history_store(history_path)
    except (HistoryStoreError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    try:
        embedder = Dinov2Embedder(args.model, device=args.device)
        print(f"model: {embedder.model_name} on {args.device}")
        for session_id in args.sessions:
            summary = embed_session(connection, session_id, embedder, sequences_dir, args.detector)
            print(
                f"{summary.session_id}: {summary.computed} embedded, "
                f"{summary.already_cached} already cached"
            )
    except (
        AppearanceError,
        EmbeddingError,
        TrackSessionError,
        OSError,
        sqlite3.Error,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        print("embeddings stored before the error are kept", file=sys.stderr)
        return 1
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
