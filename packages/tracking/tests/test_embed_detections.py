"""Embedding detections and tracking with them, using generated images and a stub embedder.

The stub turns each crop into its mean colour (as a unit vector), so crops of differently
coloured squares get different embeddings without any model.
"""

import sqlite3
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from objhist_core import BoundingBox, Detection, DetectorRun, FrameDetections, SessionDetections
from objhist_tracking.appearance import PADDING_RGB, square_crop
from objhist_tracking.embed_detections import embed_session, photo_path
from objhist_tracking.embedding_store import (
    EmbeddingError,
    embedding_model_for,
    load_embeddings,
)
from objhist_tracking.history_store import open_history_store
from objhist_tracking.import_detections import import_session
from objhist_tracking.track_session import run_tracker

SESSION_ID = "2026-10-18-0930"
DETECTOR = "test-detector-000000000000-single1280"
START = datetime(2026, 10, 18, 13, 30, tzinfo=UTC)
WIDTH_PX, HEIGHT_PX = 400, 200
WHITE, RED = (250, 250, 250), (220, 30, 30)


class MeanColourEmbedder:
    model_name = "stub/mean-colour@rev1"

    def __init__(self) -> None:
        self.crops_seen = 0

    def embed(self, crops: Sequence[Image.Image]) -> npt.NDArray[np.float32]:
        self.crops_seen += len(crops)
        rows = []
        for crop in crops:
            pixels = np.asarray(crop, dtype=np.float32).reshape(-1, 3)
            # Ignore the padding so only the object's colour counts.
            object_pixels = pixels[np.any(pixels != PADDING_RGB, axis=1)]
            mean = object_pixels.mean(axis=0)
            rows.append(mean / np.linalg.norm(mean))
        return np.array(rows, dtype=np.float32)


def box(x_px: float) -> BoundingBox:
    return BoundingBox(x_min=x_px - 20, y_min=80, x_max=x_px + 20, y_max=120)


def write_photo(path: Path, cups: list[tuple[float, tuple[int, int, int]]]) -> None:
    """A grey counter with coloured squares where the cups are."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (WIDTH_PX, HEIGHT_PX), (90, 90, 90))
    for x_px, colour in cups:
        b = box(x_px)
        image.paste(colour, (int(b.x_min), int(b.y_min), int(b.x_max), int(b.y_max)))
    image.save(path)


def make_session(
    sequences_dir: Path, photos: list[list[tuple[float, tuple[int, int, int]]]]
) -> SessionDetections:
    """Writes the photos and returns matching detections, one cup per square."""
    run = DetectorRun(
        version=DETECTOR,
        weights_sha256="0" * 64,
        class_names=("cup",),
        confidence_threshold=0.25,
        settings={},
        framework_version="test",
    )
    frames = []
    for index, cups in enumerate(photos):
        frame_id = f"{SESSION_ID}/f{index}.png"
        write_photo(sequences_dir / frame_id, cups)
        frames.append(
            FrameDetections(
                frame_id=frame_id,
                captured_at=START + timedelta(minutes=15 * index),
                image_width_px=WIDTH_PX,
                image_height_px=HEIGHT_PX,
                detector_version=DETECTOR,
                detections=tuple(
                    Detection(class_name="cup", confidence=0.9, box=box(x_px)) for x_px, _ in cups
                ),
            )
        )
    return SessionDetections(
        session_id=SESSION_ID,
        created_at=START,
        capture_time_source="exif",
        detector=run,
        frames=tuple(frames),
    )


@pytest.fixture
def history(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    connection = open_history_store(tmp_path / "history.sqlite")
    yield connection
    connection.close()


def test_square_crop_keeps_a_long_object_whole() -> None:
    photo = Image.new("RGB", (300, 100), (10, 20, 30))

    crop = square_crop(photo, BoundingBox(x_min=50, y_min=40, x_max=250, y_max=60))

    assert crop.size == (200, 200)
    assert crop.getpixel((100, 100)) == (10, 20, 30)
    assert crop.getpixel((100, 0)) == PADDING_RGB


def test_embeddings_are_stored_once_per_detection(
    history: sqlite3.Connection, tmp_path: Path
) -> None:
    sequences = tmp_path / "sequences"
    import_session(history, make_session(sequences, [[(100, WHITE)], [(100, WHITE), (300, RED)]]))
    embedder = MeanColourEmbedder()

    first = embed_session(history, SESSION_ID, embedder, sequences)
    second = embed_session(history, SESSION_ID, embedder, sequences)

    assert (first.computed, first.already_cached) == (3, 0)
    assert (second.computed, second.already_cached) == (0, 3)
    assert embedder.crops_seen == 3
    stored = load_embeddings(history, SESSION_ID, DETECTOR, embedder.model_name)
    assert len(stored) == 3
    assert all(np.linalg.norm(vector) == pytest.approx(1.0) for vector in stored.values())


def test_photo_changed_since_detection_is_refused(
    history: sqlite3.Connection, tmp_path: Path
) -> None:
    sequences = tmp_path / "sequences"
    import_session(history, make_session(sequences, [[(100, WHITE)]]))
    Image.new("RGB", (WIDTH_PX + 1, HEIGHT_PX)).save(sequences / SESSION_ID / "f0.png")

    with pytest.raises(EmbeddingError, match="has changed since detection"):
        embed_session(history, SESSION_ID, MeanColourEmbedder(), sequences)


def test_frame_id_cannot_point_outside_the_photo_folder(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError, match="does not name a photo"):
        photo_path(tmp_path / "sequences", "../../etc/passwd")


def test_tracking_needs_embeddings_from_the_named_model(history: sqlite3.Connection) -> None:
    with pytest.raises(EmbeddingError, match="no stub/mean-colour embeddings"):
        embedding_model_for(history, "stub/mean-colour", SESSION_ID)


def test_appearance_run_keeps_cups_apart_by_colour(
    history: sqlite3.Connection, tmp_path: Path
) -> None:
    # The two cups trade places between photos; only their colours tell them apart.
    sequences = tmp_path / "sequences"
    import_session(
        history,
        make_session(sequences, [[(100, WHITE), (300, RED)], [(100, RED), (300, WHITE)]]),
    )
    embed_session(history, SESSION_ID, MeanColourEmbedder(), sequences)

    summary = run_tracker(history, SESSION_ID, appearance_model_id="stub/mean-colour")

    assert (summary.object_count, summary.added_count, summary.removed_count) == (2, 2, 0)
    run = history.execute("SELECT tracker_version, appearance_model FROM tracker_runs").fetchone()
    assert run == ("appearance-gated-v1", "stub/mean-colour@rev1")
    similarities = history.execute(
        "SELECT appearance_similarity FROM observations WHERE appearance_similarity IS NOT NULL"
    ).fetchall()
    assert [value for (value,) in similarities] == [pytest.approx(1.0)] * 2
