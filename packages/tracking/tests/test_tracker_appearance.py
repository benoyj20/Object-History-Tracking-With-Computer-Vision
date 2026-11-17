"""Appearance-first tracking with hand-made embeddings (no model involved).

Embeddings are unit vectors in a few named directions, so their cosine similarities are known:
the same look gives 1.0, and `RED` versus `WHITE` gives 0.0.
"""

from datetime import UTC, datetime, timedelta
from itertools import count

import numpy as np
import pytest

from objhist_core import BoundingBox
from objhist_tracking.tracker import (
    APPEARANCE_GATE_DISTANCE,
    AppearanceSettings,
    Embedding,
    TrackerDetection,
    TrackerFrame,
    TrackerInputError,
    TrackerSettings,
    TrackEvent,
    track_session,
)

SIZE_PX = 1000
START = datetime(2026, 10, 18, 13, 30, tzinfo=UTC)
SETTINGS = TrackerSettings(
    min_confidence=0.25, appearance=AppearanceSettings(embedding_model="stub@test")
)


def look(*values: float) -> Embedding:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


WHITE = look(1, 0, 0)
RED = look(0, 1, 0)
WHITE_SEEN_AT_AN_ANGLE = look(0.8, 0.2, 0)

Spot = tuple[float, Embedding | None]
"""Centre x as a fraction of the photo (y is always 0.5) and the crop's embedding."""


def photos(*contents: list[Spot]) -> list[TrackerFrame]:
    ids = count(1)
    return [
        TrackerFrame(
            frame_id=f"f{index}",
            captured_at=START + timedelta(minutes=15 * index),
            image_width_px=SIZE_PX,
            image_height_px=SIZE_PX,
            detections=tuple(
                TrackerDetection(
                    detection_id=next(ids),
                    class_name="cup",
                    confidence=0.9,
                    box=BoundingBox(
                        x_min=x * SIZE_PX - 20, y_min=480, x_max=x * SIZE_PX + 20, y_max=520
                    ),
                    embedding=embedding,
                )
                for x, embedding in spots
            ),
        )
        for index, spots in enumerate(contents)
    ]


def test_camera_jump_beyond_the_position_limit_is_matched_by_appearance() -> None:
    # 0.3 of the photo is twice the position-only limit but inside the appearance gate.
    result = track_session(photos([(0.3, WHITE)], [(0.6, WHITE)]), SETTINGS)

    assert len(result.objects) == 1
    second = result.observations[1]
    assert second.appearance_similarity == pytest.approx(1.0)
    assert second.position_distance == pytest.approx(0.3)
    assert second.match_cost == pytest.approx(0.1 * 0.3)


def test_different_look_at_the_same_place_is_a_new_object() -> None:
    result = track_session(photos([(0.3, WHITE)], [(0.3, RED)], [(0.3, RED)]), SETTINGS)

    assert result.events == (
        TrackEvent(1, "added", None, "f0"),
        TrackEvent(2, "added", "f0", "f1"),
        TrackEvent(1, "removed", "f0", "f1"),
    )


def test_two_cups_that_trade_places_keep_their_identities() -> None:
    result = track_session(photos([(0.3, WHITE), (0.6, RED)], [(0.3, RED), (0.6, WHITE)]), SETTINGS)

    assert len(result.objects) == 2
    later = {o.object_key: o.detection_id for o in result.observations[2:]}
    assert later == {1: 4, 2: 3}


def test_position_decides_between_identical_looking_cups() -> None:
    result = track_session(
        photos([(0.3, WHITE), (0.6, WHITE)], [(0.32, WHITE), (0.62, WHITE)]), SETTINGS
    )

    later = {o.object_key: o.detection_id for o in result.observations[2:]}
    assert later == {1: 3, 2: 4}


def test_object_is_compared_with_its_latest_view() -> None:
    # The angled view (similarity 0.97 to white) becomes the reference; red stays far off.
    result = track_session(
        photos([(0.3, WHITE)], [(0.3, WHITE_SEEN_AT_AN_ANGLE)], [(0.3, WHITE_SEEN_AT_AN_ANGLE)]),
        SETTINGS,
    )

    assert len(result.objects) == 1
    assert result.observations[2].appearance_similarity == pytest.approx(1.0)


def test_a_jump_beyond_the_gate_is_a_new_object_even_if_it_looks_the_same() -> None:
    far = 0.1 + APPEARANCE_GATE_DISTANCE + 0.1
    result = track_session(photos([(0.1, WHITE)], [(far, WHITE)]), SETTINGS)

    assert len(result.objects) == 2


def test_one_missed_photo_is_still_not_a_removal() -> None:
    result = track_session(photos([(0.3, WHITE)], [], [(0.5, WHITE)]), SETTINGS)

    assert [e.event_type for e in result.events] == ["added"]


def test_settings_record_names_the_model_and_every_threshold() -> None:
    assert SETTINGS.tracker_version == "appearance-gated-v1"
    assert SETTINGS.as_record() == {
        "tracker_version": "appearance-gated-v1",
        "position_method": "normalized-box-centre",
        "removal_missing_frames": 2,
        "min_confidence": 0.25,
        "embedding_model": "stub@test",
        "min_appearance_similarity": 0.5,
        "max_gate_distance": 0.5,
        "position_weight": 0.1,
    }


def test_missing_embedding_is_refused() -> None:
    with pytest.raises(TrackerInputError, match="has no embedding"):
        track_session(photos([(0.3, WHITE), (0.6, None)]), SETTINGS)


def test_embeddings_of_different_lengths_are_refused() -> None:
    with pytest.raises(TrackerInputError, match="unlike the others"):
        track_session(photos([(0.3, WHITE), (0.6, look(1, 0))]), SETTINGS)


@pytest.mark.parametrize(
    ("min_similarity", "max_gate_distance", "position_weight"),
    [(1.5, 0.5, 0.1), (0.5, 0.0, 0.1), (0.5, 0.5, -0.1), (0.5, float("inf"), 0.1)],
)
def test_invalid_appearance_settings_are_refused(
    min_similarity: float, max_gate_distance: float, position_weight: float
) -> None:
    with pytest.raises(TrackerInputError):
        AppearanceSettings("stub@test", min_similarity, max_gate_distance, position_weight)
