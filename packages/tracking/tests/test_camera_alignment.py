"""Camera-movement estimation and aligned tracking on synthetic scenes.

A scene is a few objects at known positions; "moving the camera" applies a known shift,
rotation, and zoom to all of them, which the alignment should recover.
"""

import math
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import count

import numpy as np
import pytest

from objhist_core import BoundingBox
from objhist_tracking.camera_alignment import (
    AlignmentSettings,
    SimilarityTransform,
    estimate_alignment,
    fit_similarity,
)
from objhist_tracking.tracker import (
    TrackerDetection,
    TrackerFrame,
    TrackEvent,
    default_settings,
    track_session,
)

SETTINGS = AlignmentSettings()
COUNTER: list[tuple[str, tuple[float, float]]] = [
    ("plate", (0.30, 0.30)),
    ("cup", (0.55, 0.25)),
    ("cup", (0.62, 0.40)),
    ("fork", (0.40, 0.45)),
    ("knife", (0.45, 0.45)),
    ("spoon", (0.70, 0.30)),
]
CAMERA_MOVE = SimilarityTransform(
    scale=1.1, rotation_radians=math.radians(12), shift_x=0.08, shift_y=-0.05
)


def moved(scene: list[tuple[str, tuple[float, float]]], transform: SimilarityTransform):
    return [(name, transform.apply(point)) for name, point in scene]


def test_fit_recovers_a_known_transform_exactly() -> None:
    source = np.array([point for _, point in COUNTER])
    target = np.array([CAMERA_MOVE.apply(tuple(point)) for point in source])

    fitted = fit_similarity(source, target)

    assert fitted is not None
    assert fitted.scale == pytest.approx(1.1)
    assert fitted.rotation_radians == pytest.approx(math.radians(12))
    assert (fitted.shift_x, fitted.shift_y) == pytest.approx((0.08, -0.05))


def test_fit_refuses_coinciding_points() -> None:
    assert fit_similarity(np.array([[0.5, 0.5], [0.5, 0.5]]), np.array([[0, 0], [1, 1]])) is None


def test_camera_move_is_recovered_despite_a_removed_and_an_added_object() -> None:
    # The spoon is gone and a glass appeared; the other five still pin down the move.
    after = moved(COUNTER[:-1], CAMERA_MOVE) + [("glass", (0.1, 0.1))]

    alignment = estimate_alignment(COUNTER, after, SETTINGS)

    assert alignment is not None
    assert alignment.inlier_count == 5
    assert alignment.transform.rotation_radians == pytest.approx(math.radians(12), abs=1e-6)


def test_too_few_objects_lining_up_gives_no_alignment() -> None:
    two = COUNTER[:2]

    assert estimate_alignment(two, moved(two, CAMERA_MOVE), SETTINGS) is None


def test_implausible_zoom_is_refused() -> None:
    zoom = SimilarityTransform(scale=3.0, rotation_radians=0, shift_x=0, shift_y=0)

    assert estimate_alignment(COUNTER, moved(COUNTER, zoom), SETTINGS) is None


@pytest.mark.parametrize(
    ("inlier_distance", "min_inliers", "max_zoom"), [(0, 3, 2), (0.04, 1, 2), (0.04, 3, 0.5)]
)
def test_invalid_alignment_settings_are_refused(
    inlier_distance: float, min_inliers: int, max_zoom: float
) -> None:
    with pytest.raises(ValueError):
        AlignmentSettings(inlier_distance, min_inliers, max_zoom)


SIZE_PX = 1000
START = datetime(2026, 10, 18, 13, 30, tzinfo=UTC)


def photos(*scenes: list[tuple[str, tuple[float, float]]]) -> list[TrackerFrame]:
    """One square photo per scene; positions are long-side units, so pixels are x 1000."""
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
                    class_name=name,
                    confidence=0.9,
                    box=BoundingBox(
                        x_min=x * SIZE_PX - 10,
                        y_min=y * SIZE_PX - 10,
                        x_max=x * SIZE_PX + 10,
                        y_max=y * SIZE_PX + 10,
                    ),
                )
                for name, (x, y) in scene
            ),
        )
        for index, scene in enumerate(scenes)
    ]


def test_aligned_tracker_keeps_objects_through_a_camera_move() -> None:
    # Large enough that raw positions no longer match, as in about a third of real photos.
    big_move = SimilarityTransform(1.0, math.radians(5), 0.25, 0.0)
    frames = photos(COUNTER, moved(COUNTER, big_move))

    unaligned = track_session(frames, default_settings(0.25))
    aligned = track_session(frames, default_settings(0.25, align=True))

    assert len(unaligned.objects) > len(COUNTER)
    assert len(aligned.objects) == len(COUNTER)
    assert aligned.alignments[0].inlier_count == len(COUNTER)


def test_aligned_tracker_still_sees_a_real_removal() -> None:
    after = moved(COUNTER[:-1], CAMERA_MOVE)
    result = track_session(photos(COUNTER, after, after), default_settings(0.25, align=True))

    removals = [e for e in result.events if e.event_type == "removed"]
    assert removals == [TrackEvent(6, "removed", "f0", "f1")]


def test_missing_objects_move_with_the_camera_too() -> None:
    # The spoon is hidden in f1 while the camera moves, then reappears in f2 at its moved spot.
    second_move = SimilarityTransform(1.0, 0.0, 0.05, 0.0)
    f1 = moved(COUNTER[:-1], CAMERA_MOVE)
    f2 = moved(moved(COUNTER, CAMERA_MOVE), second_move)

    result = track_session(photos(COUNTER, f1, f2), default_settings(0.25, align=True))

    assert len(result.objects) == len(COUNTER)
    assert [e.event_type for e in result.events] == ["added"] * len(COUNTER)


def test_settings_record_names_the_alignment_rules() -> None:
    record = default_settings(0.25, "stub@test", align=True).as_record()

    assert record["tracker_version"] == "appearance-aligned-v2"
    assert record["position_method"] == "long-side-box-centre-aligned"
    assert record["alignment_min_inliers"] == 3
    assert record["max_gate_distance"] == 0.2


def with_looks(frames: list[TrackerFrame]) -> list[TrackerFrame]:
    """Gives every detection an embedding unique to its object (its index in COUNTER)."""
    looks = np.eye(len(COUNTER), dtype=np.float32)
    return [
        replace(
            frame,
            detections=tuple(
                replace(detection, embedding=looks[index])
                for index, detection in enumerate(frame.detections)
            ),
        )
        for frame in frames
    ]


# The camera moves and, in the same shot, the spoon is carried across the counter, about 0.66
# long-side units from where it was: far outside the 0.2 gate for objects that stayed put.
SPOON_MOVED = moved(COUNTER[:-1], CAMERA_MOVE) + [("spoon", CAMERA_MOVE.apply((0.3, 0.75)))]


def test_moved_object_is_kept_by_appearance_after_alignment() -> None:
    frames = with_looks(photos(COUNTER, SPOON_MOVED))

    result = track_session(frames, default_settings(0.25, "stub@test", align=True))

    assert len(result.objects) == len(COUNTER)
    assert all(e.frame_before_id is None for e in result.events)
    spoon = next(o for o in result.observations[len(COUNTER) :] if o.object_key == 6)
    assert spoon.position_distance is not None and spoon.position_distance > 0.5
    assert spoon.appearance_similarity == pytest.approx(1.0)


def test_moved_object_that_looks_different_is_not_forced_to_match() -> None:
    first, second = with_looks(photos(COUNTER, SPOON_MOVED))
    plate_look = first.detections[0].embedding
    unfamiliar_spoon = replace(second.detections[-1], embedding=plate_look)
    second = replace(second, detections=second.detections[:-1] + (unfamiliar_spoon,))

    result = track_session([first, second], default_settings(0.25, "stub@test", align=True))

    assert len(result.objects) == len(COUNTER) + 1
