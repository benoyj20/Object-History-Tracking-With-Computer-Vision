"""Synthetic photo sequences through the position-only tracker.

Positions are box centres as fractions of a 1000 x 1000 px photo, so `at(0.3, 0.5)` is a box
centred 30% across and halfway down.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

from objhist_core import BoundingBox
from objhist_tracking.tracker import (
    MAX_MATCH_DISTANCE,
    TrackerDetection,
    TrackerFrame,
    TrackerInputError,
    TrackerSettings,
    TrackEvent,
    TrackingResult,
    track_session,
)

SETTINGS = TrackerSettings(min_confidence=0.25)
SIZE_PX = 1000
START = datetime(2026, 10, 18, 13, 30, tzinfo=UTC)
BOX_HALF_PX = 20

Spot = tuple[str, float, float, float]
"""A detection to place: class name, centre x and y as fractions, confidence."""


def at(x: float, y: float = 0.5, class_name: str = "cup", confidence: float = 0.9) -> Spot:
    return (class_name, x, y, confidence)


def photos(*contents: list[Spot]) -> list[TrackerFrame]:
    """One photo per argument, 15 minutes apart, named f0, f1, ...; detection IDs are unique."""
    ids = count(1)
    frames = []
    for index, detections in enumerate(contents):
        frames.append(
            TrackerFrame(
                frame_id=f"f{index}",
                captured_at=START + timedelta(minutes=15 * index),
                image_width_px=SIZE_PX,
                image_height_px=SIZE_PX,
                detections=tuple(
                    TrackerDetection(
                        detection_id=next(ids),
                        class_name=class_name,
                        confidence=confidence,
                        box=BoundingBox(
                            x_min=x * SIZE_PX - BOX_HALF_PX,
                            y_min=y * SIZE_PX - BOX_HALF_PX,
                            x_max=x * SIZE_PX + BOX_HALF_PX,
                            y_max=y * SIZE_PX + BOX_HALF_PX,
                        ),
                    )
                    for class_name, x, y, confidence in detections
                ),
            )
        )
    return frames


def frames_seen(result: TrackingResult, object_key: int) -> list[str]:
    return [o.frame_id for o in result.observations if o.object_key == object_key]


def test_object_in_every_photo_is_one_object_present_from_the_start() -> None:
    result = track_session(photos([at(0.3)], [at(0.31)], [at(0.3)]), SETTINGS)

    assert [o.class_name for o in result.objects] == ["cup"]
    assert frames_seen(result, 1) == ["f0", "f1", "f2"]
    assert result.events == (TrackEvent(1, "added", None, "f0"),)


def test_new_object_is_placed_between_the_previous_photo_and_this_one() -> None:
    result = track_session(photos([], [], [at(0.3)]), SETTINGS)

    assert result.events == (TrackEvent(1, "added", "f1", "f2"),)


def test_first_sighting_has_no_match_evidence_and_later_ones_do() -> None:
    result = track_session(photos([at(0.3)], [at(0.34)]), SETTINGS)

    first, second = result.observations
    assert first.position_distance is None
    assert second.position_distance == pytest.approx(0.04)


def test_one_missed_photo_is_not_a_removal() -> None:
    result = track_session(photos([at(0.3)], [], [at(0.3)]), SETTINGS)

    assert len(result.objects) == 1
    assert frames_seen(result, 1) == ["f0", "f2"]
    assert [e.event_type for e in result.events] == ["added"]


def test_missing_twice_in_a_row_is_removed_between_last_seen_and_first_missing() -> None:
    result = track_session(photos([at(0.3)], [at(0.3)], [], [], []), SETTINGS)

    assert result.events == (
        TrackEvent(1, "added", None, "f0"),
        TrackEvent(1, "removed", "f1", "f2"),
    )


def test_missing_only_at_the_end_of_the_session_gets_no_event() -> None:
    result = track_session(photos([at(0.3)], [at(0.3)], []), SETTINGS)

    assert [e.event_type for e in result.events] == ["added"]


def test_object_back_after_a_confirmed_removal_is_a_new_object() -> None:
    result = track_session(photos([at(0.3)], [], [], [at(0.3)]), SETTINGS)

    assert [
        (e.object_key, e.event_type, e.frame_before_id, e.frame_after_id) for e in result.events
    ] == [
        (1, "added", None, "f0"),
        (1, "removed", "f0", "f1"),
        (2, "added", "f2", "f3"),
    ]


def test_two_identical_cups_stay_two_objects() -> None:
    result = track_session(photos([at(0.3), at(0.6)], [at(0.3), at(0.6)]), SETTINGS)

    assert len(result.objects) == 2
    assert frames_seen(result, 1) == ["f0", "f1"]
    assert frames_seen(result, 2) == ["f0", "f1"]


def test_identical_cups_shifted_by_the_camera_are_not_swapped() -> None:
    # Both cups shift right by 0.06. Matching the closest pair first would pair the right
    # cup with the left cup's new position and leave the other pair too far apart to match.
    result = track_session(photos([at(0.30), at(0.40)], [at(0.36), at(0.46)]), SETTINGS)

    assert len(result.objects) == 2
    by_object = {o.object_key: o.position_distance for o in result.observations[2:]}
    assert by_object == {1: pytest.approx(0.06), 2: pytest.approx(0.06)}


def test_different_classes_are_never_matched() -> None:
    result = track_session(photos([at(0.3)], [at(0.3, class_name="fork")]), SETTINGS)

    assert [o.class_name for o in result.objects] == ["cup", "fork"]
    assert (2, "added", "f0", "f1") in [
        (e.object_key, e.event_type, e.frame_before_id, e.frame_after_id) for e in result.events
    ]


def test_a_jump_beyond_the_match_distance_is_a_different_object() -> None:
    far = 0.3 + MAX_MATCH_DISTANCE + 0.05
    result = track_session(photos([at(0.3)], [at(far)]), SETTINGS)

    assert len(result.objects) == 2


def test_empty_photo_in_the_middle_is_a_gap_not_a_removal() -> None:
    result = track_session(photos([at(0.3), at(0.7)], [], [at(0.3), at(0.7)]), SETTINGS)

    assert len(result.objects) == 2
    assert all(e.event_type == "added" for e in result.events)


def test_empty_session_has_no_objects() -> None:
    result = track_session(photos([], []), SETTINGS)

    assert (result.objects, result.observations, result.events) == ((), (), ())


def test_detections_below_the_confidence_threshold_are_ignored() -> None:
    settings = TrackerSettings(min_confidence=0.5)
    result = track_session(photos([at(0.3, confidence=0.4)]), settings)

    assert result.objects == ()


def test_settings_record_names_every_rule() -> None:
    assert SETTINGS.as_record() == {
        "tracker_version": "position-centre-v1",
        "position_method": "normalized-box-centre",
        "max_match_distance": MAX_MATCH_DISTANCE,
        "removal_missing_frames": 2,
        "min_confidence": 0.25,
    }


def test_photos_out_of_order_are_refused() -> None:
    first, second = photos([], [])

    with pytest.raises(TrackerInputError, match="not captured after"):
        track_session([second, first], SETTINGS)


def test_repeated_detection_ids_are_refused() -> None:
    (frame,) = photos([at(0.3)])

    with pytest.raises(TrackerInputError, match="appears twice"):
        track_session(
            [frame, replace(frame, frame_id="f1", captured_at=START + timedelta(1))], SETTINGS
        )


@pytest.mark.parametrize(
    ("min_confidence", "max_match_distance", "removal_missing_frames"),
    [(1.5, MAX_MATCH_DISTANCE, 2), (0.25, 0.0, 2), (0.25, float("nan"), 2), (0.25, 0.1, 0)],
)
def test_invalid_settings_are_refused(
    min_confidence: float, max_match_distance: float, removal_missing_frames: int
) -> None:
    with pytest.raises(TrackerInputError):
        TrackerSettings(min_confidence, max_match_distance, removal_missing_frames)
