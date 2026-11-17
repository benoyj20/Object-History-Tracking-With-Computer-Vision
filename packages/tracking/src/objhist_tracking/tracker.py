"""Object tracking over one session's photos, by position alone or by appearance first.

Each photo's detections are matched to the objects already being tracked, class by class.
Matching is one-to-one and minimizes the total cost (Hungarian algorithm), so two identical
cups are not swapped. Positions are box centres as fractions of the photo's width and height,
which keeps photos of different sizes comparable but does not undo camera movement.

Two matching modes:
- Position only (`position-centre-v1`, the evaluation baseline): the cost is the centre
  distance, and pairs further apart than `MAX_MATCH_DISTANCE` never match.
- Appearance first (`appearance-gated-v1`, with `AppearanceSettings`): the cost is the
  appearance distance (1 - cosine similarity of the crops' embeddings) plus a small
  position term that only breaks ties between look-alikes, such as two matching forks. A pair
  matches only if its similarity reaches `min_similarity` and its centres are within the
  loose `max_gate_distance`, wide enough to survive handheld camera jumps. An object is
  compared with the embedding of its latest sighting, the view most like the next photo.

Either mode can compensate for camera movement (`AlignmentSettings`): before matching a
photo, the shift, rotation, and zoom that best carry the tracked objects onto the new boxes
are estimated (`objhist_tracking.camera_alignment`) and applied to every tracked position,
including missing objects'. Positions are then measured in units of the photo's long side,
which rotation does not distort, and tight position limits work again. A photo where too few
objects line up is matched without compensation; `TrackingResult.alignments` records which.

From the matches the tracker derives the history:
- added: an unmatched detection starts a new object. Its placement interval runs from the
  previous photo to this one, or is open (no earlier photo) in the session's first photo.
- removed: an object missing from `removal_missing_frames` processed photos in a row. The
  interval runs from the last photo it was seen in to the first photo it was missing from.
  A shorter gap (a missed detection, a hand in the way) is not a removal, and an object
  missing only at the session's end gets no event.
- An object stays matchable while missing until its removal is confirmed; one that reappears
  after that is a new object.

Moved events are not produced: the "moved" threshold is still open in
docs/evaluation-protocol.md. Observations keep every position, so they can be derived later.

The tracker is a pure function of its input; reading from and writing to the history store,
and computing embeddings, are done elsewhere.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

from objhist_core import BoundingBox
from objhist_tracking.camera_alignment import (
    AlignmentSettings,
    SimilarityTransform,
    estimate_alignment,
)

TRACKER_VERSION = "position-centre-v1"
"""Position-only matching. Changes whenever its matching or event rules change."""

APPEARANCE_TRACKER_VERSION = "appearance-gated-v1"
"""Appearance-first matching. Changes whenever its matching or event rules change."""

POSITION_ALIGNED_TRACKER_VERSION = "position-aligned-v1"
"""Position-only matching with camera-movement compensation."""

APPEARANCE_ALIGNED_TRACKER_VERSION = "appearance-aligned-v2"
"""Appearance-first matching with camera-movement compensation. v2: objects outside the
position gate can still match by appearance, because they moved; v1, whose runs are in the
history, rejected them."""

POSITION_METHOD = "normalized-box-centre"
ALIGNED_POSITION_METHOD = "long-side-box-centre-aligned"

ALIGNED_MATCH_DISTANCE = 0.08
"""Position only, aligned: largest centre distance, in long-side units, after compensating for
camera movement. Provisional: about twice the alignment inlier distance."""

ALIGNED_GATE_DISTANCE = 0.2
"""Appearance first, aligned: in long-side units after compensation, the distance within which
an object counts as having stayed put and is matched first. Objects further away can still
match afterwards, by appearance alone, as moved objects."""

ALIGNED_POSITION_WEIGHT = 1.0
"""Appearance first, aligned: once positions are compensated, position is a real clue, so it
weighs as much as appearance distance: 0.1 long-side units cost as much as 0.1 similarity."""

MAX_MATCH_DISTANCE = 0.15
"""Position only: largest centre distance, in fractions of the photo (0 to about 1.41),
still counted as the same object. Provisional and not tuned. On the 19 provided sessions only
69% of detections had a same-class box this close in the previous photo, because the camera
moves; that is why the appearance mode exists."""

MIN_APPEARANCE_SIMILARITY = 0.5
"""Appearance first: lowest cosine similarity between embeddings still counted as the same
object. Provisional and not tuned; tune on validation scenarios, never on the test set."""

APPEARANCE_GATE_DISTANCE = 0.5
"""Appearance first: largest centre distance, in fractions of the photo, for a match. Loose on
purpose: about 90% of the provided sessions' photo-to-photo shifts are below 0.41."""

POSITION_TIE_BREAK_WEIGHT = 0.1
"""Appearance first: weight of the centre distance in the match cost. Small, so position only
decides between look-alikes whose appearance distances are nearly equal."""

REMOVAL_MISSING_FRAMES = 2
"""Processed photos in a row an object must be missing before it counts as removed; 2
tolerates one missed detection or a brief occlusion."""

TrackEventType = Literal["added", "removed"]
Embedding = npt.NDArray[np.float32]


class TrackerInputError(ValueError):
    """The photos or settings given to the tracker are inconsistent."""


@dataclass(frozen=True)
class AppearanceSettings:
    embedding_model: str
    """The model and revision that produced the embeddings, e.g. 'facebook/dinov2-small@...'."""
    min_similarity: float = MIN_APPEARANCE_SIMILARITY
    max_gate_distance: float = APPEARANCE_GATE_DISTANCE
    position_weight: float = POSITION_TIE_BREAK_WEIGHT

    def __post_init__(self) -> None:
        if not self.embedding_model:
            raise TrackerInputError("embedding_model must name the model that made the embeddings")
        if not -1 <= self.min_similarity <= 1:
            raise TrackerInputError(f"min_similarity {self.min_similarity} is not in -1..1")
        if not math.isfinite(self.max_gate_distance) or self.max_gate_distance <= 0:
            raise TrackerInputError(
                f"max_gate_distance {self.max_gate_distance} must be a positive number"
            )
        if not math.isfinite(self.position_weight) or self.position_weight < 0:
            raise TrackerInputError(f"position_weight {self.position_weight} must be 0 or more")


@dataclass(frozen=True)
class TrackerSettings:
    min_confidence: float
    """Detections below this are ignored; at least the detector's own threshold."""
    max_match_distance: float = MAX_MATCH_DISTANCE
    """Used only without `appearance`."""
    removal_missing_frames: int = REMOVAL_MISSING_FRAMES
    appearance: AppearanceSettings | None = None
    """Match by appearance first; None for the position-only baseline."""
    alignment: AlignmentSettings | None = None
    """Compensate for camera movement before matching; None to compare raw positions."""

    def __post_init__(self) -> None:
        if not 0 <= self.min_confidence <= 1:
            raise TrackerInputError(f"min_confidence {self.min_confidence} is not in 0..1")
        if not math.isfinite(self.max_match_distance) or self.max_match_distance <= 0:
            raise TrackerInputError(
                f"max_match_distance {self.max_match_distance} must be a positive number"
            )
        if self.removal_missing_frames < 1:
            raise TrackerInputError(
                f"removal_missing_frames {self.removal_missing_frames} must be at least 1"
            )

    @property
    def tracker_version(self) -> str:
        if self.alignment is None:
            return TRACKER_VERSION if self.appearance is None else APPEARANCE_TRACKER_VERSION
        if self.appearance is None:
            return POSITION_ALIGNED_TRACKER_VERSION
        return APPEARANCE_ALIGNED_TRACKER_VERSION

    def as_record(self) -> dict[str, str | float | int]:
        """Everything that decided the result, for `tracker_runs.settings_json`."""
        record: dict[str, str | float | int] = {
            "tracker_version": self.tracker_version,
            "position_method": (
                POSITION_METHOD if self.alignment is None else ALIGNED_POSITION_METHOD
            ),
            "removal_missing_frames": self.removal_missing_frames,
            "min_confidence": self.min_confidence,
        }
        if self.appearance is None:
            record["max_match_distance"] = self.max_match_distance
        else:
            record |= {
                "embedding_model": self.appearance.embedding_model,
                "min_appearance_similarity": self.appearance.min_similarity,
                "max_gate_distance": self.appearance.max_gate_distance,
                "position_weight": self.appearance.position_weight,
            }
        if self.alignment is not None:
            record |= {
                "alignment_inlier_distance": self.alignment.inlier_distance,
                "alignment_min_inliers": self.alignment.min_inliers,
                "alignment_max_zoom": self.alignment.max_zoom,
            }
        return record


def default_settings(
    min_confidence: float, embedding_model: str | None = None, align: bool = False
) -> TrackerSettings:
    """The named default thresholds for each of the four tracker modes."""
    appearance: AppearanceSettings | None = None
    if embedding_model is not None:
        appearance = (
            AppearanceSettings(
                embedding_model,
                max_gate_distance=ALIGNED_GATE_DISTANCE,
                position_weight=ALIGNED_POSITION_WEIGHT,
            )
            if align
            else AppearanceSettings(embedding_model)
        )
    return TrackerSettings(
        min_confidence=min_confidence,
        max_match_distance=ALIGNED_MATCH_DISTANCE if align else MAX_MATCH_DISTANCE,
        appearance=appearance,
        alignment=AlignmentSettings() if align else None,
    )


@dataclass(frozen=True)
class TrackerDetection:
    """A stored detection: `detection_id` is its row in the history's `detections` table."""

    detection_id: int
    class_name: str
    confidence: float
    box: BoundingBox
    embedding: Embedding | None = field(default=None, compare=False)
    """Unit-length appearance embedding of the crop; required in appearance mode."""


@dataclass(frozen=True)
class TrackerFrame:
    """One processed photo. An empty `detections` means the detector found nothing in it."""

    frame_id: str
    captured_at: datetime
    image_width_px: int
    image_height_px: int
    detections: tuple[TrackerDetection, ...] = ()


@dataclass(frozen=True)
class TrackedObject:
    object_key: int
    """Numbers objects within one result, in order of first sighting, from 1."""
    class_name: str


@dataclass(frozen=True)
class Observation:
    """One detection assigned to an object. The evidence is None on the sighting that created
    the object, and `appearance_similarity` is None in position-only mode."""

    object_key: int
    detection_id: int
    frame_id: str
    position_distance: float | None
    appearance_similarity: float | None
    match_cost: float | None


@dataclass(frozen=True)
class TrackEvent:
    object_key: int
    event_type: TrackEventType
    frame_before_id: str | None
    frame_after_id: str


@dataclass(frozen=True)
class FrameAlignment:
    """Camera-movement compensation applied before matching one photo."""

    frame_id: str
    transform: SimilarityTransform | None
    """None when too few objects lined up, so positions were compared uncompensated."""
    inlier_count: int


@dataclass(frozen=True)
class TrackingResult:
    objects: tuple[TrackedObject, ...]
    observations: tuple[Observation, ...]
    events: tuple[TrackEvent, ...]
    alignments: tuple[FrameAlignment, ...] = ()
    """One per photo after the first that had tracked objects, in alignment mode only."""


@dataclass
class _ObjectState:
    key: int
    class_name: str
    centre: tuple[float, float]
    embedding: Embedding | None
    last_seen_frame_id: str
    missing_frames: int = 0
    first_missing_frame_id: str | None = None


@dataclass
class _Tracks:
    active: list[_ObjectState] = field(default_factory=list)
    objects: list[TrackedObject] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    events: list[TrackEvent] = field(default_factory=list)
    alignments: list[FrameAlignment] = field(default_factory=list)

    def start_object(
        self,
        detection: TrackerDetection,
        centre: tuple[float, float],
        frame: TrackerFrame,
        previous_frame_id: str | None,
    ) -> None:
        key = len(self.objects) + 1
        self.objects.append(TrackedObject(key, detection.class_name))
        self.active.append(
            _ObjectState(key, detection.class_name, centre, detection.embedding, frame.frame_id)
        )
        self.observations.append(
            Observation(key, detection.detection_id, frame.frame_id, None, None, None)
        )
        self.events.append(TrackEvent(key, "added", previous_frame_id, frame.frame_id))


@dataclass(frozen=True)
class _PairCosts:
    """Per (object, detection) pair: the cost to minimize, whether the pair may match, and
    the evidence to record."""

    costs: npt.NDArray[np.float64]
    allowed: npt.NDArray[np.bool_]
    distances: npt.NDArray[np.float64]
    similarities: npt.NDArray[np.float64] | None


def track_session(frames: Sequence[TrackerFrame], settings: TrackerSettings) -> TrackingResult:
    """Track objects through one session's processed photos, given in capture order."""
    _check_frames(frames, settings)
    tracks = _Tracks()
    previous_frame_id: str | None = None
    for frame in frames:
        detections = [d for d in frame.detections if d.confidence >= settings.min_confidence]
        centres = {d.detection_id: detection_centre(d, frame, settings) for d in detections}
        if settings.alignment is not None and tracks.active:
            _compensate_camera_movement(tracks, detections, centres, frame, settings.alignment)
        matched_keys: set[int] = set()
        for class_name in sorted({d.class_name for d in detections}):
            class_detections = [d for d in detections if d.class_name == class_name]
            candidates = [o for o in tracks.active if o.class_name == class_name]
            unmatched = _match_class(
                candidates, class_detections, centres, frame, settings, tracks, matched_keys
            )
            for detection in unmatched:
                tracks.start_object(
                    detection, centres[detection.detection_id], frame, previous_frame_id
                )
                matched_keys.add(tracks.objects[-1].object_key)
        _update_missing(tracks, frame, matched_keys, settings)
        previous_frame_id = frame.frame_id
    return TrackingResult(
        tuple(tracks.objects),
        tuple(tracks.observations),
        tuple(tracks.events),
        tuple(tracks.alignments),
    )


def detection_centre(
    detection: TrackerDetection, frame: TrackerFrame, settings: TrackerSettings
) -> tuple[float, float]:
    """Box centre as fractions of the photo's width and height, or, with alignment, both in
    units of its long side so that rotations are not distorted."""
    box = detection.box
    x_px, y_px = (box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2
    if settings.alignment is None:
        return (x_px / frame.image_width_px, y_px / frame.image_height_px)
    long_side_px = max(frame.image_width_px, frame.image_height_px)
    return (x_px / long_side_px, y_px / long_side_px)


def _compensate_camera_movement(
    tracks: _Tracks,
    detections: list[TrackerDetection],
    centres: dict[int, tuple[float, float]],
    frame: TrackerFrame,
    alignment_settings: AlignmentSettings,
) -> None:
    """Carry every tracked position into this photo's coordinates, if the move is clear."""
    alignment = estimate_alignment(
        [(obj.class_name, obj.centre) for obj in tracks.active],
        [(d.class_name, centres[d.detection_id]) for d in detections],
        alignment_settings,
    )
    if alignment is None:
        tracks.alignments.append(FrameAlignment(frame.frame_id, None, 0))
        return
    for obj in tracks.active:
        obj.centre = alignment.transform.apply(obj.centre)
    tracks.alignments.append(
        FrameAlignment(frame.frame_id, alignment.transform, alignment.inlier_count)
    )


def _pair_costs(
    candidates: list[_ObjectState],
    detections: list[TrackerDetection],
    centres: list[tuple[float, float]],
    settings: TrackerSettings,
) -> _PairCosts:
    distances = np.array(
        [[math.dist(obj.centre, centre) for centre in centres] for obj in candidates]
    )
    appearance = settings.appearance
    if appearance is None:
        return _PairCosts(distances, distances <= settings.max_match_distance, distances, None)
    object_embeddings = np.stack([_embedding_of(obj.embedding, obj.key) for obj in candidates])
    detection_embeddings = np.stack(
        [_embedding_of(d.embedding, d.detection_id) for d in detections]
    )
    similarities = (object_embeddings @ detection_embeddings.T).astype(np.float64)
    costs = (1 - similarities) + appearance.position_weight * distances
    allowed = (similarities >= appearance.min_similarity) & (
        distances <= appearance.max_gate_distance
    )
    return _PairCosts(costs, allowed, distances, similarities)


def _embedding_of(embedding: Embedding | None, owner: int) -> Embedding:
    if embedding is None:
        # _check_frames requires embeddings in appearance mode; this guards direct callers.
        raise TrackerInputError(f"{owner} has no appearance embedding")
    return embedding


def _match_class(
    candidates: list[_ObjectState],
    detections: list[TrackerDetection],
    all_centres: dict[int, tuple[float, float]],
    frame: TrackerFrame,
    settings: TrackerSettings,
    tracks: _Tracks,
    matched_keys: set[int],
) -> list[TrackerDetection]:
    """Match one class's detections to its active objects; return the unmatched detections.

    With appearance and alignment, matching has two stages. First, objects that stayed put:
    pairs within the position gate whose appearance agrees. Then objects that moved: the
    leftovers, by appearance alone, since a moved object is legitimately far from its old
    place relative to the counter."""
    if not candidates:
        return detections
    centres = [all_centres[d.detection_id] for d in detections]
    pairs = _pair_costs(candidates, detections, centres, settings)
    assignment = _assign(pairs.costs, pairs.allowed)
    if (
        settings.alignment is not None
        and settings.appearance is not None
        and pairs.similarities is not None
    ):
        used_rows = {row for row, _ in assignment}
        used_columns = {column for _, column in assignment}
        moved_allowed = pairs.similarities >= settings.appearance.min_similarity
        moved_allowed[list(used_rows), :] = False
        moved_allowed[:, list(used_columns)] = False
        assignment += _assign(pairs.costs, moved_allowed)
    for row, column in assignment:
        obj, detection = candidates[row], detections[column]
        obj.centre = centres[column]
        obj.embedding = detection.embedding
        obj.last_seen_frame_id = frame.frame_id
        obj.missing_frames = 0
        obj.first_missing_frame_id = None
        similarity = None if pairs.similarities is None else float(pairs.similarities[row, column])
        tracks.observations.append(
            Observation(
                obj.key,
                detection.detection_id,
                frame.frame_id,
                position_distance=float(pairs.distances[row, column]),
                appearance_similarity=similarity,
                match_cost=float(pairs.costs[row, column]),
            )
        )
        matched_keys.add(obj.key)
    matched_columns = {column for _, column in assignment}
    return [d for index, d in enumerate(detections) if index not in matched_columns]


def _assign(
    costs: npt.NDArray[np.float64], allowed: npt.NDArray[np.bool_]
) -> list[tuple[int, int]]:
    """One-to-one (row, column) pairs among the allowed ones, with the most pairs and then
    the lowest total cost."""
    if not allowed.any():
        return []
    # Disallowed pairs get a cost above any total of allowed pairs, so the assignment first
    # maximizes the number of allowed matches and then minimizes their cost; any disallowed
    # pair it still has to use is dropped.
    max_allowed_cost = float(costs[allowed].max())
    infeasible_cost = (max_allowed_cost + 1) * (costs.shape[0] + costs.shape[1] + 1)
    rows, columns = linear_sum_assignment(np.where(allowed, costs, infeasible_cost))
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns, strict=True)
        if allowed[row, column]
    ]


def _update_missing(
    tracks: _Tracks, frame: TrackerFrame, matched_keys: set[int], settings: TrackerSettings
) -> None:
    still_active: list[_ObjectState] = []
    for obj in tracks.active:
        if obj.key not in matched_keys:
            obj.missing_frames += 1
            if obj.first_missing_frame_id is None:
                obj.first_missing_frame_id = frame.frame_id
            if obj.missing_frames >= settings.removal_missing_frames:
                tracks.events.append(
                    TrackEvent(
                        obj.key, "removed", obj.last_seen_frame_id, obj.first_missing_frame_id
                    )
                )
                continue
        still_active.append(obj)
    tracks.active = still_active


def _check_frames(frames: Sequence[TrackerFrame], settings: TrackerSettings) -> None:
    seen_frames: set[str] = set()
    seen_detections: set[int] = set()
    dimensions: int | None = None
    for index, frame in enumerate(frames):
        if frame.frame_id in seen_frames:
            raise TrackerInputError(f"frame {frame.frame_id} appears twice")
        seen_frames.add(frame.frame_id)
        if index > 0 and frame.captured_at <= frames[index - 1].captured_at:
            raise TrackerInputError(
                f"frame {frame.frame_id} is not captured after {frames[index - 1].frame_id}"
            )
        if frame.image_width_px <= 0 or frame.image_height_px <= 0:
            raise TrackerInputError(f"frame {frame.frame_id} has no image size")
        for detection in frame.detections:
            if detection.detection_id in seen_detections:
                raise TrackerInputError(f"detection {detection.detection_id} appears twice")
            seen_detections.add(detection.detection_id)
            if settings.appearance is None or detection.confidence < settings.min_confidence:
                continue
            if detection.embedding is None:
                raise TrackerInputError(
                    f"detection {detection.detection_id} in {frame.frame_id} has no embedding "
                    f"from {settings.appearance.embedding_model}; compute embeddings first"
                )
            if detection.embedding.ndim != 1 or (
                dimensions is not None and detection.embedding.shape[0] != dimensions
            ):
                raise TrackerInputError(
                    f"detection {detection.detection_id} has an embedding of shape "
                    f"{detection.embedding.shape}, unlike the others"
                )
            dimensions = detection.embedding.shape[0]
