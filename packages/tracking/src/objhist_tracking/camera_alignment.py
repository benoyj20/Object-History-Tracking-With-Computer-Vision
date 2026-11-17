"""Estimating how the handheld camera moved between photos, from the detected boxes alone.

The counter is roughly a plane, and most objects stay put between photos, so the same
transform carries most tracked objects' positions onto the new photo's boxes. This module
fits a similarity transform (shift, rotation, uniform zoom) to box centres:

1. Candidate correspondences are all pairs of a tracked object and a new box of the same class.
2. Every two candidates with distinct points define a transform. Each is scored by how many
   objects it carries to within `inlier_distance` of a distinct same-class box. Trying every
   hypothesis instead of a random sample (RANSAC) keeps results repeatable; with 5 to 8
   objects per photo there are at most a few thousand hypotheses.
3. The best hypothesis is refit by least squares (Umeyama) on its inliers.

Objects that were really added, removed, or moved do not line up and so do not bend the
estimate. A similarity transform does not model perspective (tilting the camera); that needs
four or more well-spread points per photo, which a few objects rarely give reliably.

Positions are in units of the photo's long side, so a rotation does not distort them.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt

ALIGNMENT_INLIER_DISTANCE = 0.04
"""How close, in long-side units, a carried-over object must land to a box to support a
transform. Provisional: below the usual spacing of neighbouring cutlery, above detector jitter."""

MIN_ALIGNMENT_INLIERS = 3
"""Objects that must line up before a transform is trusted. Two points always fit exactly,
so two are no evidence at all."""

MAX_ALIGNMENT_ZOOM = 2.0
"""Largest zoom in or out between consecutive photos accepted as camera movement."""

Point = tuple[float, float]
Points = npt.NDArray[np.float64]


@dataclass(frozen=True)
class AlignmentSettings:
    inlier_distance: float = ALIGNMENT_INLIER_DISTANCE
    min_inliers: int = MIN_ALIGNMENT_INLIERS
    max_zoom: float = MAX_ALIGNMENT_ZOOM

    def __post_init__(self) -> None:
        if not math.isfinite(self.inlier_distance) or self.inlier_distance <= 0:
            raise ValueError(f"inlier_distance {self.inlier_distance} must be a positive number")
        if self.min_inliers < 2:
            raise ValueError(f"min_inliers {self.min_inliers} must be at least 2")
        if not math.isfinite(self.max_zoom) or self.max_zoom < 1:
            raise ValueError(f"max_zoom {self.max_zoom} must be 1 or more")


@dataclass(frozen=True)
class SimilarityTransform:
    """Maps a point p to `scale * R(rotation) @ p + (shift_x, shift_y)`."""

    scale: float
    rotation_radians: float
    shift_x: float
    shift_y: float

    def apply(self, point: Point) -> Point:
        cos, sin = math.cos(self.rotation_radians), math.sin(self.rotation_radians)
        x, y = point
        return (
            self.scale * (cos * x - sin * y) + self.shift_x,
            self.scale * (sin * x + cos * y) + self.shift_y,
        )


@dataclass(frozen=True)
class Alignment:
    transform: SimilarityTransform
    inlier_count: int


def estimate_alignment(
    source: Sequence[tuple[str, Point]],
    target: Sequence[tuple[str, Point]],
    settings: AlignmentSettings,
) -> Alignment | None:
    """The camera movement carrying `source` (class, position) onto `target`, or None when
    too few objects line up or the zoom is implausible."""
    candidates = [
        (s, t)
        for s, (source_class, _) in enumerate(source)
        for t, (target_class, _) in enumerate(target)
        if source_class == target_class
    ]
    best: tuple[int, float, list[tuple[int, int]]] | None = None
    for (s1, t1), (s2, t2) in combinations(candidates, 2):
        if s1 == s2 or t1 == t2:
            continue
        transform = fit_similarity(
            np.array([source[s1][1], source[s2][1]]), np.array([target[t1][1], target[t2][1]])
        )
        if transform is None or not _plausible(transform, settings):
            continue
        inliers, residual = _inliers(transform, source, target, settings.inlier_distance)
        if best is None or (len(inliers), -residual) > (len(best[2]), -best[1]):
            best = (len(inliers), residual, inliers)
    if best is None or best[0] < settings.min_inliers:
        return None
    pairs = best[2]
    refined = fit_similarity(
        np.array([source[s][1] for s, _ in pairs]), np.array([target[t][1] for _, t in pairs])
    )
    if refined is None or not _plausible(refined, settings):
        return None
    inliers, _ = _inliers(refined, source, target, settings.inlier_distance)
    if len(inliers) < settings.min_inliers:
        return None
    return Alignment(refined, len(inliers))


def fit_similarity(source: Points, target: Points) -> SimilarityTransform | None:
    """Least-squares similarity transform (Umeyama, without reflection) from at least two
    corresponding points; None when the source points coincide."""
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centred, target_centred = source - source_mean, target - target_mean
    source_variance = float((source_centred**2).sum() / len(source))
    if source_variance < 1e-12:
        return None
    covariance = target_centred.T @ source_centred / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    reflection = np.diag([1.0, float(np.sign(np.linalg.det(u) * np.linalg.det(vt)) or 1.0)])
    rotation = u @ reflection @ vt
    scale = float(np.trace(np.diag(singular_values) @ reflection) / source_variance)
    shift = target_mean - scale * rotation @ source_mean
    return SimilarityTransform(
        scale=scale,
        rotation_radians=math.atan2(rotation[1, 0], rotation[0, 0]),
        shift_x=float(shift[0]),
        shift_y=float(shift[1]),
    )


def _plausible(transform: SimilarityTransform, settings: AlignmentSettings) -> bool:
    return 1 / settings.max_zoom <= transform.scale <= settings.max_zoom


def _inliers(
    transform: SimilarityTransform,
    source: Sequence[tuple[str, Point]],
    target: Sequence[tuple[str, Point]],
    inlier_distance: float,
) -> tuple[list[tuple[int, int]], float]:
    """Source-target pairs within the distance, one-to-one, closest first; and their total
    distance, which breaks ties between hypotheses with as many inliers."""
    moved = [transform.apply(point) for _, point in source]
    close = sorted(
        (math.dist(moved[s], target_point), s, t)
        for s, (source_class, _) in enumerate(source)
        for t, (target_class, target_point) in enumerate(target)
        if source_class == target_class and math.dist(moved[s], target_point) <= inlier_distance
    )
    used_source: set[int] = set()
    used_target: set[int] = set()
    pairs: list[tuple[int, int]] = []
    total = 0.0
    for distance, s, t in close:
        if s in used_source or t in used_target:
            continue
        used_source.add(s)
        used_target.add(t)
        pairs.append((s, t))
        total += distance
    return pairs, total
