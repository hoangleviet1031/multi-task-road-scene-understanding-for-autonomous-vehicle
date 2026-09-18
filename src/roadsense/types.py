"""Framework-neutral data structures shared across RoadSense modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
UInt8Array = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Detection:
    """One normalized BDD100K object annotation.

    Attributes:
        box: Pixel coordinates ``(x1, y1, x2, y2)`` as floats.
        label: Zero-based detection class ID.
        category: Human-readable BDD100K class name.
        attributes: Object metadata such as occlusion or truncation.
    """

    box: tuple[float, float, float, float]
    label: int
    category: str
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DetectionFrame:
    """Normalized annotations associated with one image.

    Attributes:
        name: Original image filename, usually including ``.jpg``.
        stem: Filename without extension and the repository-wide sample ID.
        sequence_id: Video/sequence grouping key used to prevent split leakage.
        attributes: Frame metadata such as weather, scene, and time of day.
        detections: Valid annotations belonging to the ten evaluation classes.
        segmentations: Legacy lane/drivable polygon annotations, when present.
    """

    name: str
    stem: str
    sequence_id: str
    attributes: dict[str, Any]
    detections: tuple[Detection, ...]
    segmentations: tuple[dict[str, Any], ...] = field(default_factory=tuple)


@dataclass(slots=True)
class RoadSceneSample:
    """Canonical Milestone 1 sample returned by the joint dataset.

    Attributes:
        name: Stable filename stem.
        image: RGB ``uint8`` array shaped ``[H, W, 3]``.
        boxes: ``float32`` boxes shaped ``[N, 4]`` in XYXY format.
        labels: Zero-based ``int64`` class IDs shaped ``[N]``.
        drivable_mask: Optional normalized ``uint8[H, W]`` mask.
        lane_mask: Optional binary ``uint8[H, W]`` mask.
        attributes: Frame-level metadata.
        sequence_id: Grouping key used when creating leakage-safe splits.
        task_available: Availability flags for detection, drivable, and lane.
    """

    name: str
    image: UInt8Array
    boxes: FloatArray
    labels: IntArray
    drivable_mask: UInt8Array | None
    lane_mask: UInt8Array | None
    attributes: dict[str, Any]
    sequence_id: str
    task_available: dict[str, bool]
