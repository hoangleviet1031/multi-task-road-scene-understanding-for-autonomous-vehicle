"""Dataset integrity audit used before any model training."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from roadsense.data.bdd100k import BDD100KDataset
from roadsense.data.labels import DETECTION_CLASSES


def _sample_indices(length: int, limit: int | None) -> list[int]:
    """Choose deterministic indices spread over the complete dataset.

    Args:
        length: Dataset size.
        limit: Maximum samples, or ``None`` for all samples.

    Returns:
        Sorted unique zero-based indices.
    """

    if length <= 0:
        return []
    if limit is None or limit >= length:
        return list(range(length))
    if limit <= 0:
        raise ValueError("Audit limit must be positive or omitted.")
    return sorted(set(np.linspace(0, length - 1, num=limit, dtype=int).tolist()))


def audit_dataset(
    dataset: BDD100KDataset, limit: int | None = 500
) -> dict[str, Any]:
    """Inspect label alignment, shapes, boxes, classes, and scene metadata.

    Args:
        dataset: Initialized joint BDD100K dataset.
        limit: Maximum number of pixel samples inspected. Indices are distributed
            across the dataset instead of taking only the first records.

    Returns:
        JSON-serializable report containing inventory, distributions, errors,
        warnings, and an overall ``ok`` flag.
    """

    indices = _sample_indices(len(dataset), limit)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()
    metadata_counts: dict[str, Counter[str]] = {
        "weather": Counter(),
        "scene": Counter(),
        "timeofday": Counter(),
    }
    drivable_values: Counter[str] = Counter()
    lane_values: Counter[str] = Counter()
    invalid_boxes = 0
    out_of_bounds_boxes = 0
    empty_detection_samples = 0

    for index in indices:
        sample_name = dataset.names[index]
        try:
            sample = dataset[index]
        except Exception as exc:  # Audit must report bad samples rather than stop.
            errors.append(
                {
                    "sample": sample_name,
                    "kind": "load_error",
                    "message": str(exc),
                }
            )
            continue

        height, width = sample.image.shape[:2]
        if sample.image.ndim != 3 or sample.image.shape[2] != 3:
            errors.append(
                {
                    "sample": sample.name,
                    "kind": "image_shape",
                    "message": f"Expected RGB image, got {sample.image.shape}.",
                }
            )

        if sample.drivable_mask is not None:
            if sample.drivable_mask.shape != (height, width):
                errors.append(
                    {
                        "sample": sample.name,
                        "kind": "drivable_shape",
                        "message": (
                            f"Image {(height, width)} versus mask "
                            f"{sample.drivable_mask.shape}."
                        ),
                    }
                )
            for value, count in zip(
                *np.unique(sample.drivable_mask, return_counts=True)
            ):
                drivable_values[str(int(value))] += int(count)

        if sample.lane_mask is not None:
            if sample.lane_mask.shape != (height, width):
                errors.append(
                    {
                        "sample": sample.name,
                        "kind": "lane_shape",
                        "message": (
                            f"Image {(height, width)} versus mask {sample.lane_mask.shape}."
                        ),
                    }
                )
            for value, count in zip(*np.unique(sample.lane_mask, return_counts=True)):
                lane_values[str(int(value))] += int(count)

        if len(sample.boxes) == 0:
            empty_detection_samples += 1
        for box, label in zip(sample.boxes, sample.labels):
            x1, y1, x2, y2 = (float(value) for value in box)
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
                warnings.append(
                    {
                        "sample": sample.name,
                        "kind": "invalid_box",
                        "message": f"Non-positive box: {[x1, y1, x2, y2]}",
                    }
                )
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                out_of_bounds_boxes += 1
                warnings.append(
                    {
                        "sample": sample.name,
                        "kind": "out_of_bounds_box",
                        "message": (
                            f"Box {[x1, y1, x2, y2]} outside image "
                            f"{width}×{height}."
                        ),
                    }
                )
            label_id = int(label)
            class_name = (
                DETECTION_CLASSES[label_id]
                if 0 <= label_id < len(DETECTION_CLASSES)
                else f"unknown:{label_id}"
            )
            class_counts[class_name] += 1

        for field in metadata_counts:
            metadata_counts[field][str(sample.attributes.get(field, "undefined"))] += 1

    error_examples = errors[:100]
    warning_examples = warnings[:100]
    return {
        "schema_version": 1,
        "split": dataset.split,
        "required_tasks": list(dataset.required_tasks),
        "inventory": dataset.inventory.to_dict(),
        "inspected_samples": len(indices),
        "loadable_samples": len(indices) - sum(
            item["kind"] == "load_error" for item in errors
        ),
        "detection_class_counts": dict(sorted(class_counts.items())),
        "metadata_counts": {
            field: dict(sorted(counts.items()))
            for field, counts in metadata_counts.items()
        },
        "drivable_pixel_values": dict(sorted(drivable_values.items())),
        "lane_pixel_values": dict(sorted(lane_values.items())),
        "empty_detection_samples": empty_detection_samples,
        "invalid_boxes": invalid_boxes,
        "out_of_bounds_boxes": out_of_bounds_boxes,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "error_examples": error_examples,
        "warning_examples": warning_examples,
        "ok": len(errors) == 0,
    }
