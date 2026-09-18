"""NumPy object-detection metrics for protocol verification."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray


def _boxes(array: Any, field_name: str) -> NDArray[np.float64]:
    """Validate and normalize an ``N×4`` box array.

    Args:
        array: Array-like XYXY boxes.
        field_name: Name included in validation errors.

    Returns:
        ``float64[N,4]`` array.
    """

    result = np.asarray(array, dtype=np.float64)
    if result.size == 0:
        return np.empty((0, 4), dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"{field_name} must have shape [N,4], got {result.shape}.")
    if not bool(np.all(np.isfinite(result))):
        raise ValueError(f"{field_name} contains non-finite coordinates.")
    return result


def box_iou(boxes_a: Any, boxes_b: Any) -> NDArray[np.float64]:
    """Compute pairwise intersection over union for XYXY boxes.

    Args:
        boxes_a: Array-like boxes shaped ``[N,4]``.
        boxes_b: Array-like boxes shaped ``[M,4]``.

    Returns:
        ``float64[N,M]`` pairwise IoU matrix. Invalid negative-area boxes have
        zero area rather than producing a negative score.
    """

    first = _boxes(boxes_a, "boxes_a")
    second = _boxes(boxes_b, "boxes_b")
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float64)

    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_size = np.clip(bottom_right - top_left, 0.0, None)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    first_size = np.clip(first[:, 2:] - first[:, :2], 0.0, None)
    second_size = np.clip(second[:, 2:] - second[:, :2], 0.0, None)
    first_area = first_size[:, 0] * first_size[:, 1]
    second_area = second_size[:, 0] * second_size[:, 1]
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def _average_precision(recall: NDArray[np.float64], precision: NDArray[np.float64]) -> float:
    """Integrate an interpolated precision-recall curve.

    Args:
        recall: Monotonically non-decreasing recall samples.
        precision: Precision samples corresponding to ``recall``.

    Returns:
        Area under the interpolated precision-recall curve in ``[0,1]``.
    """

    extended_recall = np.concatenate(([0.0], recall, [1.0]))
    extended_precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(extended_precision) - 2, -1, -1):
        extended_precision[index] = max(
            extended_precision[index], extended_precision[index + 1]
        )
    change_points = np.where(extended_recall[1:] != extended_recall[:-1])[0]
    return float(
        np.sum(
            (extended_recall[change_points + 1] - extended_recall[change_points])
            * extended_precision[change_points + 1]
        )
    )


def _record_arrays(record: dict[str, Any], prediction: bool) -> tuple[NDArray, ...]:
    """Validate a single prediction or target record.

    Args:
        record: Mapping with boxes and labels, plus scores for predictions.
        prediction: Whether a scores field is required.

    Returns:
        Boxes, labels, and optionally scores as NumPy arrays.
    """

    boxes = _boxes(record.get("boxes", []), "boxes")
    labels = np.asarray(record.get("labels", []), dtype=np.int64).reshape(-1)
    if len(boxes) != len(labels):
        raise ValueError("Detection boxes and labels must have equal length.")
    if not prediction:
        return boxes, labels
    scores = np.asarray(record.get("scores", []), dtype=np.float64).reshape(-1)
    if len(scores) != len(boxes):
        raise ValueError("Prediction boxes and scores must have equal length.")
    if not bool(np.all(np.isfinite(scores))):
        raise ValueError("Prediction scores must be finite.")
    return boxes, labels, scores


def mean_average_precision(
    predictions: Sequence[dict[str, Any]],
    targets: Sequence[dict[str, Any]],
    class_ids: Iterable[int],
    iou_thresholds: Iterable[float] = (0.5,),
) -> dict[str, Any]:
    """Compute greedy per-class AP and mean AP for controlled experiments.

    This implementation is intentionally transparent and suitable for unit tests
    and internal validation. Final leaderboard reporting should also be checked
    with the official BDD100K evaluator.

    Args:
        predictions: Per-image dictionaries with ``boxes``, ``labels``, and
            ``scores`` arrays.
        targets: Per-image dictionaries with ``boxes`` and ``labels`` arrays.
        class_ids: Class IDs to evaluate.
        iou_thresholds: IoU match thresholds, for example ``0.50..0.95``.

    Returns:
        JSON-serializable dictionary containing AP per class and threshold,
        mAP per threshold, and overall mAP. Classes without ground truth receive
        ``None`` and are excluded from means.

    Raises:
        ValueError: If image counts, array lengths, or thresholds are invalid.
    """

    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must contain the same images.")
    classes = tuple(int(value) for value in class_ids)
    thresholds = tuple(float(value) for value in iou_thresholds)
    if not classes:
        raise ValueError("class_ids must not be empty.")
    if not thresholds or any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("IoU thresholds must be in the interval (0,1].")

    parsed_predictions = [_record_arrays(item, True) for item in predictions]
    parsed_targets = [_record_arrays(item, False) for item in targets]
    per_class: dict[str, dict[str, float | None]] = {}
    threshold_values: dict[float, list[float]] = {value: [] for value in thresholds}

    for class_id in classes:
        class_key = str(class_id)
        per_class[class_key] = {}
        ground_truth_by_image: dict[int, NDArray[np.float64]] = {}
        total_ground_truth = 0
        candidates: list[tuple[float, int, NDArray[np.float64]]] = []

        for image_id, (prediction_record, target_record) in enumerate(
            zip(parsed_predictions, parsed_targets)
        ):
            pred_boxes, pred_labels, pred_scores = prediction_record
            target_boxes, target_labels = target_record
            ground_truth = target_boxes[target_labels == class_id]
            ground_truth_by_image[image_id] = ground_truth
            total_ground_truth += len(ground_truth)
            mask = pred_labels == class_id
            for box, score in zip(pred_boxes[mask], pred_scores[mask]):
                candidates.append((float(score), image_id, box))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for threshold in thresholds:
            metric_key = f"ap@{threshold:.2f}"
            if total_ground_truth == 0:
                per_class[class_key][metric_key] = None
                continue

            matched = {
                image_id: np.zeros(len(boxes), dtype=bool)
                for image_id, boxes in ground_truth_by_image.items()
            }
            true_positive = np.zeros(len(candidates), dtype=np.float64)
            false_positive = np.zeros(len(candidates), dtype=np.float64)
            for index, (_, image_id, box) in enumerate(candidates):
                ground_truth = ground_truth_by_image[image_id]
                if len(ground_truth) == 0:
                    false_positive[index] = 1.0
                    continue
                overlaps = box_iou(box.reshape(1, 4), ground_truth)[0]
                best = int(np.argmax(overlaps))
                if overlaps[best] >= threshold and not matched[image_id][best]:
                    true_positive[index] = 1.0
                    matched[image_id][best] = True
                else:
                    false_positive[index] = 1.0

            accumulated_tp = np.cumsum(true_positive)
            accumulated_fp = np.cumsum(false_positive)
            recall = accumulated_tp / total_ground_truth
            precision = np.divide(
                accumulated_tp,
                accumulated_tp + accumulated_fp,
                out=np.zeros_like(accumulated_tp),
                where=(accumulated_tp + accumulated_fp) > 0,
            )
            ap = _average_precision(recall, precision)
            per_class[class_key][metric_key] = ap
            threshold_values[threshold].append(ap)

    map_by_threshold = {
        f"map@{threshold:.2f}": (
            float(np.mean(values)) if values else None
        )
        for threshold, values in threshold_values.items()
    }
    all_values = [value for values in threshold_values.values() for value in values]
    return {
        "per_class": per_class,
        "map_by_threshold": map_by_threshold,
        "map": float(np.mean(all_values)) if all_values else None,
    }
