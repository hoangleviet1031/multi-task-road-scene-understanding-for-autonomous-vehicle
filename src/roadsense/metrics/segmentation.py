"""NumPy metrics for dense road-scene predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def confusion_matrix(
    prediction: NDArray[np.generic],
    target: NDArray[np.generic],
    num_classes: int,
    ignore_index: int | None = None,
) -> NDArray[np.int64]:
    """Build a semantic-segmentation confusion matrix.

    Rows represent ground-truth classes and columns represent predictions.

    Args:
        prediction: Integer prediction mask of any shape.
        target: Equal-shaped integer ground-truth mask.
        num_classes: Number of classes numbered from zero.
        ignore_index: Optional ground-truth value excluded from evaluation.

    Returns:
        ``int64[num_classes, num_classes]`` confusion matrix.

    Raises:
        ValueError: If shapes differ, class count is invalid, or evaluated pixels
            contain class IDs outside ``[0, num_classes)``.
    """

    predicted = np.asarray(prediction)
    expected = np.asarray(target)
    if predicted.shape != expected.shape:
        raise ValueError(
            f"Prediction shape {predicted.shape} differs from target {expected.shape}."
        )
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    predicted = predicted.reshape(-1).astype(np.int64, copy=False)
    expected = expected.reshape(-1).astype(np.int64, copy=False)
    valid = np.ones(expected.shape, dtype=bool)
    if ignore_index is not None:
        valid &= expected != int(ignore_index)
    predicted = predicted[valid]
    expected = expected[valid]

    if expected.size:
        invalid_target = (expected < 0) | (expected >= num_classes)
        invalid_prediction = (predicted < 0) | (predicted >= num_classes)
        if bool(np.any(invalid_target)):
            raise ValueError(
                f"Target contains invalid class IDs: {np.unique(expected[invalid_target])}"
            )
        if bool(np.any(invalid_prediction)):
            raise ValueError(
                "Prediction contains invalid class IDs: "
                f"{np.unique(predicted[invalid_prediction])}"
            )

    encoded = expected * num_classes + predicted
    counts = np.bincount(encoded, minlength=num_classes**2)
    return counts.reshape(num_classes, num_classes).astype(np.int64, copy=False)


def segmentation_metrics(
    prediction: NDArray[np.generic],
    target: NDArray[np.generic],
    num_classes: int,
    ignore_index: int | None = None,
) -> dict[str, Any]:
    """Compute IoU, Dice, and accuracy for semantic segmentation.

    Classes absent from both prediction and target receive ``None`` and are
    excluded from macro means. This prevents a non-existent class from inflating
    the score.

    Args:
        prediction: Integer prediction mask.
        target: Equal-shaped integer ground-truth mask.
        num_classes: Number of class IDs.
        ignore_index: Optional target value excluded from evaluation.

    Returns:
        JSON-serializable dictionary with confusion matrix, per-class values,
        mean IoU, mean Dice, and pixel accuracy.
    """

    matrix = confusion_matrix(prediction, target, num_classes, ignore_index)
    true_positive = np.diag(matrix).astype(np.float64)
    ground_truth = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    union = ground_truth + predicted - true_positive
    dice_denominator = ground_truth + predicted

    iou_values: list[float | None] = []
    dice_values: list[float | None] = []
    for class_id in range(num_classes):
        iou_values.append(
            None
            if union[class_id] == 0
            else float(true_positive[class_id] / union[class_id])
        )
        dice_values.append(
            None
            if dice_denominator[class_id] == 0
            else float(2.0 * true_positive[class_id] / dice_denominator[class_id])
        )

    valid_iou = [value for value in iou_values if value is not None]
    valid_dice = [value for value in dice_values if value is not None]
    total = float(matrix.sum())
    return {
        "confusion_matrix": matrix.tolist(),
        "per_class_iou": iou_values,
        "per_class_dice": dice_values,
        "mean_iou": float(np.mean(valid_iou)) if valid_iou else None,
        "mean_dice": float(np.mean(valid_dice)) if valid_dice else None,
        "pixel_accuracy": float(true_positive.sum() / total) if total else None,
    }


def binary_dilation(
    mask: NDArray[np.generic], radius: int
) -> NDArray[np.bool_]:
    """Dilate a binary mask using a square neighborhood and NumPy only.

    Args:
        mask: Two-dimensional binary-like mask.
        radius: Non-negative pixel radius. Zero returns the original foreground.

    Returns:
        Boolean dilated mask with the same shape.

    Raises:
        ValueError: If the mask is not 2-D or radius is negative.
    """

    source = np.asarray(mask).astype(bool)
    if source.ndim != 2:
        raise ValueError(f"Expected a 2-D binary mask, got shape {source.shape}.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")
    if radius == 0:
        return source.copy()

    padded = np.pad(source, radius, mode="constant", constant_values=False)
    output = np.zeros_like(source, dtype=bool)
    height, width = source.shape
    for delta_y in range(2 * radius + 1):
        for delta_x in range(2 * radius + 1):
            output |= padded[delta_y : delta_y + height, delta_x : delta_x + width]
    return output


def lane_f1_with_tolerance(
    prediction: NDArray[np.generic],
    target: NDArray[np.generic],
    tolerance: int = 2,
) -> dict[str, float | int]:
    """Compute symmetric lane precision/recall with pixel tolerance.

    A predicted lane pixel is correct when it lies within ``tolerance`` pixels
    of any target lane pixel. Recall is computed symmetrically by dilating the
    prediction. Empty prediction and target masks are treated as a perfect match.

    Args:
        prediction: Two-dimensional binary-like prediction mask.
        target: Equal-shaped binary-like ground-truth mask.
        tolerance: Non-negative tolerance radius in pixels.

    Returns:
        Dictionary with precision, recall, F1, and foreground pixel counts.

    Raises:
        ValueError: If shapes differ or inputs are not two-dimensional.
    """

    predicted = np.asarray(prediction).astype(bool)
    expected = np.asarray(target).astype(bool)
    if predicted.shape != expected.shape:
        raise ValueError(
            f"Prediction shape {predicted.shape} differs from target {expected.shape}."
        )
    if predicted.ndim != 2:
        raise ValueError("Lane masks must be two-dimensional.")

    predicted_count = int(predicted.sum())
    target_count = int(expected.sum())
    if predicted_count == 0 and target_count == 0:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "prediction_pixels": 0,
            "target_pixels": 0,
        }

    expected_dilated = binary_dilation(expected, tolerance)
    predicted_dilated = binary_dilation(predicted, tolerance)
    matched_prediction = int(np.logical_and(predicted, expected_dilated).sum())
    matched_target = int(np.logical_and(expected, predicted_dilated).sum())
    precision = matched_prediction / predicted_count if predicted_count else 0.0
    recall = matched_target / target_count if target_count else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "prediction_pixels": predicted_count,
        "target_pixels": target_count,
    }
