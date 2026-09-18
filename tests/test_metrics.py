"""Unit tests for transparent Milestone 1 metrics."""

import numpy as np

from roadsense.metrics.detection import box_iou, mean_average_precision
from roadsense.metrics.segmentation import lane_f1_with_tolerance, segmentation_metrics


def test_segmentation_metrics_perfect_prediction() -> None:
    """An identical three-class mask must score one on every aggregate metric."""

    target = np.array([[0, 1], [2, 2]], dtype=np.uint8)
    result = segmentation_metrics(target, target, num_classes=3)
    assert result["mean_iou"] == 1.0
    assert result["mean_dice"] == 1.0
    assert result["pixel_accuracy"] == 1.0


def test_lane_f1_accepts_one_pixel_offset_with_tolerance() -> None:
    """A one-pixel lane shift is a match when tolerance is one pixel."""

    target = np.zeros((5, 5), dtype=np.uint8)
    prediction = np.zeros_like(target)
    target[:, 2] = 1
    prediction[:, 3] = 1
    assert lane_f1_with_tolerance(prediction, target, tolerance=1)["f1"] == 1.0
    assert lane_f1_with_tolerance(prediction, target, tolerance=0)["f1"] == 0.0


def test_box_iou_known_value() -> None:
    """Two partially overlapping boxes produce the analytically known IoU."""

    result = box_iou([[0, 0, 10, 10]], [[5, 5, 15, 15]])
    assert np.isclose(result[0, 0], 25 / 175)


def test_detection_map_perfect_prediction() -> None:
    """One exact high-confidence match has AP and mAP equal to one."""

    result = mean_average_precision(
        predictions=[
            {
                "boxes": np.array([[0, 0, 10, 10]], dtype=float),
                "labels": np.array([2]),
                "scores": np.array([0.9]),
            }
        ],
        targets=[
            {
                "boxes": np.array([[0, 0, 10, 10]], dtype=float),
                "labels": np.array([2]),
            }
        ],
        class_ids=[2],
        iou_thresholds=[0.5, 0.75],
    )
    assert result["map"] == 1.0
    assert result["map_by_threshold"]["map@0.50"] == 1.0
