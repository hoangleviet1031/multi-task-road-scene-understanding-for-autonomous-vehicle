"""Framework-neutral evaluation metrics used from Milestone 1 onward."""

from roadsense.metrics.detection import box_iou, mean_average_precision
from roadsense.metrics.segmentation import (
    confusion_matrix,
    lane_f1_with_tolerance,
    segmentation_metrics,
)

__all__ = [
    "box_iou",
    "confusion_matrix",
    "lane_f1_with_tolerance",
    "mean_average_precision",
    "segmentation_metrics",
]
