"""Synthetic end-to-end verification for Milestone 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from roadsense.data.audit import audit_dataset
from roadsense.data.bdd100k import BDD100KDataset
from roadsense.data.config import load_dataset_config
from roadsense.data.metadata import metadata_records_from_dataset
from roadsense.data.split import create_group_stratified_split
from roadsense.metrics.detection import mean_average_precision
from roadsense.metrics.segmentation import lane_f1_with_tolerance, segmentation_metrics
from roadsense.visualization import save_sample_visualization


def _write_synthetic_dataset(root: Path) -> Path:
    """Create a tiny valid BDD-like dataset and return its YAML config path.

    Args:
        root: Directory in which synthetic files are written.

    Returns:
        Path to the generated dataset configuration.
    """

    image_dir = root / "images" / "train"
    drivable_dir = root / "labels" / "drivable" / "train"
    lane_dir = root / "labels" / "lane" / "train"
    detection_path = root / "labels" / "detection_train.json"
    for directory in (image_dir, drivable_dir, lane_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    conditions = (
        ("daytime", "clear", "city street"),
        ("daytime", "rainy", "city street"),
        ("night", "clear", "highway"),
        ("night", "rainy", "residential"),
    )
    height, width = 96, 160
    for index, (timeofday, weather, scene) in enumerate(conditions):
        stem = f"synthetic_{index:03d}"
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = np.linspace(30, 100, width, dtype=np.uint8)
        image[..., 1] = 70 + index * 20
        image[..., 2] = np.linspace(140, 40, height, dtype=np.uint8)[:, None]
        Image.fromarray(image, mode="RGB").save(image_dir / f"{stem}.jpg")

        drivable = np.full((height, width), 2, dtype=np.uint8)
        drivable[height // 2 :, width // 4 : 3 * width // 4] = 0
        drivable[2 * height // 3 :, : width // 4] = 1
        Image.fromarray(drivable, mode="L").save(drivable_dir / f"{stem}.png")

        lane = np.full((height, width), 255, dtype=np.uint8)
        lane[height // 2 :, width // 2 + index] = 6
        Image.fromarray(lane, mode="L").save(lane_dir / f"{stem}.png")

        frames.append(
            {
                "name": f"{stem}.jpg",
                "videoName": f"video_{index:03d}",
                "attributes": {
                    "timeofday": timeofday,
                    "weather": weather,
                    "scene": scene,
                },
                "labels": [
                    {
                        "id": f"object_{index}",
                        "category": "car",
                        "attributes": {"occluded": False, "truncated": False},
                        "box2d": {"x1": 50, "y1": 35, "x2": 105, "y2": 78},
                    }
                ],
            }
        )

    detection_path.parent.mkdir(parents=True, exist_ok=True)
    with detection_path.open("w", encoding="utf-8") as handle:
        json.dump(frames, handle, ensure_ascii=False)

    config = {
        "root": str(root),
        "cache_dir": str(root / "cache"),
        "splits": {
            "train": {
                "images": "images/train",
                "detection_labels": "labels/detection_train.json",
                "drivable_masks": "labels/drivable/train",
                "lane_masks": "labels/lane/train",
            }
        },
        "required_tasks": ["detection", "drivable", "lane"],
        "drivable": {"direct_id": 0, "alternative_id": 1, "background_id": 2},
        "lane": {"background_bit": 3},
    }
    config_path = root / "synthetic_bdd100k.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path


def run_self_check(output_dir: str | Path) -> dict[str, Any]:
    """Run the complete Milestone 1 path against controlled synthetic data.

    Args:
        output_dir: Directory receiving the synthetic dataset, report, and image.

    Returns:
        JSON-serializable verification report. A successful run has ``ok=true``
        and perfect metrics because predictions equal their targets.

    Raises:
        RuntimeError: If any core invariant or expected perfect metric fails.
    """

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = _write_synthetic_dataset(output / "synthetic_dataset")
    dataset = BDD100KDataset(load_dataset_config(config_path), "train")
    audit = audit_dataset(dataset, limit=None)
    split = create_group_stratified_split(
        metadata_records_from_dataset(dataset), dev_ratio=0.25, seed=42
    )
    sample = dataset[0]

    if sample.drivable_mask is None or sample.lane_mask is None:
        raise RuntimeError("Synthetic sample unexpectedly lacks segmentation labels.")
    drivable_metrics = segmentation_metrics(
        sample.drivable_mask, sample.drivable_mask, num_classes=3
    )
    lane_metrics = lane_f1_with_tolerance(sample.lane_mask, sample.lane_mask, 2)
    detection_metrics = mean_average_precision(
        predictions=[
            {"boxes": sample.boxes, "labels": sample.labels, "scores": [0.99]}
        ],
        targets=[{"boxes": sample.boxes, "labels": sample.labels}],
        class_ids=[2],
        iou_thresholds=(0.5, 0.75),
    )
    overlay_path = save_sample_visualization(sample, output / "synthetic_overlay.jpg")

    checks = {
        "dataset_length_is_four": len(dataset) == 4,
        "audit_has_no_errors": bool(audit["ok"]),
        "split_has_no_sample_overlap": not bool(set(split["train"]) & set(split["dev"])),
        "drivable_miou_is_one": drivable_metrics["mean_iou"] == 1.0,
        "lane_f1_is_one": lane_metrics["f1"] == 1.0,
        "detection_map_is_one": detection_metrics["map"] == 1.0,
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "ok": all(checks.values()),
        "checks": checks,
        "audit": audit,
        "split": split,
        "metrics": {
            "drivable": drivable_metrics,
            "lane": lane_metrics,
            "detection": detection_metrics,
        },
        "artifacts": {
            "config": str(config_path),
            "overlay": str(overlay_path),
        },
    }
    report_path = output / "report.json"
    report["artifacts"]["report"] = str(report_path)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    if not report["ok"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Milestone 1 self-check failed: {failed}")
    return report
