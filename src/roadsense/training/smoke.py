"""Framework-only forward/backward verification for all Milestone 2 models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from roadsense.losses.single_task import DrivableLoss, LaneLoss
from roadsense.models.single_task import (
    ResNet18FPNSegmenter,
    build_detection_model,
    count_trainable_parameters,
)
from roadsense.training.engine import seed_everything


def _finite(value: torch.Tensor, name: str) -> float:
    """Validate a scalar tensor and convert it to a Python float.

    Args:
        value: Scalar loss tensor.
        name: Component name used in errors.

    Returns:
        Finite Python float.

    Raises:
        RuntimeError: If the value is not finite.
    """

    result = float(value.detach().cpu())
    if not torch.isfinite(value).item():
        raise RuntimeError(f"Smoke-test component '{name}' is non-finite: {result}")
    return result


def run_smoke_test(output_dir: str | Path) -> dict[str, Any]:
    """Run forward and backward passes for all three independent baselines.

    This test uses random tensors, disables pretrained downloads, and runs on
    CPU. It validates architecture/loss integration rather than model accuracy.

    Args:
        output_dir: Directory receiving ``report.json``.

    Returns:
        Report containing output shapes, finite losses, and parameter counts.
    """

    seed_everything(123)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "torch_version": torch.__version__,
        "device": "cpu",
        "tasks": {},
    }

    image_batch = torch.rand(1, 3, 64, 96)
    drivable_target = torch.randint(0, 3, (1, 64, 96), dtype=torch.int64)
    drivable_model = ResNet18FPNSegmenter(3, fpn_channels=32, pretrained_backbone=False)
    drivable_model.train()
    drivable_logits = drivable_model(image_batch)
    drivable_loss = DrivableLoss(torch.tensor([1.0, 2.0, 0.5]))(
        drivable_logits, drivable_target
    )
    drivable_loss["loss"].backward()
    report["tasks"]["drivable"] = {
        "output_shape": list(drivable_logits.shape),
        "losses": {name: _finite(value, name) for name, value in drivable_loss.items()},
        "trainable_parameters": count_trainable_parameters(drivable_model),
    }
    del drivable_model, drivable_logits, drivable_loss

    lane_target = torch.randint(0, 2, (1, 64, 96), dtype=torch.int64)
    lane_model = ResNet18FPNSegmenter(1, fpn_channels=32, pretrained_backbone=False)
    lane_model.train()
    lane_logits = lane_model(image_batch)
    lane_loss = LaneLoss(positive_weight=10.0)(lane_logits, lane_target)
    lane_loss["loss"].backward()
    report["tasks"]["lane"] = {
        "output_shape": list(lane_logits.shape),
        "losses": {name: _finite(value, name) for name, value in lane_loss.items()},
        "trainable_parameters": count_trainable_parameters(lane_model),
    }
    del lane_model, lane_logits, lane_loss

    detection_model = build_detection_model(
        num_classes=11,
        pretrained_backbone=False,
        image_size=(96, 160),
        trainable_backbone_layers=3,
    )
    detection_model.train()
    detection_image = torch.rand(3, 96, 160)
    detection_target = {
        "boxes": torch.tensor([[20.0, 20.0, 90.0, 75.0]], dtype=torch.float32),
        "labels": torch.tensor([3], dtype=torch.int64),
        "image_id": torch.tensor(0, dtype=torch.int64),
        "area": torch.tensor([3850.0], dtype=torch.float32),
        "iscrowd": torch.tensor([0], dtype=torch.int64),
    }
    detection_losses = detection_model([detection_image], [detection_target])
    detection_total = sum(detection_losses.values())
    detection_total.backward()
    report["tasks"]["detection"] = {
        "losses": {
            "loss": _finite(detection_total, "loss"),
            **{
                name: _finite(value, name)
                for name, value in detection_losses.items()
            },
        },
        "trainable_parameters": count_trainable_parameters(detection_model),
    }
    report["ok"] = True
    report_path = output / "report.json"
    report["report"] = str(report_path)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report
