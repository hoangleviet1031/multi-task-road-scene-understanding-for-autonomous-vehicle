"""Focused tests for Milestone 2 data, model, and loss contracts."""

from pathlib import Path

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from roadsense.losses.single_task import DrivableLoss, LaneLoss  # noqa: E402
from roadsense.models.single_task import ResNet18FPNSegmenter  # noqa: E402
from roadsense.training.config import load_experiment_config  # noqa: E402
from roadsense.training.data import letterbox_sample  # noqa: E402


def test_experiment_config_loads_cpu_smoke_profile() -> None:
    """Checked-in smoke config resolves valid task, dimensions, and sample limits."""

    config = load_experiment_config("configs/experiments/cpu_smoke_drivable.yaml")
    assert config.task == "drivable"
    assert config.data.image_size == (128, 224)
    assert config.data.max_train_samples == 2


def test_letterbox_transforms_boxes_and_mask() -> None:
    """Letterbox keeps alignment among image, XYXY boxes, and class mask."""

    image = np.zeros((50, 100, 3), dtype=np.uint8)
    mask = np.ones((50, 100), dtype=np.uint8)
    image_tensor, boxes, transformed_mask, metadata = letterbox_sample(
        image,
        np.array([[10, 5, 90, 45]], dtype=np.float32),
        mask,
        output_size=(64, 64),
        mask_fill=2,
    )
    assert tuple(image_tensor.shape) == (3, 64, 64)
    assert tuple(transformed_mask.shape) == (64, 64)
    assert metadata.resized_size == (32, 64)
    assert metadata.offset_y == 16
    assert torch.allclose(boxes[0], torch.tensor([6.4, 19.2, 57.6, 44.8]))
    assert int(transformed_mask[0, 0]) == 2


def test_segmentation_models_and_losses_backward() -> None:
    """Both segmentation baselines preserve resolution and backpropagate."""

    images = torch.rand(2, 3, 64, 96)
    drivable_model = ResNet18FPNSegmenter(3, 32, pretrained_backbone=False)
    drivable_logits = drivable_model(images)
    assert tuple(drivable_logits.shape) == (2, 3, 64, 96)
    drivable_target = torch.randint(0, 3, (2, 64, 96))
    drivable_losses = DrivableLoss(torch.tensor([1.0, 2.0, 0.5]))(
        drivable_logits, drivable_target
    )
    drivable_losses["loss"].backward()

    lane_model = ResNet18FPNSegmenter(1, 32, pretrained_backbone=False)
    lane_logits = lane_model(images)
    assert tuple(lane_logits.shape) == (2, 1, 64, 96)
    lane_target = torch.randint(0, 2, (2, 64, 96))
    lane_losses = LaneLoss()(lane_logits, lane_target)
    lane_losses["loss"].backward()
    assert torch.isfinite(lane_losses["loss"])
