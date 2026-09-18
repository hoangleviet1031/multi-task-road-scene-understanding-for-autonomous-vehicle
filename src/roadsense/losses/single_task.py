"""Losses for independent drivable-area and lane baselines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional
from torch import nn

from roadsense.training.config import ExperimentConfig


def multiclass_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute macro soft Dice loss for mutually exclusive classes.

    Args:
        logits: Float tensor ``[B,C,H,W]``.
        target: Integer class IDs ``[B,H,W]``.
        epsilon: Numerical stabilizer.

    Returns:
        Scalar ``1 - mean_soft_dice``.
    """

    if logits.ndim != 4 or target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError("Multiclass Dice expects [B,C,H,W] logits and [B,H,W] target.")
    probabilities = torch.softmax(logits, dim=1)
    one_hot = functional.one_hot(target.long(), num_classes=logits.shape[1])
    one_hot = one_hot.permute(0, 3, 1, 2).to(probabilities.dtype)
    dimensions = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + one_hot.sum(dim=dimensions)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def binary_soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute foreground soft Dice loss for binary lane segmentation.

    Args:
        logits: Float tensor ``[B,1,H,W]`` or ``[B,H,W]``.
        target: Binary tensor ``[B,H,W]``.
        epsilon: Numerical stabilizer.

    Returns:
        Scalar foreground Dice loss.
    """

    if logits.ndim == 4 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.shape != target.shape:
        raise ValueError("Binary Dice logits and target must have equal [B,H,W] shape.")
    probabilities = torch.sigmoid(logits)
    expected = target.to(probabilities.dtype)
    dimensions = (1, 2)
    intersection = (probabilities * expected).sum(dim=dimensions)
    denominator = probabilities.sum(dim=dimensions) + expected.sum(dim=dimensions)
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


class DrivableLoss(nn.Module):
    """Weighted cross-entropy plus macro Dice for three drivable classes."""

    def __init__(
        self,
        class_weights: torch.Tensor | None = None,
        cross_entropy_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        """Initialize the combined drivable-area loss.

        Args:
            class_weights: Optional three-element tensor for direct,
                alternative, and background classes.
            cross_entropy_weight: Contribution of pixel cross-entropy.
            dice_weight: Contribution of macro soft Dice.
        """

        super().__init__()
        if class_weights is not None and class_weights.numel() != 3:
            raise ValueError("Drivable class_weights must contain three values.")
        self.register_buffer("class_weights", class_weights)
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.dice_weight = float(dice_weight)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Calculate total and component drivable losses.

        Args:
            logits: Model output ``[B,3,H,W]``.
            target: Class IDs ``[B,H,W]``.

        Returns:
            Dictionary with ``loss``, ``cross_entropy``, and ``dice`` scalars.
        """

        cross_entropy = functional.cross_entropy(
            logits, target.long(), weight=self.class_weights
        )
        dice = multiclass_soft_dice_loss(logits, target)
        total = self.cross_entropy_weight * cross_entropy + self.dice_weight * dice
        return {"loss": total, "cross_entropy": cross_entropy, "dice": dice}


class LaneLoss(nn.Module):
    """Positive-weighted BCE plus foreground Dice for thin lane structures."""

    def __init__(
        self,
        positive_weight: float = 10.0,
        binary_cross_entropy_weight: float = 1.0,
        dice_weight: float = 1.0,
    ) -> None:
        """Initialize the combined lane loss.

        Args:
            positive_weight: BCE multiplier for rare lane foreground pixels.
            binary_cross_entropy_weight: Contribution of BCE.
            dice_weight: Contribution of foreground soft Dice.
        """

        super().__init__()
        if positive_weight <= 0:
            raise ValueError("positive_weight must be positive.")
        self.register_buffer("positive_weight", torch.tensor(float(positive_weight)))
        self.binary_cross_entropy_weight = float(binary_cross_entropy_weight)
        self.dice_weight = float(dice_weight)

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Calculate total and component lane losses.

        Args:
            logits: Model output ``[B,1,H,W]``.
            target: Binary mask ``[B,H,W]``.

        Returns:
            Dictionary with ``loss``, ``binary_cross_entropy``, and ``dice``.
        """

        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError("Lane logits must have shape [B,1,H,W].")
        squeezed = logits[:, 0]
        expected = target.to(logits.dtype)
        binary_cross_entropy = functional.binary_cross_entropy_with_logits(
            squeezed, expected, pos_weight=self.positive_weight
        )
        dice = binary_soft_dice_loss(squeezed, expected)
        total = (
            self.binary_cross_entropy_weight * binary_cross_entropy
            + self.dice_weight * dice
        )
        return {
            "loss": total,
            "binary_cross_entropy": binary_cross_entropy,
            "dice": dice,
        }


def build_loss(config: ExperimentConfig, device: torch.device) -> nn.Module | None:
    """Build the configured task loss on a target device.

    Args:
        config: Validated experiment settings.
        device: CPU or CUDA device.

    Returns:
        Drivable/lane loss module, or ``None`` for detection because Faster
        R-CNN returns its own loss dictionary during training.
    """

    settings: dict[str, Any] = config.loss
    if config.task == "detection":
        return None
    if config.task == "drivable":
        weights = torch.tensor(
            settings.get("class_weights", [1.0, 2.0, 0.5]), dtype=torch.float32
        )
        return DrivableLoss(
            class_weights=weights,
            cross_entropy_weight=float(settings.get("cross_entropy_weight", 1.0)),
            dice_weight=float(settings.get("dice_weight", 1.0)),
        ).to(device)
    return LaneLoss(
        positive_weight=float(settings.get("positive_weight", 10.0)),
        binary_cross_entropy_weight=float(
            settings.get("binary_cross_entropy_weight", 1.0)
        ),
        dice_weight=float(settings.get("dice_weight", 1.0)),
    ).to(device)
