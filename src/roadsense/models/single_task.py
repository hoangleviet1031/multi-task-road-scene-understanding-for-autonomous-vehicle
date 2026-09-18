"""Independent detection and segmentation baselines."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights, ResNet18_Weights, resnet18
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

from roadsense.training.config import ExperimentConfig


def _normalization_groups(channels: int) -> int:
    """Choose the largest GroupNorm group count up to eight that divides channels.

    Args:
        channels: Positive channel count.

    Returns:
        Valid GroupNorm group count.
    """

    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormActivation(nn.Sequential):
    """Convolution, GroupNorm, and ReLU used by the lightweight decoder."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        """Construct a 3×3 decoder block.

        Args:
            input_channels: Number of incoming feature channels.
            output_channels: Number of outgoing feature channels.
        """

        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_normalization_groups(output_channels), output_channels),
            nn.ReLU(inplace=True),
        )


class ResNet18FPNSegmenter(nn.Module):
    """ResNet18 encoder with a four-level FPN segmentation decoder."""

    def __init__(
        self,
        num_classes: int,
        fpn_channels: int = 128,
        pretrained_backbone: bool = True,
    ) -> None:
        """Initialize a full-resolution segmentation baseline.

        Args:
            num_classes: Output channels; three for drivable and one for lane.
            fpn_channels: Shared width of P2–P5 feature maps.
            pretrained_backbone: Load ImageNet ResNet18 weights when true.

        Raises:
            ValueError: If class or FPN channel counts are not positive.
        """

        super().__init__()
        if num_classes <= 0 or fpn_channels <= 0:
            raise ValueError("num_classes and fpn_channels must be positive.")
        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = resnet18(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.lateral2 = nn.Conv2d(64, fpn_channels, 1)
        self.lateral3 = nn.Conv2d(128, fpn_channels, 1)
        self.lateral4 = nn.Conv2d(256, fpn_channels, 1)
        self.lateral5 = nn.Conv2d(512, fpn_channels, 1)
        self.smooth2 = ConvNormActivation(fpn_channels, fpn_channels)
        self.smooth3 = ConvNormActivation(fpn_channels, fpn_channels)
        self.smooth4 = ConvNormActivation(fpn_channels, fpn_channels)
        self.smooth5 = ConvNormActivation(fpn_channels, fpn_channels)
        self.decoder = nn.Sequential(
            ConvNormActivation(4 * fpn_channels, fpn_channels),
            nn.Dropout2d(p=0.1),
            ConvNormActivation(fpn_channels, fpn_channels),
            nn.Conv2d(fpn_channels, num_classes, 1),
        )
        self.num_classes = int(num_classes)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    @staticmethod
    def _upsample_add(high: torch.Tensor, lateral: torch.Tensor) -> torch.Tensor:
        """Resize a coarser map and add it to the same-width lateral map.

        Args:
            high: Coarser FPN feature.
            lateral: Finer lateral feature.

        Returns:
            Fused feature with the lateral map's spatial resolution.
        """

        return functional.interpolate(
            high, size=lateral.shape[-2:], mode="nearest"
        ) + lateral

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Predict full-resolution segmentation logits.

        Args:
            images: Float RGB tensor ``[B,3,H,W]`` in ``[0,1]``.

        Returns:
            Logits ``[B,num_classes,H,W]``.

        Raises:
            ValueError: If the image tensor is not four-dimensional RGB.
        """

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W] input, got {tuple(images.shape)}.")
        output_size = images.shape[-2:]
        normalized = (images - self.image_mean) / self.image_std
        c2 = self.layer1(self.stem(normalized))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        p5 = self.smooth5(self.lateral5(c5))
        p4 = self.smooth4(self._upsample_add(p5, self.lateral4(c4)))
        p3 = self.smooth3(self._upsample_add(p4, self.lateral3(c3)))
        p2 = self.smooth2(self._upsample_add(p3, self.lateral2(c2)))
        fused_size = p2.shape[-2:]
        fused = torch.cat(
            [
                p2,
                functional.interpolate(p3, fused_size, mode="bilinear", align_corners=False),
                functional.interpolate(p4, fused_size, mode="bilinear", align_corners=False),
                functional.interpolate(p5, fused_size, mode="bilinear", align_corners=False),
            ],
            dim=1,
        )
        logits = self.decoder(fused)
        return functional.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )


def build_detection_model(
    num_classes: int = 11,
    pretrained_backbone: bool = True,
    image_size: tuple[int, int] = (384, 640),
    trainable_backbone_layers: int = 3,
) -> nn.Module:
    """Build the CPU-friendly Torchvision Faster R-CNN baseline.

    Args:
        num_classes: Classes including background; BDD100K uses 11.
        pretrained_backbone: Load ImageNet MobileNetV3 weights when true.
        image_size: Expected letterbox ``(height,width)`` used to disable an
            unintended second resize inside Torchvision.
        trainable_backbone_layers: Number of trainable layers from the end.

    Returns:
        Faster R-CNN MobileNetV3-FPN module.
    """

    if num_classes <= 1:
        raise ValueError("Detection num_classes must include background and foreground.")
    height, width = image_size
    weights_backbone = (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    )
    arguments = {
        "weights": None,
        "weights_backbone": weights_backbone,
        "num_classes": num_classes,
        "min_size": height,
        "max_size": width,
    }
    # Torchvision deliberately trains every layer for a randomly initialized
    # backbone and warns if a partial-freezing request is supplied in that case.
    if pretrained_backbone:
        arguments["trainable_backbone_layers"] = trainable_backbone_layers
    return fasterrcnn_mobilenet_v3_large_320_fpn(
        **arguments,
    )


def build_model(config: ExperimentConfig) -> nn.Module:
    """Build the model selected by an experiment configuration.

    Args:
        config: Validated detection, drivable, or lane experiment settings.

    Returns:
        Task-appropriate PyTorch module.
    """

    if config.task == "detection":
        return build_detection_model(
            num_classes=11,
            pretrained_backbone=config.model.pretrained_backbone,
            image_size=config.data.image_size,
            trainable_backbone_layers=config.model.trainable_backbone_layers,
        )
    classes = 3 if config.task == "drivable" else 1
    return ResNet18FPNSegmenter(
        num_classes=classes,
        fpn_channels=config.model.fpn_channels,
        pretrained_backbone=config.model.pretrained_backbone,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters updated by the optimizer.

    Args:
        model: Any PyTorch module.

    Returns:
        Number of scalar parameters with ``requires_grad=True``.
    """

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
