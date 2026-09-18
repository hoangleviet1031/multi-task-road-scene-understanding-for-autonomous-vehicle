"""PyTorch adapters for Milestone 1 road-scene samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from roadsense.data.bdd100k import BDD100KDataset
from roadsense.data.config import load_dataset_config
from roadsense.training.config import ExperimentConfig


@dataclass(frozen=True, slots=True)
class LetterboxMetadata:
    """Geometry needed to map resized predictions back to source pixels."""

    original_size: tuple[int, int]
    output_size: tuple[int, int]
    resized_size: tuple[int, int]
    scale_x: float
    scale_y: float
    offset_x: int
    offset_y: int


def letterbox_sample(
    image: np.ndarray,
    boxes: np.ndarray | None,
    mask: np.ndarray | None,
    output_size: tuple[int, int],
    mask_fill: int = 0,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, LetterboxMetadata]:
    """Resize one image with preserved aspect ratio and symmetric padding.

    Args:
        image: RGB ``uint8[H,W,3]`` array.
        boxes: Optional XYXY array shaped ``[N,4]``.
        mask: Optional integer mask shaped ``[H,W]``.
        output_size: Target ``(height, width)``.
        mask_fill: Class ID used in padded mask pixels.

    Returns:
        Float image tensor ``[3,Hout,Wout]`` in ``[0,1]``, transformed boxes,
        transformed integer mask, and geometry metadata.

    Raises:
        ValueError: If shapes or output dimensions are invalid.
    """

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {array.shape}.")
    source_height, source_width = array.shape[:2]
    output_height, output_width = (int(value) for value in output_size)
    if min(source_height, source_width, output_height, output_width) <= 0:
        raise ValueError("Image and output dimensions must be positive.")
    if mask is not None and np.asarray(mask).shape != (source_height, source_width):
        raise ValueError("Mask shape must match the source image.")

    # PIL-backed arrays can be C-contiguous but read-only. ``ascontiguousarray``
    # may therefore return the same read-only view, while PyTorch tensors require
    # writable storage. An explicit copy makes ownership and mutability safe.
    image_tensor = torch.from_numpy(np.array(array, copy=True, order="C")).permute(2, 0, 1)
    image_tensor = image_tensor.to(dtype=torch.float32).div_(255.0)
    scale = min(output_width / source_width, output_height / source_height)
    resized_width = max(1, min(output_width, round(source_width * scale)))
    resized_height = max(1, min(output_height, round(source_height * scale)))
    resized = functional.interpolate(
        image_tensor.unsqueeze(0),
        size=(resized_height, resized_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    offset_x = (output_width - resized_width) // 2
    offset_y = (output_height - resized_height) // 2
    canvas = torch.full(
        (3, output_height, output_width), 114.0 / 255.0, dtype=torch.float32
    )
    canvas[
        :, offset_y : offset_y + resized_height, offset_x : offset_x + resized_width
    ] = resized

    transformed_boxes: torch.Tensor | None = None
    scale_x = resized_width / source_width
    scale_y = resized_height / source_height
    if boxes is not None:
        box_array = np.asarray(boxes, dtype=np.float32)
        if box_array.size == 0:
            transformed_boxes = torch.empty((0, 4), dtype=torch.float32)
        else:
            if box_array.ndim != 2 or box_array.shape[1] != 4:
                raise ValueError(f"boxes must have shape [N,4], got {box_array.shape}.")
            transformed_boxes = torch.from_numpy(np.ascontiguousarray(box_array)).clone()
            transformed_boxes[:, [0, 2]] = (
                transformed_boxes[:, [0, 2]] * scale_x + offset_x
            )
            transformed_boxes[:, [1, 3]] = (
                transformed_boxes[:, [1, 3]] * scale_y + offset_y
            )
            transformed_boxes[:, [0, 2]].clamp_(0, output_width)
            transformed_boxes[:, [1, 3]].clamp_(0, output_height)

    transformed_mask: torch.Tensor | None = None
    if mask is not None:
        source_mask = torch.from_numpy(
            np.array(mask, copy=True, order="C")
        ).to(torch.float32)
        resized_mask = functional.interpolate(
            source_mask[None, None],
            size=(resized_height, resized_width),
            mode="nearest",
        )[0, 0].to(torch.int64)
        transformed_mask = torch.full(
            (output_height, output_width), int(mask_fill), dtype=torch.int64
        )
        transformed_mask[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized_mask

    metadata = LetterboxMetadata(
        original_size=(source_height, source_width),
        output_size=(output_height, output_width),
        resized_size=(resized_height, resized_width),
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=offset_x,
        offset_y=offset_y,
    )
    return canvas, transformed_boxes, transformed_mask, metadata


def _horizontal_flip(
    image: np.ndarray,
    boxes: np.ndarray,
    mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Flip aligned image, boxes, and optional mask horizontally.

    Args:
        image: RGB array.
        boxes: XYXY boxes.
        mask: Optional aligned segmentation mask.

    Returns:
        Contiguous flipped image, boxes, and mask.
    """

    width = image.shape[1]
    flipped_image = np.ascontiguousarray(image[:, ::-1])
    flipped_boxes = np.asarray(boxes, dtype=np.float32).copy()
    if flipped_boxes.size:
        old_x1 = flipped_boxes[:, 0].copy()
        old_x2 = flipped_boxes[:, 2].copy()
        flipped_boxes[:, 0] = width - old_x2
        flipped_boxes[:, 2] = width - old_x1
    flipped_mask = None if mask is None else np.ascontiguousarray(mask[:, ::-1])
    return flipped_image, flipped_boxes, flipped_mask


class SingleTaskDataset(Dataset):
    """Task-specific PyTorch view over a Milestone 1 BDD100K dataset."""

    def __init__(
        self,
        base: BDD100KDataset,
        task: str,
        sample_names: Sequence[str],
        image_size: tuple[int, int],
        training: bool,
        horizontal_flip_probability: float = 0.0,
        limit: int | None = None,
    ) -> None:
        """Create a deterministic subset for one task.

        Args:
            base: Joint framework-neutral dataset.
            task: ``detection``, ``drivable``, or ``lane``.
            sample_names: Ordered IDs from the split manifest.
            image_size: Letterbox output ``(height,width)``.
            training: Enables random augmentation when true.
            horizontal_flip_probability: Training flip probability.
            limit: Optional deterministic prefix length for smoke runs.

        Raises:
            ValueError: If task or sample names are invalid.
        """

        if task not in {"detection", "drivable", "lane"}:
            raise ValueError(f"Unsupported single task: {task}")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0,1].")
        index_by_name = {name: index for index, name in enumerate(base.names)}
        missing = [name for name in sample_names if name not in index_by_name]
        if missing:
            raise ValueError(
                f"Manifest contains {len(missing)} names absent from the dataset; "
                f"examples: {missing[:5]}"
            )
        selected = list(sample_names)
        if limit is not None:
            selected = selected[:limit]
        if not selected:
            raise ValueError("SingleTaskDataset cannot be empty.")
        self.base = base
        self.task = task
        self.names = tuple(selected)
        self.indices = tuple(index_by_name[name] for name in selected)
        self.image_size = image_size
        self.training = bool(training)
        self.horizontal_flip_probability = float(horizontal_flip_probability)

    def __len__(self) -> int:
        """Return the number of selected manifest samples."""

        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        """Return one detection or segmentation training item.

        Args:
            index: Dataset-local integer index.

        Returns:
            Detection: image plus target dictionary containing boxes, one-based
            labels, image ID, area, and crowd flags. Segmentation: image plus a
            long class mask.
        """

        sample = self.base[self.indices[index]]
        mask: np.ndarray | None
        mask_fill: int
        if self.task == "drivable":
            mask = sample.drivable_mask
            mask_fill = 2
        elif self.task == "lane":
            mask = sample.lane_mask
            mask_fill = 0
        else:
            mask = None
            mask_fill = 0
        if self.task != "detection" and mask is None:
            raise RuntimeError(f"Sample {sample.name} lacks the {self.task} mask.")

        image = sample.image
        boxes = sample.boxes
        if self.training and torch.rand(()) < self.horizontal_flip_probability:
            image, boxes, mask = _horizontal_flip(image, boxes, mask)

        image_tensor, transformed_boxes, target_mask, _ = letterbox_sample(
            image=image,
            boxes=boxes if self.task == "detection" else None,
            mask=mask,
            output_size=self.image_size,
            mask_fill=mask_fill,
        )
        if self.task != "detection":
            assert target_mask is not None
            return image_tensor, target_mask

        assert transformed_boxes is not None
        widths = transformed_boxes[:, 2] - transformed_boxes[:, 0]
        heights = transformed_boxes[:, 3] - transformed_boxes[:, 1]
        keep = (widths > 1.0) & (heights > 1.0)
        transformed_boxes = transformed_boxes[keep]
        labels = torch.from_numpy(np.ascontiguousarray(sample.labels)).to(torch.int64)
        labels = labels[keep] + 1  # Torchvision reserves class zero for background.
        area = (
            (transformed_boxes[:, 2] - transformed_boxes[:, 0])
            * (transformed_boxes[:, 3] - transformed_boxes[:, 1])
        )
        target = {
            "boxes": transformed_boxes,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate(
    batch: Sequence[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Collate variable-length detection targets into two Python lists.

    Args:
        batch: Detection dataset items.

    Returns:
        Image list and target dictionary list expected by Torchvision detection.
    """

    images, targets = zip(*batch)
    return list(images), list(targets)


def _read_manifest(path: Path, key: str) -> list[str]:
    """Read one ordered sample-name list from a split manifest.

    Args:
        path: Milestone 1 JSON manifest.
        key: Manifest list key such as ``train`` or ``dev``.

    Returns:
        List of sample IDs.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    values = manifest.get(key)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Manifest key '{key}' must contain a string list.")
    return values


def build_dataloaders(
    config: ExperimentConfig,
) -> tuple[DataLoader, DataLoader]:
    """Build deterministic training and validation DataLoaders.

    Args:
        config: Validated single-task experiment configuration.

    Returns:
        Tuple ``(train_loader, val_loader)`` using the Milestone 1 manifest.
    """

    base = BDD100KDataset(
        load_dataset_config(config.data.dataset_config),
        config.data.source_split,
    )
    train_names = _read_manifest(config.data.manifest, config.data.train_key)
    val_names = _read_manifest(config.data.manifest, config.data.val_key)
    train_dataset = SingleTaskDataset(
        base=base,
        task=config.task,
        sample_names=train_names,
        image_size=config.data.image_size,
        training=True,
        horizontal_flip_probability=config.data.horizontal_flip_probability,
        limit=config.data.max_train_samples,
    )
    val_dataset = SingleTaskDataset(
        base=base,
        task=config.task,
        sample_names=val_names,
        image_size=config.data.image_size,
        training=False,
        horizontal_flip_probability=0.0,
        limit=config.data.max_val_samples,
    )
    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.data.batch_size,
        "num_workers": config.data.num_workers,
        "pin_memory": False,
        "persistent_workers": config.data.num_workers > 0,
    }
    collate = detection_collate if config.task == "detection" else None
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate,
        **common,
    )
    return train_loader, val_loader
