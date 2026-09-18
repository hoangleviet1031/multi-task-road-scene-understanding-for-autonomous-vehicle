"""Typed configuration for Milestone 2 experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


VALID_TASKS = frozenset({"detection", "drivable", "lane"})


def _path(value: Any, field_name: str) -> Path:
    """Resolve a non-empty path from an experiment setting.

    Args:
        value: YAML path value.
        field_name: Field name used in validation errors.

    Returns:
        Absolute path resolved from the current working directory.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty path string.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def _optional_limit(value: Any, field_name: str) -> int | None:
    """Validate an optional positive sample limit.

    Args:
        value: YAML value or ``None``.
        field_name: Name used in errors.

    Returns:
        Positive integer or ``None``.
    """

    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"'{field_name}' must be positive when provided.")
    return result


@dataclass(frozen=True, slots=True)
class DataSettings:
    """Input data and DataLoader settings."""

    dataset_config: Path
    source_split: str
    manifest: Path
    train_key: str
    val_key: str
    image_size: tuple[int, int]
    batch_size: int
    num_workers: int
    horizontal_flip_probability: float
    max_train_samples: int | None
    max_val_samples: int | None


@dataclass(frozen=True, slots=True)
class ModelSettings:
    """Model family and backbone settings."""

    name: str
    pretrained_backbone: bool
    fpn_channels: int = 128
    trainable_backbone_layers: int = 3


@dataclass(frozen=True, slots=True)
class OptimizationSettings:
    """Optimizer and training-loop settings."""

    epochs: int
    optimizer: str
    learning_rate: float
    weight_decay: float
    momentum: float
    gradient_clip_norm: float | None
    amp: bool


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete validated single-task experiment definition."""

    source_path: Path
    task: str
    seed: int
    device: str
    output_dir: Path
    data: DataSettings
    model: ModelSettings
    optimization: OptimizationSettings
    loss: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate one Milestone 2 experiment YAML file.

    Args:
        path: Detection, drivable, lane, or smoke experiment YAML.

    Returns:
        Immutable :class:`ExperimentConfig` with typed nested settings.

    Raises:
        FileNotFoundError: If the configuration does not exist.
        ValueError: If a required field or range is invalid.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Experiment config not found: {source}")
    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Experiment config must be a YAML mapping.")

    task = str(raw.get("task", ""))
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}, got '{task}'.")
    requested_device = str(raw.get("device", "auto")).lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")

    data_raw = raw.get("data")
    model_raw = raw.get("model")
    optimization_raw = raw.get("optimization")
    if not all(isinstance(value, dict) for value in (data_raw, model_raw, optimization_raw)):
        raise ValueError("data, model, and optimization must be YAML mappings.")
    assert isinstance(data_raw, dict)
    assert isinstance(model_raw, dict)
    assert isinstance(optimization_raw, dict)

    size = data_raw.get("image_size", [384, 640])
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("data.image_size must be [height, width].")
    image_size = (int(size[0]), int(size[1]))
    if min(image_size) <= 0 or any(value % 32 != 0 for value in image_size):
        raise ValueError("image_size values must be positive multiples of 32.")

    batch_size = int(data_raw.get("batch_size", 1))
    workers = int(data_raw.get("num_workers", 0))
    flip_probability = float(data_raw.get("horizontal_flip_probability", 0.0))
    if batch_size <= 0 or workers < 0 or not 0.0 <= flip_probability <= 1.0:
        raise ValueError("Invalid batch_size, num_workers, or flip probability.")

    data = DataSettings(
        dataset_config=_path(data_raw.get("dataset_config"), "data.dataset_config"),
        source_split=str(data_raw.get("source_split", "train")),
        manifest=_path(data_raw.get("manifest"), "data.manifest"),
        train_key=str(data_raw.get("train_key", "train")),
        val_key=str(data_raw.get("val_key", "dev")),
        image_size=image_size,
        batch_size=batch_size,
        num_workers=workers,
        horizontal_flip_probability=flip_probability,
        max_train_samples=_optional_limit(
            data_raw.get("max_train_samples"), "data.max_train_samples"
        ),
        max_val_samples=_optional_limit(
            data_raw.get("max_val_samples"), "data.max_val_samples"
        ),
    )

    fpn_channels = int(model_raw.get("fpn_channels", 128))
    trainable_layers = int(model_raw.get("trainable_backbone_layers", 3))
    if fpn_channels <= 0 or not 0 <= trainable_layers <= 6:
        raise ValueError("Invalid fpn_channels or trainable_backbone_layers.")
    model = ModelSettings(
        name=str(model_raw.get("name", "")),
        pretrained_backbone=bool(model_raw.get("pretrained_backbone", True)),
        fpn_channels=fpn_channels,
        trainable_backbone_layers=trainable_layers,
    )
    expected_model = {
        "detection": "fasterrcnn_mobilenet_v3_large_320_fpn",
        "drivable": "resnet18_fpn_segmenter",
        "lane": "resnet18_fpn_segmenter",
    }[task]
    if model.name != expected_model:
        raise ValueError(
            f"task '{task}' requires model.name='{expected_model}', got '{model.name}'."
        )

    epochs = int(optimization_raw.get("epochs", 1))
    learning_rate = float(optimization_raw.get("learning_rate", 1e-3))
    weight_decay = float(optimization_raw.get("weight_decay", 0.0))
    gradient_clip = optimization_raw.get("gradient_clip_norm")
    gradient_clip_norm = None if gradient_clip is None else float(gradient_clip)
    optimizer_name = str(optimization_raw.get("optimizer", "adamw")).lower()
    if epochs <= 0 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid epochs, learning_rate, or weight_decay.")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when provided.")
    if optimizer_name not in {"adamw", "sgd"}:
        raise ValueError("optimizer must be 'adamw' or 'sgd'.")
    optimization = OptimizationSettings(
        epochs=epochs,
        optimizer=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=float(optimization_raw.get("momentum", 0.9)),
        gradient_clip_norm=gradient_clip_norm,
        amp=bool(optimization_raw.get("amp", False)),
    )

    loss = raw.get("loss", {}) or {}
    evaluation = raw.get("evaluation", {}) or {}
    if not isinstance(loss, dict) or not isinstance(evaluation, dict):
        raise ValueError("loss and evaluation must be YAML mappings.")

    return ExperimentConfig(
        source_path=source,
        task=task,
        seed=int(raw.get("seed", 42)),
        device=requested_device,
        output_dir=_path(raw.get("output_dir"), "output_dir"),
        data=data,
        model=model,
        optimization=optimization,
        loss=loss,
        evaluation=evaluation,
        raw=raw,
    )
