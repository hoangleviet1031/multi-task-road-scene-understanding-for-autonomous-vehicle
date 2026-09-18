"""YAML configuration loading with explicit path validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_TASKS = frozenset({"detection", "drivable", "lane"})


@dataclass(frozen=True, slots=True)
class SplitPaths:
    """Absolute input locations for one dataset split."""

    images: Path
    detection_labels: Path | None
    drivable_masks: Path | None
    lane_masks: Path | None


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Validated BDD100K dataset configuration.

    Attributes:
        source_path: YAML file from which this configuration was loaded.
        root: Absolute dataset root.
        cache_dir: Absolute directory used for disposable indices.
        splits: Mapping from split names to absolute input paths.
        required_tasks: Tasks that must be available for a joint sample.
        direct_id: Raw drivable-mask value for directly drivable pixels.
        alternative_id: Raw value for alternatively drivable pixels.
        background_id: Raw value for background pixels.
        lane_background_bit: Bit position marking lane background.
    """

    source_path: Path
    root: Path
    cache_dir: Path
    splits: dict[str, SplitPaths]
    required_tasks: tuple[str, ...]
    direct_id: int = 0
    alternative_id: int = 1
    background_id: int = 2
    lane_background_bit: int = 3

    def for_split(self, split: str) -> SplitPaths:
        """Return paths for ``split`` or raise a descriptive error.

        Args:
            split: Configured split name, for example ``"train"``.

        Returns:
            The corresponding :class:`SplitPaths` object.

        Raises:
            KeyError: If the split is not present in the YAML file.
        """

        try:
            return self.splits[split]
        except KeyError as exc:
            available = ", ".join(sorted(self.splits)) or "<none>"
            raise KeyError(
                f"Unknown split '{split}'. Configured splits: {available}."
            ) from exc


def _resolve_from_root(root: Path, value: Any, field_name: str) -> Path | None:
    """Resolve an optional configuration value relative to the dataset root.

    Args:
        root: Absolute dataset root.
        value: YAML value containing a relative or absolute path, or ``None``.
        field_name: Name used in validation errors.

    Returns:
        A resolved absolute path, or ``None`` when the field is omitted.
    """

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{field_name}' must be a non-empty path string.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def load_dataset_config(path: str | Path) -> DatasetConfig:
    """Load and validate a RoadSense BDD100K YAML configuration.

    Relative ``root`` and ``cache_dir`` entries are interpreted relative to the
    current working directory. Paths inside each split are interpreted relative
    to ``root``.

    Args:
        path: YAML configuration file.

    Returns:
        A validated :class:`DatasetConfig` with absolute paths.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValueError: If required keys or supported task names are invalid.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Dataset config not found: {source}")

    with source.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("Dataset config must contain a YAML mapping.")

    root_value = raw.get("root")
    if not isinstance(root_value, str) or not root_value.strip():
        raise ValueError("Dataset config requires a non-empty 'root' path.")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()

    cache_value = raw.get("cache_dir", "data/interim/bdd100k")
    cache_dir = Path(str(cache_value)).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = Path.cwd() / cache_dir
    cache_dir = cache_dir.resolve()

    raw_splits = raw.get("splits")
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise ValueError("Dataset config requires at least one entry in 'splits'.")

    splits: dict[str, SplitPaths] = {}
    for split_name, split_raw in raw_splits.items():
        if not isinstance(split_raw, dict):
            raise ValueError(f"Split '{split_name}' must be a YAML mapping.")
        images = _resolve_from_root(root, split_raw.get("images"), "images")
        if images is None:
            raise ValueError(f"Split '{split_name}' requires an images path.")
        splits[str(split_name)] = SplitPaths(
            images=images,
            detection_labels=_resolve_from_root(
                root, split_raw.get("detection_labels"), "detection_labels"
            ),
            drivable_masks=_resolve_from_root(
                root, split_raw.get("drivable_masks"), "drivable_masks"
            ),
            lane_masks=_resolve_from_root(
                root, split_raw.get("lane_masks"), "lane_masks"
            ),
        )

    task_values = raw.get("required_tasks", sorted(VALID_TASKS))
    if not isinstance(task_values, list) or not task_values:
        raise ValueError("'required_tasks' must be a non-empty YAML list.")
    required_tasks = tuple(str(task) for task in task_values)
    unknown_tasks = set(required_tasks) - VALID_TASKS
    if unknown_tasks:
        raise ValueError(f"Unsupported required tasks: {sorted(unknown_tasks)}")

    drivable = raw.get("drivable", {}) or {}
    lane = raw.get("lane", {}) or {}
    if not isinstance(drivable, dict) or not isinstance(lane, dict):
        raise ValueError("'drivable' and 'lane' settings must be mappings.")

    background_bit = int(lane.get("background_bit", 3))
    if not 0 <= background_bit <= 7:
        raise ValueError("lane.background_bit must be between 0 and 7.")

    return DatasetConfig(
        source_path=source,
        root=root,
        cache_dir=cache_dir,
        splits=splits,
        required_tasks=required_tasks,
        direct_id=int(drivable.get("direct_id", 0)),
        alternative_id=int(drivable.get("alternative_id", 1)),
        background_id=int(drivable.get("background_id", 2)),
        lane_background_bit=background_bit,
    )
