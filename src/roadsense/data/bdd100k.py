"""Framework-neutral joint BDD100K dataset implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from roadsense.data.config import DatasetConfig, VALID_TASKS
from roadsense.data.detection_index import DetectionAnnotationIndex
from roadsense.data.labels import decode_drivable_mask, decode_lane_binary_mask
from roadsense.data.rasterize import load_or_rasterize_legacy_labels
from roadsense.types import DetectionFrame, RoadSceneSample


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
MASK_EXTENSIONS = frozenset({".png", ".tif", ".tiff"})


def index_files(directory: str | Path, extensions: Iterable[str]) -> dict[str, Path]:
    """Index files recursively by filename stem.

    Args:
        directory: Root directory to search.
        extensions: Case-insensitive allowed suffixes including the leading dot.

    Returns:
        Mapping from filename stem to absolute file path.

    Raises:
        FileNotFoundError: If the directory does not exist.
        ValueError: If two files share a stem, making label matching ambiguous.
    """

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")
    allowed = {suffix.lower() for suffix in extensions}
    indexed: dict[str, Path] = {}
    for candidate in root.rglob("*"):
        if not candidate.is_file() or candidate.suffix.lower() not in allowed:
            continue
        stem = candidate.stem
        if stem in indexed:
            raise ValueError(
                f"Duplicate filename stem '{stem}' in {indexed[stem]} and {candidate}."
            )
        indexed[stem] = candidate.resolve()
    return indexed


@dataclass(frozen=True, slots=True)
class DatasetInventory:
    """Counts and missing-label examples calculated before loading pixels."""

    images: int
    detection: int
    drivable: int
    lane: int
    joint_samples: int
    missing_from_detection: tuple[str, ...]
    missing_from_drivable: tuple[str, ...]
    missing_from_lane: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Convert inventory into a JSON-serializable mapping.

        Returns:
            Primitive dictionary suitable for reports.
        """

        return {
            "images": self.images,
            "detection": self.detection,
            "drivable": self.drivable,
            "lane": self.lane,
            "joint_samples": self.joint_samples,
            "missing_from_detection": list(self.missing_from_detection),
            "missing_from_drivable": list(self.missing_from_drivable),
            "missing_from_lane": list(self.missing_from_lane),
        }


class BDD100KDataset:
    """Lazy joint dataset for BDD100K images and three perception tasks."""

    def __init__(
        self,
        config: DatasetConfig,
        split: str,
        required_tasks: Iterable[str] | None = None,
    ) -> None:
        """Index a configured BDD100K split.

        The dataset contains the intersection of image IDs and every requested
        task. Images and masks remain lazy and are read only in ``__getitem__``.

        Args:
            config: Validated dataset configuration.
            split: Configured split name.
            required_tasks: Tasks required for membership. Defaults to the YAML
                setting and accepts ``detection``, ``drivable``, and ``lane``.

        Raises:
            ValueError: If a task name is unsupported or no joint samples exist.
            FileNotFoundError: If a required source path is absent.
        """

        self.config = config
        self.split = split
        self.paths = config.for_split(split)
        tasks = tuple(required_tasks or config.required_tasks)
        unknown = set(tasks) - VALID_TASKS
        if unknown:
            raise ValueError(f"Unsupported required tasks: {sorted(unknown)}")
        self.required_tasks = tasks

        self.image_paths = index_files(self.paths.images, IMAGE_EXTENSIONS)
        self.drivable_paths: dict[str, Path] = {}
        self.lane_paths: dict[str, Path] = {}
        self.detection_index: DetectionAnnotationIndex | None = None

        if self.paths.drivable_masks is not None and self.paths.drivable_masks.is_dir():
            self.drivable_paths = index_files(self.paths.drivable_masks, MASK_EXTENSIONS)

        if self.paths.lane_masks is not None and self.paths.lane_masks.is_dir():
            self.lane_paths = index_files(self.paths.lane_masks, MASK_EXTENSIONS)

        detection_stems: set[str] = set()
        if (
            self.paths.detection_labels is not None
            and self.paths.detection_labels.is_file()
        ):
            cache_path = config.cache_dir / f"detection_{split}.sqlite"
            self.detection_index = DetectionAnnotationIndex(
                self.paths.detection_labels, cache_path
            )
            detection_stems = self.detection_index.stems()
        elif "detection" in tasks:
            raise FileNotFoundError(
                f"Required detection JSON not found: {self.paths.detection_labels}"
            )

        if "drivable" in tasks and not self.drivable_paths and self.detection_index is None:
            raise FileNotFoundError(
                "Drivable masks are missing and no polygon label JSON is available."
            )
        if "lane" in tasks and not self.lane_paths and self.detection_index is None:
            raise FileNotFoundError(
                "Lane masks are missing and no polygon label JSON is available."
            )

        effective_drivable_stems = (
            set(self.drivable_paths) if self.drivable_paths else detection_stems
        )
        effective_lane_stems = set(self.lane_paths) if self.lane_paths else detection_stems

        required_sets: list[set[str]] = [set(self.image_paths)]
        for task in tasks:
            if task == "detection":
                required_sets.append(detection_stems)
            elif task == "drivable":
                required_sets.append(effective_drivable_stems)
            elif task == "lane":
                required_sets.append(effective_lane_stems)
        names = set.intersection(*required_sets)
        self.names = tuple(sorted(names))
        if not self.names:
            raise ValueError(
                f"No joint samples found for split '{split}' and tasks {tasks}. "
                "Inspect the configured paths and filename stems."
            )

        image_stems = set(self.image_paths)
        self.inventory = DatasetInventory(
            images=len(self.image_paths),
            detection=len(detection_stems),
            drivable=len(effective_drivable_stems),
            lane=len(effective_lane_stems),
            joint_samples=len(self.names),
            missing_from_detection=tuple(sorted(image_stems - detection_stems)[:20]),
            missing_from_drivable=tuple(
                sorted(image_stems - effective_drivable_stems)[:20]
            ),
            missing_from_lane=tuple(sorted(image_stems - effective_lane_stems)[:20]),
        )

    def __len__(self) -> int:
        """Return the number of joint samples.

        Returns:
            Number of image IDs satisfying all required tasks.
        """

        return len(self.names)

    def detection_frame(self, name: str) -> DetectionFrame | None:
        """Fetch normalized detection annotations for a sample ID.

        Args:
            name: Filename stem.

        Returns:
            Detection frame, or ``None`` when detection labels are unavailable.
        """

        return None if self.detection_index is None else self.detection_index.get(name)

    def __getitem__(self, index: int) -> RoadSceneSample:
        """Load one image and every available task label.

        Args:
            index: Zero-based integer index. Negative Python indices are allowed.

        Returns:
            A framework-neutral :class:`RoadSceneSample`.

        Raises:
            IndexError: If ``index`` is outside the dataset.
            ValueError: If an official mask contains unsupported values.
        """

        name = self.names[index]
        with Image.open(self.image_paths[name]) as image_file:
            image = np.asarray(image_file.convert("RGB"), dtype=np.uint8)

        frame = self.detection_frame(name)
        if frame is None:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)
            attributes: dict[str, object] = {}
            sequence_id = name
        else:
            boxes = np.asarray([item.box for item in frame.detections], dtype=np.float32)
            boxes = (
                np.empty((0, 4), dtype=np.float32)
                if boxes.size == 0
                else boxes.reshape(-1, 4)
            )
            labels = np.asarray(
                [item.label for item in frame.detections], dtype=np.int64
            )
            attributes = dict(frame.attributes)
            sequence_id = frame.sequence_id

        drivable_mask = None
        if name in self.drivable_paths:
            with Image.open(self.drivable_paths[name]) as mask_file:
                raw_drivable = np.asarray(mask_file, dtype=np.uint8)
            drivable_mask = decode_drivable_mask(
                raw_drivable,
                direct_id=self.config.direct_id,
                alternative_id=self.config.alternative_id,
                background_id=self.config.background_id,
            )

        lane_mask = None
        if name in self.lane_paths:
            with Image.open(self.lane_paths[name]) as mask_file:
                raw_lane = np.asarray(mask_file, dtype=np.uint8)
            lane_mask = decode_lane_binary_mask(
                raw_lane, background_bit=self.config.lane_background_bit
            )

        if frame is not None and (drivable_mask is None or lane_mask is None):
            rendered_drivable, rendered_lane = load_or_rasterize_legacy_labels(
                frame,
                image_size=(image.shape[1], image.shape[0]),
                cache_root=self.config.cache_dir / "rasterized" / self.split,
            )
            if drivable_mask is None:
                drivable_mask = rendered_drivable
            if lane_mask is None:
                lane_mask = rendered_lane

        return RoadSceneSample(
            name=name,
            image=np.ascontiguousarray(image),
            boxes=boxes,
            labels=labels,
            drivable_mask=drivable_mask,
            lane_mask=lane_mask,
            attributes=attributes,
            sequence_id=sequence_id,
            task_available={
                "detection": frame is not None,
                "drivable": drivable_mask is not None,
                "lane": lane_mask is not None,
            },
        )

    def metadata_records(self) -> list[dict[str, str]]:
        """Return records used for leakage-safe stratified splitting.

        Returns:
            List containing sample name, sequence ID, weather, scene, and time.
        """

        records: list[dict[str, str]] = []
        for name in self.names:
            frame = self.detection_frame(name)
            attributes = {} if frame is None else frame.attributes
            records.append(
                {
                    "name": name,
                    "sequence_id": name if frame is None else frame.sequence_id,
                    "weather": str(attributes.get("weather", "undefined")),
                    "scene": str(attributes.get("scene", "undefined")),
                    "timeofday": str(attributes.get("timeofday", "undefined")),
                }
            )
        return records
