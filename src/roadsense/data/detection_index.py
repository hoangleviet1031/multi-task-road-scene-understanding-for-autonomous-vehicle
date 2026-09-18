"""Memory-safe indexing of BDD100K detection annotations."""

from __future__ import annotations

import json
import math
import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Iterator

import ijson

from roadsense.data.labels import DETECTION_CLASS_TO_ID, safe_attributes
from roadsense.types import Detection, DetectionFrame


INDEX_SCHEMA_VERSION = 2


def _as_finite_float(value: Any) -> float | None:
    """Convert a JSON value to a finite float.

    Args:
        value: Candidate numeric value.

    Returns:
        A finite float, or ``None`` when conversion is impossible.
    """

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalize_poly2d(value: Any) -> list[dict[str, Any]]:
    """Normalize Scalabel poly2d data and convert Decimal coordinates to float.

    Args:
        value: Raw ``poly2d`` JSON value.

    Returns:
        JSON-serializable paths with vertices, command types, and closed flags.
    """

    if not isinstance(value, list):
        return []
    output: list[dict[str, Any]] = []
    for poly in value:
        if not isinstance(poly, dict) or not isinstance(poly.get("vertices"), list):
            continue
        vertices: list[list[float]] = []
        for vertex in poly["vertices"]:
            if not isinstance(vertex, (list, tuple)) or len(vertex) < 2:
                continue
            x = _as_finite_float(vertex[0])
            y = _as_finite_float(vertex[1])
            if x is not None and y is not None:
                vertices.append([x, y])
        if not vertices:
            continue
        output.append(
            {
                "vertices": vertices,
                "types": str(poly.get("types", "L" * len(vertices))),
                "closed": bool(poly.get("closed", False)),
            }
        )
    return output


def _extract_sequence_id(frame: dict[str, Any], stem: str) -> str:
    """Choose a stable sequence identifier from a BDD frame.

    Args:
        frame: Raw BDD frame dictionary.
        stem: Fallback image stem.

    Returns:
        A non-empty grouping key. Official BDD100K keyframes fall back to their
        unique stem because each labeled keyframe corresponds to one video.
    """

    attributes = safe_attributes(frame.get("attributes"))
    candidates = (
        frame.get("videoName"),
        frame.get("video_name"),
        frame.get("sequence_id"),
        frame.get("sequence"),
        attributes.get("videoName"),
        attributes.get("sequence_id"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return stem


def normalize_detection_frame(frame: dict[str, Any]) -> DetectionFrame:
    """Normalize current or legacy BDD100K detection JSON for one frame.

    Current Scalabel records store objects under ``labels``. Legacy BDD100K
    records may contain a ``frames`` list whose first item stores ``objects``.
    Only the ten official detection categories are retained. Malformed boxes are
    ignored; geometrically invalid but numeric boxes are retained so the audit
    can report them.

    Args:
        frame: One JSON-derived BDD100K frame mapping.

    Returns:
        A normalized :class:`DetectionFrame`.

    Raises:
        ValueError: If an image name cannot be determined.
    """

    if not isinstance(frame, dict):
        raise ValueError("BDD100K frame must be a JSON object.")

    source = frame
    labels: Any = frame.get("labels")
    if labels is None and isinstance(frame.get("frames"), list) and frame["frames"]:
        nested = frame["frames"][0]
        if isinstance(nested, dict):
            source = nested
            labels = nested.get("labels", nested.get("objects"))
    if labels is None:
        labels = source.get("objects", [])

    name_value = source.get("name") or frame.get("name")
    if not isinstance(name_value, str) or not name_value.strip():
        raise ValueError("BDD100K frame is missing a valid image name.")
    name = Path(name_value).name
    stem = Path(name).stem

    frame_attributes = safe_attributes(frame.get("attributes"))
    frame_attributes.update(safe_attributes(source.get("attributes")))
    sequence_id = _extract_sequence_id({**frame, **source}, stem)

    detections: list[Detection] = []
    segmentations: list[dict[str, Any]] = []
    if isinstance(labels, list):
        for label in labels:
            if not isinstance(label, dict):
                continue
            category = label.get("category")
            if not isinstance(category, str):
                continue
            normalized_polygons = _normalize_poly2d(label.get("poly2d"))
            if category in {"drivable area", "lane"} and normalized_polygons:
                segmentations.append(
                    {
                        "category": category,
                        "attributes": safe_attributes(label.get("attributes")),
                        "poly2d": normalized_polygons,
                    }
                )
            if category not in DETECTION_CLASS_TO_ID:
                continue
            box = label.get("box2d")
            if not isinstance(box, dict):
                continue
            coordinates = tuple(
                _as_finite_float(box.get(key)) for key in ("x1", "y1", "x2", "y2")
            )
            if any(value is None for value in coordinates):
                continue
            x1, y1, x2, y2 = coordinates
            assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    label=DETECTION_CLASS_TO_ID[category],
                    category=category,
                    attributes=safe_attributes(label.get("attributes")),
                )
            )

    return DetectionFrame(
        name=name,
        stem=stem,
        sequence_id=sequence_id,
        attributes=frame_attributes,
        detections=tuple(detections),
        segmentations=tuple(segmentations),
    )


def iter_detection_json(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream top-level BDD100K frames without loading the JSON into memory.

    Args:
        path: JSON file containing either a root array or ``{"frames": [...]}``.

    Yields:
        One frame dictionary at a time.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the root is neither a JSON array nor an object.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Detection label file not found: {source}")

    with source.open("rb") as handle:
        first = b""
        while not first:
            chunk = handle.read(1)
            if not chunk:
                raise ValueError(f"Detection label file is empty: {source}")
            if not chunk.isspace():
                first = chunk
        handle.seek(0)
        if first == b"[":
            iterator = ijson.items(handle, "item")
        elif first == b"{":
            iterator = ijson.items(handle, "frames.item")
        else:
            raise ValueError(f"Unsupported JSON root in {source}")
        for item in iterator:
            if isinstance(item, dict):
                yield item


def _frame_to_json(frame: DetectionFrame) -> tuple[str, str, str, str, str, str]:
    """Serialize a normalized frame into one SQLite row.

    Args:
        frame: Normalized detection frame.

    Returns:
        Tuple containing stem, filename, sequence ID, attributes JSON, and
        detections JSON.
    """

    detections = [
        {
            "box": list(item.box),
            "label": item.label,
            "category": item.category,
            "attributes": item.attributes,
        }
        for item in frame.detections
    ]
    return (
        frame.stem,
        frame.name,
        frame.sequence_id,
        json.dumps(frame.attributes, ensure_ascii=False, separators=(",", ":")),
        json.dumps(detections, ensure_ascii=False, separators=(",", ":")),
        json.dumps(frame.segmentations, ensure_ascii=False, separators=(",", ":")),
    )


def _row_to_frame(row: sqlite3.Row) -> DetectionFrame:
    """Deserialize one SQLite row into a detection frame.

    Args:
        row: SQLite row produced by this module's schema.

    Returns:
        Reconstructed :class:`DetectionFrame`.
    """

    raw_detections = json.loads(row["detections_json"])
    detections = tuple(
        Detection(
            box=tuple(float(value) for value in item["box"]),  # type: ignore[arg-type]
            label=int(item["label"]),
            category=str(item["category"]),
            attributes=safe_attributes(item.get("attributes")),
        )
        for item in raw_detections
    )
    return DetectionFrame(
        name=str(row["image_name"]),
        stem=str(row["stem"]),
        sequence_id=str(row["sequence_id"]),
        attributes=safe_attributes(json.loads(row["attributes_json"])),
        detections=detections,
        segmentations=tuple(json.loads(row["segmentations_json"])),
    )


class DetectionAnnotationIndex:
    """SQLite-backed random-access index for a large BDD100K JSON file."""

    def __init__(self, source_path: str | Path, cache_path: str | Path) -> None:
        """Create an index descriptor without immediately parsing annotations.

        Args:
            source_path: Official BDD100K detection JSON.
            cache_path: Disposable SQLite cache generated from ``source_path``.
        """

        self.source_path = Path(source_path).expanduser().resolve()
        self.cache_path = Path(cache_path).expanduser().resolve()

    def _source_fingerprint(self) -> dict[str, str]:
        """Return stable metadata used to invalidate a stale cache.

        Returns:
            Mapping containing schema version, byte size, and nanosecond mtime.
        """

        stat = self.source_path.stat()
        return {
            "schema_version": str(INDEX_SCHEMA_VERSION),
            "source_size": str(stat.st_size),
            "source_mtime_ns": str(stat.st_mtime_ns),
        }

    def _is_current(self) -> bool:
        """Check whether the cache matches the current source file.

        Returns:
            ``True`` only when the cache exists and its fingerprint matches.
        """

        if not self.cache_path.is_file() or not self.source_path.is_file():
            return False
        try:
            with closing(sqlite3.connect(self.cache_path)) as connection:
                rows = connection.execute("SELECT key, value FROM metadata").fetchall()
        except sqlite3.Error:
            return False
        return dict(rows) == self._source_fingerprint()

    def ensure_built(self, rebuild: bool = False) -> int:
        """Build or validate the streaming SQLite annotation index.

        Args:
            rebuild: Force regeneration even when the fingerprint is current.

        Returns:
            Number of indexed frames.

        Raises:
            FileNotFoundError: If the source JSON is missing.
            ValueError: If no valid frame can be indexed.
        """

        if not self.source_path.is_file():
            raise FileNotFoundError(
                f"Detection label file not found: {self.source_path}"
            )
        if not rebuild and self._is_current():
            return len(self)

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(f"{self.cache_path}.tmp")
        if temporary.exists():
            temporary.unlink()

        count = 0
        try:
            with closing(sqlite3.connect(temporary)) as connection:
                connection.execute(
                    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                connection.execute(
                    """
                    CREATE TABLE frames (
                        stem TEXT PRIMARY KEY,
                        image_name TEXT NOT NULL,
                        sequence_id TEXT NOT NULL,
                        attributes_json TEXT NOT NULL,
                        detections_json TEXT NOT NULL,
                        segmentations_json TEXT NOT NULL
                    )
                    """
                )
                batch: list[tuple[str, str, str, str, str, str]] = []
                for raw_frame in iter_detection_json(self.source_path):
                    try:
                        normalized = normalize_detection_frame(raw_frame)
                    except ValueError:
                        continue
                    batch.append(_frame_to_json(normalized))
                    if len(batch) >= 1000:
                        connection.executemany(
                            "INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?, ?, ?)", batch
                        )
                        count += len(batch)
                        batch.clear()
                if batch:
                    connection.executemany(
                        "INSERT OR REPLACE INTO frames VALUES (?, ?, ?, ?, ?, ?)", batch
                    )
                    count += len(batch)
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    self._source_fingerprint().items(),
                )
                connection.commit()
            if count == 0:
                raise ValueError(
                    f"No valid detection frames were found in {self.source_path}"
                )
            os.replace(temporary, self.cache_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return count

    def __len__(self) -> int:
        """Return the number of cached frames.

        Returns:
            Integer row count.

        Raises:
            FileNotFoundError: If :meth:`ensure_built` has not been called.
        """

        if not self.cache_path.is_file():
            raise FileNotFoundError(f"Detection index not built: {self.cache_path}")
        with closing(sqlite3.connect(self.cache_path)) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0])

    def stems(self) -> set[str]:
        """Return every indexed image stem.

        Returns:
            Set of sample IDs available in the detection annotation file.
        """

        self.ensure_built()
        with closing(sqlite3.connect(self.cache_path)) as connection:
            return {str(row[0]) for row in connection.execute("SELECT stem FROM frames")}

    def get(self, stem: str) -> DetectionFrame | None:
        """Look up normalized detection data for one image stem.

        Args:
            stem: Filename without extension.

        Returns:
            A :class:`DetectionFrame`, or ``None`` if it is absent.
        """

        self.ensure_built()
        with closing(sqlite3.connect(self.cache_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM frames WHERE stem = ?", (stem,)
            ).fetchone()
        return None if row is None else _row_to_frame(row)
