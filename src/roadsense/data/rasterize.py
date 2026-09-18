"""Rasterize legacy BDD100K poly2d labels into training masks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from roadsense.types import DetectionFrame


def _point(value: Any) -> tuple[float, float] | None:
    """Convert a JSON vertex to an ``(x, y)`` pair.

    Args:
        value: Candidate two-element array.

    Returns:
        Finite coordinate pair, or ``None`` for malformed input.
    """

    try:
        x, y = float(value[0]), float(value[1])
    except (IndexError, TypeError, ValueError):
        return None
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def sample_poly2d(poly: dict[str, Any], curve_steps: int = 24) -> list[tuple[float, float]]:
    """Convert Scalabel line/cubic path commands to a dense point sequence.

    Three consecutive ``C`` vertices are interpreted as cubic Bézier control 1,
    control 2, and endpoint, matching Matplotlib's ``CURVE4`` convention used by
    the official BDD100K converter.

    Args:
        poly: Mapping containing ``vertices`` and a parallel ``types`` string.
        curve_steps: Number of samples per cubic Bézier segment.

    Returns:
        Ordered pixel-space points suitable for Pillow polygon/line drawing.
    """

    raw_vertices = poly.get("vertices", [])
    if not isinstance(raw_vertices, list):
        return []
    vertices = [_point(value) for value in raw_vertices]
    if not vertices or vertices[0] is None:
        return []
    types = str(poly.get("types", "L" * len(vertices)))
    if len(types) < len(vertices):
        types += "L" * (len(vertices) - len(types))

    points = [vertices[0]]
    current = vertices[0]
    index = 1
    steps = max(2, int(curve_steps))
    while index < len(vertices):
        command = types[index].upper()
        if (
            command == "C"
            and index + 2 < len(vertices)
            and types[index + 1].upper() == "C"
            and types[index + 2].upper() == "C"
            and vertices[index] is not None
            and vertices[index + 1] is not None
            and vertices[index + 2] is not None
        ):
            control_1 = vertices[index]
            control_2 = vertices[index + 1]
            endpoint = vertices[index + 2]
            assert current is not None
            for step in range(1, steps + 1):
                t = step / steps
                inverse = 1.0 - t
                x = (
                    inverse**3 * current[0]
                    + 3.0 * inverse**2 * t * control_1[0]
                    + 3.0 * inverse * t**2 * control_2[0]
                    + t**3 * endpoint[0]
                )
                y = (
                    inverse**3 * current[1]
                    + 3.0 * inverse**2 * t * control_1[1]
                    + 3.0 * inverse * t**2 * control_2[1]
                    + t**3 * endpoint[1]
                )
                points.append((x, y))
            current = endpoint
            index += 3
            continue

        vertex = vertices[index]
        if vertex is not None:
            points.append(vertex)
            current = vertex
        index += 1
    return points


def _iter_polygons(annotation: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield valid poly2d dictionaries from one stored annotation.

    Args:
        annotation: Normalized segmentation annotation.

    Yields:
        Individual polygon/path mappings.
    """

    value = annotation.get("poly2d", [])
    if isinstance(value, list):
        for poly in value:
            if isinstance(poly, dict):
                yield poly


def rasterize_legacy_labels(
    frame: DetectionFrame,
    image_size: tuple[int, int],
    lane_width: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize legacy drivable and lane polygons for one BDD100K frame.

    Args:
        frame: Detection frame carrying normalized ``segmentations`` entries.
        image_size: ``(width, height)`` of the corresponding RGB image.
        lane_width: Positive foreground stroke width in pixels.

    Returns:
        Tuple ``(drivable_mask, lane_mask)``. Drivable values are 0 direct,
        1 alternative, and 2 background. Lane is binary with foreground one.

    Raises:
        ValueError: If image dimensions or lane width are not positive.
    """

    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height.")
    if lane_width <= 0:
        raise ValueError("lane_width must be positive.")

    drivable_image = Image.new("L", (width, height), color=2)
    lane_image = Image.new("L", (width, height), color=0)
    drivable_draw = ImageDraw.Draw(drivable_image)
    lane_draw = ImageDraw.Draw(lane_image)

    for annotation in frame.segmentations:
        category = str(annotation.get("category", ""))
        attributes = annotation.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}
        for poly in _iter_polygons(annotation):
            points = sample_poly2d(poly)
            if category == "drivable area" and len(points) >= 3:
                area_type = str(attributes.get("areaType", "direct"))
                value = 1 if area_type == "alternative" else 0
                drivable_draw.polygon(points, fill=value)
            elif category == "lane" and len(points) >= 2:
                lane_draw.line(points, fill=1, width=lane_width, joint="curve")

    return (
        np.asarray(drivable_image, dtype=np.uint8),
        np.asarray(lane_image, dtype=np.uint8),
    )


def load_or_rasterize_legacy_labels(
    frame: DetectionFrame,
    image_size: tuple[int, int],
    cache_root: str | Path,
    lane_width: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cached legacy masks or rasterize and atomically cache them.

    Args:
        frame: Normalized frame containing lane/drivable polygons.
        image_size: Source ``(width,height)``.
        cache_root: Writable derived-data root; raw dataset files are untouched.
        lane_width: Lane stroke width passed to rasterization.

    Returns:
        Normalized drivable and binary lane masks.
    """

    root = Path(cache_root).expanduser().resolve()
    drivable_path = root / "drivable" / f"{frame.stem}.png"
    lane_path = root / "lane" / f"{frame.stem}.png"
    expected_shape = (int(image_size[1]), int(image_size[0]))
    if drivable_path.is_file() and lane_path.is_file():
        with Image.open(drivable_path) as file:
            drivable = np.asarray(file, dtype=np.uint8)
        with Image.open(lane_path) as file:
            lane = np.asarray(file, dtype=np.uint8)
        if drivable.shape == expected_shape and lane.shape == expected_shape:
            return drivable, lane

    drivable, lane = rasterize_legacy_labels(frame, image_size, lane_width)
    drivable_path.parent.mkdir(parents=True, exist_ok=True)
    lane_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}-{uuid4().hex}"
    drivable_temp = drivable_path.with_name(f".{drivable_path.stem}-{token}.tmp.png")
    lane_temp = lane_path.with_name(f".{lane_path.stem}-{token}.tmp.png")
    try:
        Image.fromarray(drivable, mode="L").save(drivable_temp)
        Image.fromarray(lane, mode="L").save(lane_temp)
        os.replace(drivable_temp, drivable_path)
        os.replace(lane_temp, lane_path)
    finally:
        drivable_temp.unlink(missing_ok=True)
        lane_temp.unlink(missing_ok=True)
    return drivable, lane
