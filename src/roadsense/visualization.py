"""Pillow-based visual checks for joint road-scene annotations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from roadsense.data.labels import DETECTION_CLASSES
from roadsense.types import RoadSceneSample


BOX_COLORS = (
    "#ff4d6d",
    "#ff8c42",
    "#00b4d8",
    "#4361ee",
    "#8338ec",
    "#9d4edd",
    "#fb5607",
    "#ffbe0b",
    "#06d6a0",
    "#2ec4b6",
)


def _blend_mask(
    image: np.ndarray,
    mask: np.ndarray,
    value: int,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """Blend one mask value into an RGB array in place.

    Args:
        image: Mutable ``uint8[H,W,3]`` image.
        mask: Equal-sized integer mask.
        value: Mask value to color.
        color: RGB overlay color.
        alpha: Overlay opacity between zero and one.

    Returns:
        ``None``; ``image`` is modified in place.
    """

    selected = mask == value
    if not bool(np.any(selected)):
        return
    base = image[selected].astype(np.float32)
    overlay = np.asarray(color, dtype=np.float32)
    image[selected] = np.clip((1.0 - alpha) * base + alpha * overlay, 0, 255).astype(
        np.uint8
    )


def render_sample(sample: RoadSceneSample) -> Image.Image:
    """Render boxes and segmentation masks on one road-scene sample.

    Args:
        sample: Canonical RoadSense sample.

    Returns:
        New Pillow RGB image with annotation overlays and a metadata banner.

    Raises:
        ValueError: If a mask shape does not match the source image.
    """

    canvas = np.asarray(sample.image, dtype=np.uint8).copy()
    height, width = canvas.shape[:2]
    if sample.drivable_mask is not None:
        if sample.drivable_mask.shape != (height, width):
            raise ValueError("Drivable mask shape does not match the image.")
        _blend_mask(canvas, sample.drivable_mask, 0, (0, 210, 120), 0.35)
        _blend_mask(canvas, sample.drivable_mask, 1, (255, 190, 0), 0.35)
    if sample.lane_mask is not None:
        if sample.lane_mask.shape != (height, width):
            raise ValueError("Lane mask shape does not match the image.")
        _blend_mask(canvas, sample.lane_mask, 1, (255, 0, 220), 0.80)

    rendered = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(rendered)
    for box, label in zip(sample.boxes, sample.labels):
        label_id = int(label)
        color = BOX_COLORS[label_id % len(BOX_COLORS)]
        x1, y1, x2, y2 = (float(value) for value in box)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        class_name = (
            DETECTION_CLASSES[label_id]
            if 0 <= label_id < len(DETECTION_CLASSES)
            else f"class:{label_id}"
        )
        text_box = draw.textbbox((x1, y1), class_name)
        draw.rectangle(text_box, fill=color)
        draw.text((x1, y1), class_name, fill="black")

    banner = (
        f"{sample.name} | weather={sample.attributes.get('weather', 'undefined')} | "
        f"scene={sample.attributes.get('scene', 'undefined')} | "
        f"time={sample.attributes.get('timeofday', 'undefined')}"
    )
    text_box = draw.textbbox((4, 4), banner)
    draw.rectangle(text_box, fill=(0, 0, 0))
    draw.text((4, 4), banner, fill=(255, 255, 255))
    return rendered


def save_sample_visualization(
    sample: RoadSceneSample, output_path: str | Path
) -> Path:
    """Render and save a sample visualization.

    Args:
        sample: Canonical RoadSense sample.
        output_path: Destination image file.

    Returns:
        Absolute path of the written image.
    """

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    render_sample(sample).save(destination)
    return destination
