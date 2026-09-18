"""BDD100K class mappings and official mask decoders."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


DETECTION_CLASSES: tuple[str, ...] = (
    "pedestrian",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
)
DETECTION_CLASS_TO_ID = {name: index for index, name in enumerate(DETECTION_CLASSES)}

DRIVABLE_CLASS_NAMES: tuple[str, ...] = (
    "direct",
    "alternative",
    "background",
)


def ensure_single_channel(mask: NDArray[np.generic]) -> NDArray[np.uint8]:
    """Convert a mask array into a single-channel ``uint8`` array.

    Args:
        mask: A two-dimensional array or an ``H×W×1`` image.

    Returns:
        A contiguous ``uint8[H,W]`` array.

    Raises:
        ValueError: If the input contains multiple meaningful channels.
    """

    array = np.asarray(mask)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Expected a one-channel mask, got shape {array.shape}.")
    return np.ascontiguousarray(array, dtype=np.uint8)


def decode_drivable_mask(
    mask: NDArray[np.generic],
    direct_id: int = 0,
    alternative_id: int = 1,
    background_id: int = 2,
) -> NDArray[np.uint8]:
    """Normalize an official BDD100K drivable-area mask.

    Args:
        mask: One-channel integer mask using configured raw class IDs.
        direct_id: Raw ID for directly drivable pixels.
        alternative_id: Raw ID for alternatively drivable pixels.
        background_id: Raw ID for background pixels.

    Returns:
        ``uint8[H,W]`` with normalized values 0, 1, and 2.

    Raises:
        ValueError: If IDs overlap or the mask contains an unknown value.
    """

    raw_ids = (int(direct_id), int(alternative_id), int(background_id))
    if len(set(raw_ids)) != 3:
        raise ValueError("Drivable raw class IDs must be distinct.")
    array = ensure_single_channel(mask)
    known = np.isin(array, raw_ids)
    if not bool(np.all(known)):
        unknown = np.unique(array[~known]).tolist()
        raise ValueError(f"Unknown drivable mask values: {unknown}")
    output = np.empty_like(array, dtype=np.uint8)
    output[array == direct_id] = 0
    output[array == alternative_id] = 1
    output[array == background_id] = 2
    return output


def decode_lane_binary_mask(
    mask: NDArray[np.generic], background_bit: int = 3
) -> NDArray[np.uint8]:
    """Decode the bit-packed official BDD100K lane mask as binary foreground.

    In the official format, bits 0–2 store category, bit 3 marks background,
    bit 4 stores style, and bit 5 stores direction. This function intentionally
    collapses all lane categories, styles, and directions into one foreground.

    Args:
        mask: One-channel BDD100K lane-marking mask.
        background_bit: Position of the background flag; official value is 3.

    Returns:
        Binary ``uint8[H,W]`` containing one for lane pixels and zero otherwise.

    Raises:
        ValueError: If ``background_bit`` is outside the byte range.
    """

    if not 0 <= int(background_bit) <= 7:
        raise ValueError("background_bit must be between 0 and 7.")
    array = ensure_single_channel(mask)
    is_background = ((array >> int(background_bit)) & 1).astype(bool)
    return (~is_background).astype(np.uint8)


def safe_attributes(value: Any) -> dict[str, Any]:
    """Return a shallow string-keyed metadata dictionary.

    Args:
        value: Arbitrary JSON-derived value.

    Returns:
        A dictionary when ``value`` is a mapping, otherwise an empty dictionary.
    """

    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}
