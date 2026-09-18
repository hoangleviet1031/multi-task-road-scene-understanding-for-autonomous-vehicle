"""Unit tests for official BDD100K label decoding."""

import numpy as np
import pytest

from roadsense.data.detection_index import normalize_detection_frame
from roadsense.data.labels import decode_drivable_mask, decode_lane_binary_mask


def test_decode_drivable_mask_preserves_official_ids() -> None:
    """Official direct, alternative, and background IDs map to 0, 1, and 2."""

    raw = np.array([[0, 1, 2]], dtype=np.uint8)
    assert np.array_equal(decode_drivable_mask(raw), raw)


def test_decode_drivable_rejects_unknown_values() -> None:
    """Unknown mask values must be reported instead of silently remapped."""

    with pytest.raises(ValueError, match="Unknown drivable"):
        decode_drivable_mask(np.array([[0, 7]], dtype=np.uint8))


def test_decode_lane_uses_background_bit() -> None:
    """Foreground values have bit 3 clear while 255 is background."""

    raw = np.array([[6, 255, 0, 8]], dtype=np.uint8)
    expected = np.array([[1, 0, 1, 0]], dtype=np.uint8)
    assert np.array_equal(decode_lane_binary_mask(raw), expected)


def test_normalize_detection_frame() -> None:
    """A current-format frame is converted to zero-based class IDs."""

    frame = normalize_detection_frame(
        {
            "name": "abc.jpg",
            "videoName": "video-a",
            "attributes": {"weather": "clear"},
            "labels": [
                {
                    "category": "car",
                    "box2d": {"x1": 1, "y1": 2, "x2": 10, "y2": 20},
                },
                {"category": "lane", "box2d": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
            ],
        }
    )
    assert frame.stem == "abc"
    assert frame.sequence_id == "video-a"
    assert frame.attributes["weather"] == "clear"
    assert len(frame.detections) == 1
    assert frame.detections[0].category == "car"
    assert frame.detections[0].label == 2
