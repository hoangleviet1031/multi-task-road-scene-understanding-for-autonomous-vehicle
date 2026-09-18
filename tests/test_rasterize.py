"""Tests for lazy conversion of legacy BDD100K polygons to masks."""

import numpy as np

from roadsense.data.detection_index import normalize_detection_frame
from roadsense.data.rasterize import rasterize_legacy_labels, sample_poly2d


def test_sample_poly2d_supports_cubic_bezier_commands() -> None:
    """Three consecutive C vertices produce a sampled cubic ending correctly."""

    points = sample_poly2d(
        {
            "vertices": [[0, 0], [0, 10], [10, 10], [10, 0]],
            "types": "LCCC",
            "closed": False,
        },
        curve_steps=4,
    )
    assert len(points) == 5
    assert points[0] == (0.0, 0.0)
    assert points[-1] == (10.0, 0.0)


def test_rasterize_legacy_drivable_and_lane_labels() -> None:
    """Polygon fill and lane stroke create normalized non-empty masks."""

    frame = normalize_detection_frame(
        {
            "name": "sample.jpg",
            "labels": [
                {
                    "category": "drivable area",
                    "attributes": {"areaType": "direct"},
                    "poly2d": [
                        {
                            "vertices": [[2, 2], [17, 2], [17, 17], [2, 17]],
                            "types": "LLLL",
                            "closed": True,
                        }
                    ],
                },
                {
                    "category": "lane",
                    "attributes": {"laneType": "single white"},
                    "poly2d": [
                        {
                            "vertices": [[5, 1], [5, 18]],
                            "types": "LL",
                            "closed": False,
                        }
                    ],
                },
            ],
        }
    )
    drivable, lane = rasterize_legacy_labels(frame, (20, 20), lane_width=2)
    assert drivable.shape == (20, 20)
    assert lane.shape == (20, 20)
    assert set(np.unique(drivable)) <= {0, 1, 2}
    assert set(np.unique(lane)) <= {0, 1}
    assert int((drivable == 0).sum()) > 0
    assert int(lane.sum()) > 0
