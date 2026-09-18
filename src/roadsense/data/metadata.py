"""Efficient metadata extraction for split generation."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing

from roadsense.data.bdd100k import BDD100KDataset


def metadata_records_from_dataset(
    dataset: BDD100KDataset,
) -> list[dict[str, str]]:
    """Read split metadata using one SQLite connection.

    Args:
        dataset: Initialized BDD100K joint dataset.

    Returns:
        Records containing name, sequence ID, weather, scene, and time of day.
        When detection metadata is unavailable, each image becomes its own
        sequence and all stratification fields are ``undefined``.
    """

    if dataset.detection_index is None:
        return [
            {
                "name": name,
                "sequence_id": name,
                "weather": "undefined",
                "scene": "undefined",
                "timeofday": "undefined",
            }
            for name in dataset.names
        ]

    index = dataset.detection_index
    index.ensure_built()
    allowed = set(dataset.names)
    records: list[dict[str, str]] = []
    with closing(sqlite3.connect(index.cache_path)) as connection:
        cursor = connection.execute(
            "SELECT stem, sequence_id, attributes_json FROM frames ORDER BY stem"
        )
        for stem, sequence_id, attributes_json in cursor:
            name = str(stem)
            if name not in allowed:
                continue
            attributes = json.loads(attributes_json)
            records.append(
                {
                    "name": name,
                    "sequence_id": str(sequence_id) or name,
                    "weather": str(attributes.get("weather", "undefined")),
                    "scene": str(attributes.get("scene", "undefined")),
                    "timeofday": str(attributes.get("timeofday", "undefined")),
                }
            )
    return records
