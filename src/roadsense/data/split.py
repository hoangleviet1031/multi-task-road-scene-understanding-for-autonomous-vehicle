"""Deterministic group-aware stratified splitting."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Iterable


STRATIFY_FIELDS = ("timeofday", "weather", "scene")


def _stratum(record: dict[str, str]) -> tuple[str, str, str]:
    """Extract a normalized stratification key from one metadata record.

    Args:
        record: Sample metadata mapping.

    Returns:
        Tuple of time of day, weather, and scene.
    """

    return tuple(str(record.get(field, "undefined")) for field in STRATIFY_FIELDS)  # type: ignore[return-value]


def create_group_stratified_split(
    records: Iterable[dict[str, str]],
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, object]:
    """Split samples while keeping every sequence entirely on one side.

    Each sequence is assigned the majority metadata stratum among its frames.
    Groups are then shuffled deterministically within each stratum. Singleton
    strata remain in training because their distribution cannot be preserved in
    both subsets.

    Args:
        records: Mappings containing at least ``name`` and preferably
            ``sequence_id``, ``timeofday``, ``weather``, and ``scene``.
        dev_ratio: Target development fraction strictly between zero and one.
        seed: Reproducibility seed.

    Returns:
        Manifest with train/dev sample IDs, counts, and per-stratum summaries.

    Raises:
        ValueError: If records are malformed, duplicated, or too few to split.
    """

    if not 0.0 < dev_ratio < 1.0:
        raise ValueError("dev_ratio must be strictly between zero and one.")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_names: set[str] = set()
    for raw_record in records:
        record = {str(key): str(value) for key, value in raw_record.items()}
        name = record.get("name", "").strip()
        if not name:
            raise ValueError("Every split record requires a non-empty 'name'.")
        if name in seen_names:
            raise ValueError(f"Duplicate sample name in split records: {name}")
        seen_names.add(name)
        sequence_id = record.get("sequence_id", name).strip() or name
        grouped[sequence_id].append(record)

    if len(grouped) < 2:
        raise ValueError("At least two independent sequence groups are required.")

    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for sequence_id, group_records in grouped.items():
        counts = Counter(_stratum(record) for record in group_records)
        majority = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        strata[majority].append(sequence_id)

    generator = random.Random(seed)
    dev_groups: set[str] = set()
    stratum_summary: dict[str, dict[str, int]] = {}
    for key in sorted(strata):
        sequence_ids = sorted(strata[key])
        generator.shuffle(sequence_ids)
        if len(sequence_ids) == 1:
            dev_count = 0
        else:
            dev_count = max(1, round(len(sequence_ids) * dev_ratio))
            dev_count = min(dev_count, len(sequence_ids) - 1)
        dev_groups.update(sequence_ids[:dev_count])
        label = "|".join(key)
        stratum_summary[label] = {
            "groups": len(sequence_ids),
            "train_groups": len(sequence_ids) - dev_count,
            "dev_groups": dev_count,
        }

    if not dev_groups:
        largest_group = max(grouped, key=lambda item: (len(grouped[item]), item))
        dev_groups.add(largest_group)
    if len(dev_groups) == len(grouped):
        dev_groups.remove(sorted(dev_groups)[-1])

    # Recompute summaries after the global fallback above so singleton strata
    # report the actual assignment rather than the initial per-stratum proposal.
    stratum_summary = {}
    for key in sorted(strata):
        sequence_ids = strata[key]
        actual_dev_count = sum(
            sequence_id in dev_groups for sequence_id in sequence_ids
        )
        label = "|".join(key)
        stratum_summary[label] = {
            "groups": len(sequence_ids),
            "train_groups": len(sequence_ids) - actual_dev_count,
            "dev_groups": actual_dev_count,
        }

    train_names: list[str] = []
    dev_names: list[str] = []
    train_groups: set[str] = set()
    for sequence_id, group_records in grouped.items():
        destination = dev_names if sequence_id in dev_groups else train_names
        destination.extend(record["name"] for record in group_records)
        if sequence_id not in dev_groups:
            train_groups.add(sequence_id)

    train_names.sort()
    dev_names.sort()
    overlap = set(train_names) & set(dev_names)
    group_overlap = train_groups & dev_groups
    if overlap or group_overlap:
        raise RuntimeError("Internal split error produced sample or group leakage.")

    return {
        "schema_version": 1,
        "seed": int(seed),
        "requested_dev_ratio": float(dev_ratio),
        "actual_dev_ratio": len(dev_names) / (len(train_names) + len(dev_names)),
        "train": train_names,
        "dev": dev_names,
        "train_count": len(train_names),
        "dev_count": len(dev_names),
        "train_group_count": len(train_groups),
        "dev_group_count": len(dev_groups),
        "strata": stratum_summary,
    }
