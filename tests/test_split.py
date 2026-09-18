"""Tests for deterministic, group-safe split construction."""

from roadsense.data.split import create_group_stratified_split


def test_group_split_has_no_sequence_leakage_and_is_deterministic() -> None:
    """Frames sharing a sequence remain together and the seed reproduces output."""

    records = []
    for sequence in range(6):
        for frame in range(2):
            records.append(
                {
                    "name": f"seq{sequence}_frame{frame}",
                    "sequence_id": f"seq{sequence}",
                    "timeofday": "daytime" if sequence < 3 else "night",
                    "weather": "clear",
                    "scene": "city street",
                }
            )
    first = create_group_stratified_split(records, dev_ratio=0.33, seed=7)
    second = create_group_stratified_split(records, dev_ratio=0.33, seed=7)
    assert first == second
    assert not set(first["train"]) & set(first["dev"])
    for sequence in range(6):
        members = {f"seq{sequence}_frame0", f"seq{sequence}_frame1"}
        assert members <= set(first["train"]) or members <= set(first["dev"])
