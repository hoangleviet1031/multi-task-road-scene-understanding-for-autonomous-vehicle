"""Tests for automatic Kaggle discovery and resolved run configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from roadsense.kaggle import (
    create_dataset_config,
    create_experiment_config,
    discover_bdd100k,
    main,
)


def _fake_bdd100k(root: Path, dataset_name: str = "bdd") -> None:
    image_root = root / dataset_name / "bdd100k" / "images" / "100k"
    (image_root / "train").mkdir(parents=True)
    (image_root / "val").mkdir(parents=True)
    (image_root / "train" / "train.jpg").write_bytes(b"image")
    (image_root / "val" / "val.jpg").write_bytes(b"image")
    label_root = root / dataset_name / "bdd100k" / "labels"
    label_root.mkdir(parents=True)
    (label_root / "bdd100k_labels_images_train.json").write_text("[]", encoding="utf-8")
    (label_root / "bdd100k_labels_images_val.json").write_text("[]", encoding="utf-8")


def test_discover_bdd100k_supports_nested_kaggle_datasets(tmp_path: Path) -> None:
    _fake_bdd100k(tmp_path)
    discovered = discover_bdd100k(tmp_path)
    assert discovered.train_images.name == "train"
    assert discovered.val_images.name == "val"
    assert discovered.train_labels.name == "bdd100k_labels_images_train.json"
    assert discovered.val_labels.name == "bdd100k_labels_images_val.json"


def test_discover_bdd100k_rejects_ambiguous_image_datasets(tmp_path: Path) -> None:
    _fake_bdd100k(tmp_path, "first")
    _fake_bdd100k(tmp_path, "second")
    with pytest.raises(ValueError, match="multiple BDD100K images"):
        discover_bdd100k(tmp_path)


def test_pilot_configs_are_absolute_limited_and_amp_enabled(tmp_path: Path) -> None:
    _fake_bdd100k(tmp_path / "input")
    discovered = discover_bdd100k(tmp_path / "input")
    output = tmp_path / "working" / "outputs" / "lane_pilot"
    dataset_path = create_dataset_config(
        discovered,
        cache_dir=tmp_path / "working" / "cache",
        destination=output / "resolved_dataset.yaml",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"train": ["a"], "dev": ["b"]}', encoding="utf-8")
    experiment_path = create_experiment_config(
        base_config=Path("configs/experiments/lane_baseline.yaml"),
        dataset_config=dataset_path,
        split_manifest=manifest,
        output_dir=output,
        destination=output / "resolved_experiment.yaml",
        mode="pilot",
        device="cuda",
        amp=True,
    )
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    assert Path(dataset["splits"]["train"]["images"]).is_absolute()
    assert dataset["cache_dir"] == str((tmp_path / "working" / "cache").resolve())
    assert experiment["data"]["max_train_samples"] == 256
    assert experiment["data"]["max_val_samples"] == 64
    assert experiment["optimization"]["epochs"] == 1
    assert experiment["optimization"]["amp"] is True
    assert experiment["device"] == "cuda"


def test_full_mode_requires_explicit_confirmation() -> None:
    with pytest.raises(SystemExit, match="requires --confirm-full"):
        main(["--task", "lane", "--mode", "full"])
