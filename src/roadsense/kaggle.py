"""Reliable Kaggle bootstrap and runner for Milestone 2 baselines.

The module deliberately generates resolved runtime configuration inside
``/kaggle/working`` instead of modifying checked-in experiment files.  It also
records enough provenance to associate downloaded checkpoints with a Git
commit, dataset layout, environment, and exact YAML configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml


TASKS = ("detection", "drivable", "lane")
BASE_EXPERIMENTS = {
    "detection": "configs/experiments/detection_baseline.yaml",
    "drivable": "configs/experiments/drivable_baseline.yaml",
    "lane": "configs/experiments/lane_baseline.yaml",
}
LABEL_FILENAMES = {
    "train": "bdd100k_labels_images_train.json",
    "val": "bdd100k_labels_images_val.json",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False, allow_unicode=True)
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class DiscoveredBDD100K:
    """Absolute paths needed by the existing BDD100K data adapter."""

    train_images: Path
    val_images: Path
    train_labels: Path
    val_labels: Path

    def serializable(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def _validate_image_directory(path: Path, name: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {result}")
    if next(result.glob("*.jpg"), None) is None and next(result.glob("*.png"), None) is None:
        raise ValueError(f"{name} contains no .jpg or .png images: {result}")
    return result


def _validate_label_file(path: Path, name: str) -> Path:
    result = path.expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{name} file not found: {result}")
    if result.stat().st_size == 0:
        raise ValueError(f"{name} file is empty: {result}")
    return result


def _bounded_candidates(input_root: Path, max_depth: int = 8) -> tuple[list[Path], dict[str, list[Path]]]:
    """Find standard image roots and exact label files without entering image folders."""

    image_roots: list[Path] = []
    labels = {"train": [], "val": []}
    root_depth = len(input_root.parts)
    for current_text, directories, files in os.walk(input_root):
        current = Path(current_text)
        depth = len(current.parts) - root_depth
        directories[:] = [
            name
            for name in directories
            if not name.startswith(".") and name not in {"__pycache__", ".git"}
        ]
        if current.name == "100k" and current.parent.name == "images":
            if "train" in directories and "val" in directories:
                image_roots.append(current.resolve())
            # Never enumerate the tens of thousands of image files.
            directories[:] = [name for name in directories if name not in {"train", "val"}]
        file_set = set(files)
        for split, filename in LABEL_FILENAMES.items():
            if filename in file_set:
                labels[split].append((current / filename).resolve())
        if depth >= max_depth:
            directories.clear()
    return sorted(set(image_roots)), {
        split: sorted(set(paths)) for split, paths in labels.items()
    }


def _unique_candidate(candidates: list[Path], description: str) -> Path:
    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-discover {description}. Attach the BDD100K Kaggle "
            "dataset or pass an explicit path override."
        )
    if len(candidates) > 1:
        formatted = "\n  - ".join(str(path) for path in candidates)
        raise ValueError(
            f"Auto-discovery found multiple {description} candidates. Pass an "
            f"explicit override to avoid selecting the wrong dataset:\n  - {formatted}"
        )
    return candidates[0]


def discover_bdd100k(
    input_root: str | Path,
    train_images: str | Path | None = None,
    val_images: str | Path | None = None,
    train_labels: str | Path | None = None,
    val_labels: str | Path | None = None,
) -> DiscoveredBDD100K:
    """Discover a BDD100K layout under Kaggle input with safe ambiguity errors."""

    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle input root not found: {root}")
    image_roots, label_candidates = _bounded_candidates(root)
    chosen_image_root: Path | None = None
    if train_images is None or val_images is None:
        chosen_image_root = _unique_candidate(image_roots, "BDD100K images/100k root")
    resolved_train_images = _validate_image_directory(
        Path(train_images) if train_images is not None else chosen_image_root / "train",
        "train images",
    )
    resolved_val_images = _validate_image_directory(
        Path(val_images) if val_images is not None else chosen_image_root / "val",
        "validation images",
    )
    resolved_train_labels = _validate_label_file(
        Path(train_labels)
        if train_labels is not None
        else _unique_candidate(label_candidates["train"], "training label JSON"),
        "training labels",
    )
    resolved_val_labels = _validate_label_file(
        Path(val_labels)
        if val_labels is not None
        else _unique_candidate(label_candidates["val"], "validation label JSON"),
        "validation labels",
    )
    return DiscoveredBDD100K(
        train_images=resolved_train_images,
        val_images=resolved_val_images,
        train_labels=resolved_train_labels,
        val_labels=resolved_val_labels,
    )


def create_dataset_config(
    discovered: DiscoveredBDD100K, cache_dir: Path, destination: Path
) -> Path:
    """Write an absolute-path dataset config suitable for read-only Kaggle input."""

    config = {
        "root": str(discovered.train_images.anchor or "/"),
        "cache_dir": str(cache_dir.resolve()),
        "splits": {
            "train": {
                "images": str(discovered.train_images),
                "detection_labels": str(discovered.train_labels),
                "drivable_masks": None,
                "lane_masks": None,
            },
            "val": {
                "images": str(discovered.val_images),
                "detection_labels": str(discovered.val_labels),
                "drivable_masks": None,
                "lane_masks": None,
            },
        },
        "required_tasks": ["detection", "drivable", "lane"],
        "drivable": {"direct_id": 0, "alternative_id": 1, "background_id": 2},
        "lane": {"background_bit": 3},
    }
    _write_yaml(destination, config)
    return destination.resolve()


def create_experiment_config(
    base_config: Path,
    dataset_config: Path,
    split_manifest: Path,
    output_dir: Path,
    destination: Path,
    mode: str,
    device: str,
    amp: bool,
    train_limit: int | None = None,
    val_limit: int | None = None,
    epochs: int | None = None,
    workers: int = 0,
    batch_size: int | None = None,
    pretrained: bool = True,
) -> Path:
    """Resolve a checked-in baseline into a self-contained Kaggle run config."""

    with base_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be 'pilot' or 'full'.")
    if mode == "pilot":
        train_limit = 256 if train_limit is None else train_limit
        val_limit = 64 if val_limit is None else val_limit
        epochs = 1 if epochs is None else epochs
    else:
        train_limit = train_limit
        val_limit = val_limit
    config["device"] = device
    config["output_dir"] = str(output_dir.resolve())
    config["data"]["dataset_config"] = str(dataset_config.resolve())
    config["data"]["manifest"] = str(split_manifest.resolve())
    config["data"]["num_workers"] = int(workers)
    config["data"]["max_train_samples"] = train_limit
    config["data"]["max_val_samples"] = val_limit
    if batch_size is not None:
        config["data"]["batch_size"] = int(batch_size)
    if epochs is not None:
        config["optimization"]["epochs"] = int(epochs)
    config["optimization"]["amp"] = bool(amp)
    config["model"]["pretrained_backbone"] = bool(pretrained)
    _write_yaml(destination, config)
    return destination.resolve()


def _git_value(repo_root: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _environment() -> dict[str, Any]:
    import torch
    import torchvision

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ],
        "kaggle_kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_inventory(output_dir: Path, hash_checkpoints: bool) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in (
        "best.pt",
        "latest.pt",
        "summary.json",
        "history.jsonl",
        "evaluation.json",
        "resolved_dataset.yaml",
        "resolved_experiment.yaml",
        "split_manifest.json",
        "environment.txt",
    ):
        path = output_dir / name
        if not path.is_file():
            continue
        details: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
        if hash_checkpoints and path.suffix == ".pt":
            details["sha256"] = _sha256(path)
        result[name] = details
    return result


def _stage_attached_weights(input_root: Path, torch_home: Path) -> dict[str, str]:
    """Copy exact torchvision weight files from an attached Kaggle dataset if found."""

    from torchvision.models import MobileNet_V3_Large_Weights, ResNet18_Weights

    expected = {
        Path(ResNet18_Weights.DEFAULT.url).name,
        Path(MobileNet_V3_Large_Weights.IMAGENET1K_V1.url).name,
    }
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    found: dict[str, str] = {}
    for current_text, directories, files in os.walk(input_root):
        current = Path(current_text)
        directories[:] = [
            name
            for name in directories
            if name not in {"train", "val", ".git", "__pycache__"}
        ]
        for filename in expected & set(files):
            source = current / filename
            destination = checkpoint_dir / filename
            if not destination.exists():
                shutil.copy2(source, destination)
            found[filename] = str(source)
        if len(found) == len(expected):
            break
    return found


def _preflight(experiment_path: Path) -> dict[str, Any]:
    """Validate config, one real sample, and pretrained model construction."""

    from roadsense.models.single_task import build_model
    from roadsense.training.config import load_experiment_config
    from roadsense.training.data import build_dataloaders

    config = load_experiment_config(experiment_path)
    train_loader, val_loader = build_dataloaders(config)
    image, target = train_loader.dataset[0]
    sample_shape = list(image.shape)
    if config.task == "detection":
        target_summary = {"boxes": int(target["boxes"].shape[0])}
    else:
        target_summary = {"mask_shape": list(target.shape)}
    model = build_model(config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model
    return {
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "sample_shape": sample_shape,
        "target": target_summary,
        "parameters": parameter_count,
    }


def _checkpoint_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload.get("epoch", -1))


def _core_versions(output_dir: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=False,
    )
    (output_dir / "environment.txt").write_text(result.stdout, encoding="utf-8")


def _run_task(
    task: str,
    args: argparse.Namespace,
    discovered: DiscoveredBDD100K,
    environment: dict[str, Any],
) -> int:
    repo_root = Path(args.repo_root).expanduser().resolve()
    working_root = Path(args.working_root).expanduser().resolve()
    output_dir = working_root / "outputs" / f"{task}_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_config = create_dataset_config(
        discovered,
        cache_dir=working_root / "cache" / "bdd100k",
        destination=output_dir / "resolved_dataset.yaml",
    )
    source_manifest = repo_root / "data" / "splits" / "bdd100k_train_dev.json"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Split manifest not found: {source_manifest}")
    run_manifest = output_dir / "split_manifest.json"
    shutil.copy2(source_manifest, run_manifest)
    device = "cuda" if environment["cuda_available"] else "cpu"
    experiment_path = create_experiment_config(
        base_config=repo_root / BASE_EXPERIMENTS[task],
        dataset_config=dataset_config,
        split_manifest=run_manifest,
        output_dir=output_dir,
        destination=output_dir / "resolved_experiment.yaml",
        mode=args.mode,
        device=device,
        amp=bool(device == "cuda" and not args.no_amp),
        train_limit=args.train_limit,
        val_limit=args.val_limit,
        epochs=args.epochs,
        workers=args.workers,
        batch_size=args.batch_size,
        pretrained=not args.no_pretrained,
    )
    _core_versions(output_dir)
    provenance: dict[str, Any] = {
        "schema_version": 1,
        "status": "preparing",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "task": task,
        "mode": args.mode,
        "git": {
            "commit": _git_value(repo_root, "rev-parse", "HEAD"),
            "branch": _git_value(repo_root, "branch", "--show-current"),
            "dirty": bool(_git_value(repo_root, "status", "--porcelain")),
        },
        "environment": environment,
        "dataset": discovered.serializable(),
        "paths": {
            "repo_root": str(repo_root),
            "working_root": str(working_root),
            "output_dir": str(output_dir),
            "experiment_config": str(experiment_path),
        },
        "arguments": vars(args),
    }
    manifest_path = output_dir / "run_manifest.json"
    _atomic_json(manifest_path, provenance)
    try:
        provenance["preflight"] = _preflight(experiment_path)
        provenance["status"] = "prepared"
        provenance["updated_at"] = _utc_now()
        _atomic_json(manifest_path, provenance)
        print(json.dumps({"task": task, "preflight": provenance["preflight"]}, indent=2))
        if args.prepare_only:
            return 0

        from roadsense.training.config import load_experiment_config

        config = load_experiment_config(experiment_path)
        latest = output_dir / "latest.pt"
        explicit_resume = None if args.resume is None else Path(args.resume).expanduser().resolve()
        if explicit_resume is not None and not explicit_resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {explicit_resume}")
        resume = explicit_resume or (latest if latest.is_file() else None)
        completed_epoch = _checkpoint_epoch(latest)
        training_complete = (
            completed_epoch is not None
            and completed_epoch + 1 >= config.optimization.epochs
        )
        command = [
            sys.executable,
            "-m",
            "roadsense.training.cli",
            "train",
            "--config",
            str(experiment_path),
        ]
        if resume is not None and not training_complete:
            command.extend(["--resume", str(resume)])
        provenance["resume_from"] = None if resume is None else str(resume)
        provenance["train_command"] = command
        provenance["status"] = "running" if not training_complete else "already_complete"
        provenance["updated_at"] = _utc_now()
        _atomic_json(manifest_path, provenance)
        if not training_complete:
            subprocess.run(command, cwd=repo_root, check=True)

        best = output_dir / "best.pt"
        selected_checkpoint = best if best.is_file() else latest
        if not selected_checkpoint.is_file() and resume is not None:
            selected_checkpoint = resume
        if not selected_checkpoint.is_file():
            raise RuntimeError("Training finished without a usable checkpoint.")
        evaluation_command = [
            sys.executable,
            "-m",
            "roadsense.training.cli",
            "evaluate",
            "--config",
            str(experiment_path),
            "--checkpoint",
            str(selected_checkpoint),
        ]
        provenance["evaluation_command"] = evaluation_command
        subprocess.run(evaluation_command, cwd=repo_root, check=True)
        provenance["status"] = "succeeded"
        provenance["completed_at"] = _utc_now()
        provenance["artifacts"] = _artifact_inventory(
            output_dir, hash_checkpoints=args.hash_checkpoints
        )
        provenance["updated_at"] = _utc_now()
        _atomic_json(manifest_path, provenance)
        return 0
    except Exception as exc:
        provenance["status"] = "failed"
        provenance["error"] = f"{type(exc).__name__}: {exc}"
        provenance["updated_at"] = _utc_now()
        provenance["artifacts"] = _artifact_inventory(
            output_dir, hash_checkpoints=False
        )
        _atomic_json(manifest_path, provenance)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m roadsense.kaggle",
        description="Auto-discover BDD100K and run reproducible Milestone 2 Kaggle training.",
    )
    parser.add_argument("--task", choices=(*TASKS, "all"), required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument("--input-root", default=os.environ.get("KAGGLE_INPUT_PATH", "/kaggle/input"))
    parser.add_argument("--working-root", default="/kaggle/working/roadsense")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--train-images")
    parser.add_argument("--val-images")
    parser.add_argument("--train-labels")
    parser.add_argument("--val-labels")
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--hash-checkpoints", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "full" and not args.confirm_full:
        raise SystemExit("Full training requires --confirm-full to prevent accidental long runs.")
    if args.task == "all" and args.resume is not None:
        raise SystemExit("--resume can only be used with one explicit task.")
    for name in ("train_limit", "val_limit", "epochs", "batch_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    if args.workers < 0:
        raise SystemExit("--workers cannot be negative.")
    environment = _environment()
    if not environment["cuda_available"] and not args.allow_cpu:
        raise SystemExit(
            "CUDA is unavailable. Enable a Kaggle GPU accelerator, or pass "
            "--allow-cpu only for a small pilot."
        )
    input_root = Path(args.input_root).expanduser().resolve()
    working_root = Path(args.working_root).expanduser().resolve()
    torch_home = working_root / "torch_cache"
    os.environ["TORCH_HOME"] = str(torch_home)
    staged_weights = _stage_attached_weights(input_root, torch_home)
    if staged_weights:
        print(json.dumps({"staged_pretrained_weights": staged_weights}, indent=2))
    discovered = discover_bdd100k(
        input_root=input_root,
        train_images=args.train_images,
        val_images=args.val_images,
        train_labels=args.train_labels,
        val_labels=args.val_labels,
    )
    print(json.dumps({"discovered_bdd100k": discovered.serializable()}, indent=2))
    tasks = TASKS if args.task == "all" else (args.task,)
    for task in tasks:
        _run_task(task, args, discovered, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
