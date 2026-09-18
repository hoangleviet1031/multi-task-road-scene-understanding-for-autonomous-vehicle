"""Training, evaluation, and checkpoint orchestration for single-task baselines."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn

from roadsense.losses.single_task import build_loss
from roadsense.metrics.detection import mean_average_precision
from roadsense.metrics.segmentation import confusion_matrix, lane_f1_with_tolerance
from roadsense.models.single_task import build_model, count_trainable_parameters
from roadsense.training.config import ExperimentConfig
from roadsense.training.data import build_dataloaders


def resolve_device(requested: str) -> torch.device:
    """Resolve an experiment device request against runtime availability.

    Args:
        requested: ``auto``, ``cpu``, or ``cuda``.

    Returns:
        Usable :class:`torch.device`.

    Raises:
        RuntimeError: If CUDA is explicitly requested but unavailable.
    """

    value = requested.lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    if value not in {"cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators.

    Args:
        seed: Reproducibility seed.

    Returns:
        ``None``.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(config: ExperimentConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Construct SGD or AdamW for trainable model parameters.

    Args:
        config: Experiment optimizer settings.
        model: Task model.

    Returns:
        Configured PyTorch optimizer.
    """

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    settings = config.optimization
    if settings.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=settings.learning_rate,
            momentum=settings.momentum,
            weight_decay=settings.weight_decay,
        )
    return torch.optim.AdamW(
        parameters,
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )


def _move_detection_targets(
    targets: Iterable[dict[str, torch.Tensor]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    """Move all tensors in detection target dictionaries to a device.

    Args:
        targets: Per-image target dictionaries.
        device: Destination device.

    Returns:
        New list of dictionaries containing moved tensors.
    """

    return [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    loss_module: nn.Module | None,
    device: torch.device,
    task: str,
    gradient_clip_norm: float | None = None,
    amp: bool = False,
) -> dict[str, float]:
    """Train one complete epoch for detection or segmentation.

    Args:
        model: Task model.
        loader: Training DataLoader.
        optimizer: PyTorch optimizer.
        loss_module: Drivable/lane loss, or ``None`` for detection.
        device: CPU or CUDA device.
        task: ``detection``, ``drivable``, or ``lane``.
        gradient_clip_norm: Optional maximum global gradient norm.
        amp: Enable CUDA automatic mixed precision.

    Returns:
        Mean total/component losses across batches.
    """

    model.train()
    amp_enabled = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    for images, targets in loader:
        optimizer.zero_grad(set_to_none=True)
        if task == "detection":
            image_list = [image.to(device) for image in images]
            target_list = _move_detection_targets(targets, device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                loss_values = model(image_list, target_list)
                if not isinstance(loss_values, dict):
                    raise RuntimeError("Detection model must return a loss dictionary in train mode.")
                total_loss = sum(loss_values.values())
                components = {"loss": total_loss, **loss_values}
        else:
            if loss_module is None:
                raise RuntimeError(f"{task} training requires a loss module.")
            image_batch = images.to(device)
            target_batch = targets.to(device)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(image_batch)
                components = loss_module(logits, target_batch)
                total_loss = components["loss"]

        if not bool(torch.isfinite(total_loss)):
            raise RuntimeError(f"Non-finite training loss encountered: {total_loss.item()}")
        scaler.scale(total_loss).backward()
        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        for name, value in components.items():
            totals[name] += float(value.detach().cpu())
        batches += 1
    if batches == 0:
        raise RuntimeError("Training DataLoader yielded no batches.")
    return {name: value / batches for name, value in totals.items()}


def _metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    """Convert an accumulated confusion matrix into segmentation metrics.

    Args:
        matrix: Square ground-truth-by-prediction count matrix.

    Returns:
        Per-class IoU/Dice, mean values, and pixel accuracy.
    """

    true_positive = np.diag(matrix).astype(np.float64)
    ground_truth = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    union = ground_truth + predicted - true_positive
    dice_denominator = ground_truth + predicted
    iou = [None if value == 0 else float(true_positive[i] / value) for i, value in enumerate(union)]
    dice = [
        None if value == 0 else float(2.0 * true_positive[i] / value)
        for i, value in enumerate(dice_denominator)
    ]
    valid_iou = [value for value in iou if value is not None]
    valid_dice = [value for value in dice if value is not None]
    total = float(matrix.sum())
    return {
        "confusion_matrix": matrix.astype(np.int64).tolist(),
        "per_class_iou": iou,
        "per_class_dice": dice,
        "mean_iou": float(np.mean(valid_iou)) if valid_iou else None,
        "mean_dice": float(np.mean(valid_dice)) if valid_dice else None,
        "pixel_accuracy": float(true_positive.sum() / total) if total else None,
    }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: Any,
    task: str,
    device: torch.device,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate one task using Milestone 1 metrics.

    Args:
        model: Trained task model.
        loader: Validation DataLoader.
        task: Detection, drivable, or lane.
        device: Evaluation device.
        settings: Task-specific evaluation YAML mapping.

    Returns:
        Detection mAP, drivable segmentation metrics, or mean lane metrics.
    """

    model.eval()
    if task == "detection":
        predictions: list[dict[str, Any]] = []
        targets_for_metric: list[dict[str, Any]] = []
        score_threshold = float(settings.get("score_threshold", 0.05))
        for images, targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, targets):
                scores = output["scores"].detach().cpu().numpy()
                keep = scores >= score_threshold
                predictions.append(
                    {
                        "boxes": output["boxes"].detach().cpu().numpy()[keep],
                        "labels": output["labels"].detach().cpu().numpy()[keep] - 1,
                        "scores": scores[keep],
                    }
                )
                targets_for_metric.append(
                    {
                        "boxes": target["boxes"].cpu().numpy(),
                        "labels": target["labels"].cpu().numpy() - 1,
                    }
                )
        return mean_average_precision(
            predictions,
            targets_for_metric,
            class_ids=range(10),
            iou_thresholds=settings.get("iou_thresholds", [0.5, 0.75]),
        )

    if task == "drivable":
        matrix = np.zeros((3, 3), dtype=np.int64)
        ignore_index = settings.get("ignore_index")
        for images, targets in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu().numpy()
            expected = targets.numpy()
            matrix += confusion_matrix(
                predictions,
                expected,
                num_classes=3,
                ignore_index=None if ignore_index is None else int(ignore_index),
            )
        return _metrics_from_confusion(matrix)

    threshold = float(settings.get("threshold", 0.5))
    tolerance = int(settings.get("tolerance", 2))
    totals: defaultdict[str, float] = defaultdict(float)
    samples = 0
    for images, targets in loader:
        logits = model(images.to(device))[:, 0]
        predictions = (torch.sigmoid(logits) >= threshold).cpu().numpy()
        expected = targets.numpy()
        for prediction, target in zip(predictions, expected):
            metrics = lane_f1_with_tolerance(prediction, target, tolerance)
            for key in ("precision", "recall", "f1"):
                totals[key] += float(metrics[key])
            samples += 1
    if samples == 0:
        raise RuntimeError("Validation DataLoader yielded no lane samples.")
    return {key: value / samples for key, value in totals.items()}


def _selection_metric(task: str, metrics: dict[str, Any]) -> float:
    """Extract the higher-is-better checkpoint selection metric.

    Args:
        task: Current task name.
        metrics: Evaluation result.

    Returns:
        Detection mAP, drivable mIoU, or lane F1.
    """

    key = {"detection": "map", "drivable": "mean_iou", "lane": "f1"}[task]
    value = metrics.get(key)
    return float("-inf") if value is None else float(value)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    epoch: int,
    best_metric: float,
    config: ExperimentConfig,
) -> Path:
    """Atomically save model and experiment state.

    Args:
        path: Destination ``.pt`` file.
        model: Model whose state is saved.
        optimizer: Optional optimizer state.
        scheduler: Optional scheduler state.
        epoch: Last completed zero-based epoch.
        best_metric: Best selection metric observed so far.
        config: Experiment configuration stored for traceability.

    Returns:
        Absolute checkpoint path.
    """

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": 1,
        "task": config.task,
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "config": config.raw,
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model and optional optimizer/scheduler state.

    Args:
        path: Existing checkpoint.
        model: Compatible model receiving weights.
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        map_location: Torch load destination.

    Returns:
        Checkpoint metadata excluding the large model/optimizer state mappings.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    payload = torch.load(source, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    return {
        "schema_version": payload.get("schema_version", 1),
        "task": payload.get("task"),
        "epoch": int(payload.get("epoch", -1)),
        "best_metric": float(payload.get("best_metric", float("-inf"))),
        "config": payload.get("config", {}),
    }


def _write_json(value: Any, path: Path) -> None:
    """Write JSON with parent creation and UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def run_training(
    config: ExperimentConfig, resume: str | Path | None = None
) -> dict[str, Any]:
    """Run a complete configured single-task experiment.

    Args:
        config: Validated experiment settings.
        resume: Optional checkpoint used to resume epoch/optimizer state.

    Returns:
        Summary with device, parameter count, best metric/checkpoint, and history.
    """

    seed_everything(config.seed)
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    loss_module = build_loss(config, device)
    optimizer = build_optimizer(config, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.optimization.epochs
    )
    start_epoch = 0
    best_metric = float("-inf")
    if resume is not None:
        metadata = load_checkpoint(resume, model, optimizer, scheduler, device)
        start_epoch = metadata["epoch"] + 1
        best_metric = metadata["best_metric"]

    history_path = config.output_dir / "history.jsonl"
    if resume is None:
        # A fresh experiment replaces stale history from an earlier run in the
        # same output directory. Checkpoints are overwritten atomically below.
        history_path.unlink(missing_ok=True)
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, config.optimization.epochs):
        epoch_learning_rate = float(optimizer.param_groups[0]["lr"])
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_module=loss_module,
            device=device,
            task=config.task,
            gradient_clip_norm=config.optimization.gradient_clip_norm,
            amp=config.optimization.amp,
        )
        validation_metrics = evaluate_model(
            model, val_loader, config.task, device, config.evaluation
        )
        scheduler.step()
        selected = _selection_metric(config.task, validation_metrics)
        record = {
            "epoch": epoch,
            "learning_rate": epoch_learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
            "selection_metric": selected,
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        save_checkpoint(
            config.output_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            max(best_metric, selected),
            config,
        )
        if selected > best_metric:
            best_metric = selected
            save_checkpoint(
                config.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                config,
            )
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "epochs": config.optimization.epochs,
                    "train_loss": train_metrics.get("loss"),
                    "selection_metric": selected,
                }
            ),
            flush=True,
        )

    summary = {
        "task": config.task,
        "device": str(device),
        "trainable_parameters": count_trainable_parameters(model),
        "train_samples": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "best_metric": best_metric,
        "best_checkpoint": str((config.output_dir / "best.pt").resolve()),
        "latest_checkpoint": str((config.output_dir / "latest.pt").resolve()),
        "history": history,
    }
    _write_json(summary, config.output_dir / "summary.json")
    return summary


def run_evaluation(
    config: ExperimentConfig, checkpoint: str | Path
) -> dict[str, Any]:
    """Evaluate a checkpoint using the configured development split.

    Args:
        config: Experiment settings controlling model, data, and metrics.
        checkpoint: Compatible checkpoint path.

    Returns:
        Metric dictionary, also written next to experiment artifacts.
    """

    seed_everything(config.seed)
    device = resolve_device(config.device)
    _, val_loader = build_dataloaders(config)
    model = build_model(config).to(device)
    metadata = load_checkpoint(checkpoint, model, map_location=device)
    metrics = evaluate_model(model, val_loader, config.task, device, config.evaluation)
    report = {
        "task": config.task,
        "device": str(device),
        "checkpoint": str(Path(checkpoint).expanduser().resolve()),
        "checkpoint_epoch": metadata["epoch"],
        "metrics": metrics,
    }
    _write_json(report, config.output_dir / "evaluation.json")
    return report
