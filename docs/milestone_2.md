# Milestone 2 — Independent single-task baselines

## Scope

Milestone 2 implements three independent reference systems. It does not share
parameters across tasks and contains no multi-task optimization; those belong to
Milestone 3.

| Task | Baseline | Output |
|---|---|---|
| Detection | Faster R-CNN MobileNetV3-FPN | boxes, BDD class IDs, confidence |
| Drivable area | ResNet18-FPN segmenter | 3-class logits `[B,3,H,W]` |
| Lane | ResNet18-FPN segmenter | binary logit `[B,1,H,W]` |

The detection baseline favors a low-resolution mobile backbone because the
current host is CPU-only. The segmentation baselines deliberately share the
same architecture so their later comparison with a shared multi-task encoder is
fair; only output channels and losses differ.

## Architecture and pipeline

```text
BDD100K image + Milestone 1 labels
              |
        synchronized flip
              |
   aspect-ratio-preserving letterbox
              |
   +----------+-----------+
   |          |           |
Faster     ResNet18     ResNet18
R-CNN      + FPN        + FPN
boxes      3 logits     1 logit
   |          |           |
RPN/ROI    CE + Dice   BCE + Dice
losses        |           |
mAP       mIoU/Dice   tolerance F1
```

One image transform updates image, boxes, and masks together, preventing label
misalignment. Letterboxing preserves road geometry instead of stretching a
1280×720 frame. Detection uses the mature Torchvision implementation so the
baseline is auditable and reproducible. The custom segmentation decoder merges
four encoder scales: fine features preserve lane boundaries while coarse
features provide road-scene context. GroupNorm is used in the decoder because
segmentation batch sizes are typically small.

The training runner seeds Python, NumPy, and PyTorch, validates all YAML fields,
selects CPU/CUDA, computes task-specific losses, clips gradients, evaluates each
epoch, and atomically writes `latest.pt` plus the best metric checkpoint. A
fresh run replaces stale JSONL history; `--resume` restores model, optimizer,
scheduler, epoch, and best score.

## Modules and reasons

| Module | Responsibility | Why it exists |
|---|---|---|
| `training/config.py` | Typed YAML parsing and validation | Fail early on invalid task, model, paths, or hyperparameters |
| `training/data.py` | Manifest subset, aligned transforms, DataLoaders | Reuse the audited Milestone 1 source of truth |
| `models/single_task.py` | Three independent model constructors | Establish a fair no-sharing baseline for later MTL comparison |
| `losses/single_task.py` | Imbalance-aware dense losses | Road/lane foreground is spatially imbalanced |
| `training/engine.py` | Train, evaluate, checkpoint, resume | Keep experiments deterministic and repeatable |
| `training/smoke.py` | Synthetic forward/backward checks | Detect framework/model breakage without the dataset |
| `training/cli.py` | `train`, `evaluate`, `smoke-test` commands | Make every experiment runnable from YAML |

## Algorithms and metrics

- Detection uses Faster R-CNN: an RPN proposes regions, ROI heads classify and
  regress them, and non-maximum suppression removes duplicates. The report uses
  per-class AP and mean AP over IoU 0.50:0.05:0.95 in the full profile. The
  transparent local evaluator should be cross-checked with the official
  BDD100K evaluator for a publication or leaderboard submission.
- Drivable segmentation uses weighted cross-entropy for pixel classification
  plus macro soft Dice to reduce majority-background dominance. Evaluation
  reports per-class IoU/Dice, mIoU, mean Dice, pixel accuracy, and the confusion
  matrix.
- Lane segmentation uses positive-weighted binary cross-entropy plus foreground
  Dice because lane pixels are thin and rare. Evaluation reports precision,
  recall, and F1 after a configurable symmetric pixel tolerance. The tolerance
  is measured in the resized/letterboxed evaluation resolution.

Smoke profiles contain only one or two samples and exist solely to verify the
software path. Their scores must never be reported as experimental findings.

## Public function contracts

### Configuration

- `load_experiment_config(path)`
  - Input: experiment YAML path.
  - Output: validated `ExperimentConfig` with absolute paths and typed nested
    settings.
- `resolve_device(requested)`
  - Input: `auto`, `cpu`, or `cuda`.
  - Output: a usable PyTorch device; explicit unavailable CUDA raises an error.

### Data

- `letterbox_sample(image, boxes, mask, output_size, mask_fill)`
  - Input: RGB array, optional boxes/mask, target `(height,width)`, padding ID.
  - Output: float image tensor, transformed boxes, transformed mask, and
    geometric metadata.
- `SingleTaskDataset(base, task, sample_names, image_size, training, flip_p)`
  - Input: Milestone 1 joint dataset plus task-specific selection settings.
  - Output: PyTorch dataset. Detection items are `(image,target_dict)`;
    segmentation items are `(image,target_mask)`.
- `build_dataloaders(config)`
  - Input: validated experiment configuration.
  - Output: training and validation DataLoaders using the manifest IDs.

### Models

- `build_detection_model(num_classes, pretrained_backbone, image_size, ...)`
  - Input: number of classes including background and model settings.
  - Output: Torchvision Faster R-CNN module.
- `ResNet18FPNSegmenter(num_classes, fpn_channels, pretrained_backbone)`
  - Input: output class count and architecture settings.
  - Output on forward: full-resolution logits `[B,C,H,W]`.
- `build_model(config)`
  - Input: experiment configuration.
  - Output: the model appropriate for its task.

### Losses

- `DrivableLoss.forward(logits, target)`
  - Input: `[B,3,H,W]` logits and `[B,H,W]` class IDs.
  - Output: dictionary containing total, cross-entropy, and Dice losses.
- `LaneLoss.forward(logits, target)`
  - Input: `[B,1,H,W]` logits and `[B,H,W]` binary target.
  - Output: dictionary containing total, BCE, and Dice losses.
- `build_loss(config, device)`
  - Input: experiment configuration and device.
  - Output: task loss module, or `None` because Torchvision detection provides
    its own training losses.

### Training and evaluation

- `train_one_epoch(...)`
  - Input: model, loader, optimizer, loss module, device, task, and clipping.
  - Output: averaged training loss dictionary.
- `evaluate_model(...)`
  - Input: model, validation loader, task, device, and evaluation settings.
  - Output: task metrics (`mAP`, `mIoU`, or lane F1).
- `save_checkpoint(...)`
  - Input: model/optimizer/scheduler state and metadata.
  - Output: absolute checkpoint path written atomically.
- `load_checkpoint(...)`
  - Input: checkpoint path and model, with optional optimizer/scheduler.
  - Output: restored metadata such as epoch and best metric.
- `run_training(config, resume)`
  - Input: experiment config and optional checkpoint.
  - Output: summary containing best metric/checkpoint and JSONL history.
- `run_evaluation(config, checkpoint)`
  - Input: experiment config and checkpoint.
  - Output: evaluation metric dictionary.
- `run_smoke_test(output_dir)`
  - Input: artifact directory.
  - Output: report proving forward/backward for all three baselines.

Every implementation function also has a docstring specifying inputs, outputs,
shapes, and raised errors.

## Installation

CPU host:

```powershell
.venv\Scripts\python.exe -m pip install -e ".[dev,train]"
```

On a CUDA host, install the PyTorch wheel selected for that machine from the
official PyTorch selector first, then install this project with `--no-deps` if
necessary to preserve that wheel.

## Verification

Framework-only smoke test:

```powershell
.venv\Scripts\roadsense-train.exe smoke-test `
  --output outputs\milestone2_smoke
```

Small real-data training run:

```powershell
.venv\Scripts\roadsense-train.exe train `
  --config configs\experiments\cpu_smoke_drivable.yaml
```

Equivalent one-epoch CPU profiles are available as
`cpu_smoke_lane.yaml` and `cpu_smoke_detection.yaml`. They use one or two
samples only and verify plumbing; their metrics are not scientifically useful.

Full baselines:

```powershell
.venv\Scripts\roadsense-train.exe train --config configs\experiments\detection_baseline.yaml
.venv\Scripts\roadsense-train.exe train --config configs\experiments\drivable_baseline.yaml
.venv\Scripts\roadsense-train.exe train --config configs\experiments\lane_baseline.yaml
```

Evaluation:

```powershell
.venv\Scripts\roadsense-train.exe evaluate `
  --config configs\experiments\drivable_baseline.yaml `
  --checkpoint outputs\experiments\drivable_baseline\best.pt
```
