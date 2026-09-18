# RoadSense-MTL

Conflict-Aware Multi-Task Road Scene Understanding for Object Detection,
Drivable-Area Segmentation and Lane Detection.

Tài liệu:

- [Documentation index](docs/README.md).
- [Tổng quan chi tiết](docs/project_overview_vi.md).
- [Thiết kế nghiên cứu conflict-aware](docs/conflict_aware_research_design_vi.md).

RoadSense-MTL studies negative transfer in joint object detection, drivable-area
segmentation, and lane detection. The repository currently implements the data
foundation (Milestone 1) and reproducible independent baselines (Milestone 2).
The multi-task model planned for Milestone 3 is intentionally not implemented
until Milestone 2 is accepted.

Current status: Milestones 1–2 are implemented and tested. The task-specific
adapter and gradient-conflict design for Milestone 3 is documented but has not
been implemented. All current model profiles use a 640×384 letterbox canvas
(`width × height`; YAML uses `[384, 640]`).

## Implemented scope

- Configurable BDD100K path handling.
- Streaming conversion of the large detection JSON into a compact SQLite index.
- Decoding of official drivable-area and lane-marking masks.
- Lazy rasterization of legacy unified-JSON lane/drivable polygons when mask
  folders are not present; raw data is never modified.
- A joint dataset interface returning one image, boxes, class IDs, two masks,
  and scene metadata.
- Data auditing, leakage-resistant train/development splitting, and sample
  visualization.
- Dependency-free NumPy implementations of segmentation, lane-tolerance, box
  IoU, and detection AP metrics.
- Unit tests and a synthetic end-to-end self-check that does not need BDD100K.
- Faster R-CNN MobileNetV3-FPN object-detection baseline.
- ResNet18-FPN drivable-area and lane-segmentation baselines.
- Task-aware losses, training/evaluation loops, atomic checkpoints, resume,
  deterministic CPU smoke profiles, and YAML experiment configuration.

## Installation

PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
```

Python 3.10 or newer is supported. PyTorch and Torchvision are required for
Milestone 2; the data-only Milestone 1 CLI remains framework-neutral.

## Fast verification without the dataset

```powershell
roadsense self-check --output outputs/milestone1_self_check
pytest
```

The self-check writes a JSON report and an overlay image. Every metric in the
report should be `1.0` for the intentionally perfect synthetic predictions.

## Prepare BDD100K

The checked-in configuration matches the workspace layout with 100K images in
`data/raw/bdd100k` and the unified legacy label JSON in
`data/raw/bdd100k_labels_release`. The loader rasterizes its lane and drivable
polygons lazily. Explicit official mask directories remain supported by setting
the two mask paths in `configs/data/bdd100k.yaml`.

Validate the configured files and build the memory-safe detection index:

```powershell
roadsense audit --config configs/data/bdd100k.yaml --split train --limit 500 `
  --output outputs/audits/bdd100k_train.json
```

Create a reproducible development split from the official training split:

```powershell
roadsense build-split --config configs/data/bdd100k.yaml --split train `
  --dev-ratio 0.10 --seed 42 --output data/splits/bdd100k_train_dev.json
```

Visualize one sample selected deterministically from the joint dataset:

```powershell
roadsense visualize --config configs/data/bdd100k.yaml --split train `
  --index 0 --output outputs/samples/train_000000.jpg
```

See `docs/milestone_1.md` for the design, expected inputs/outputs, validation
rules, and troubleshooting guide.

## Train and evaluate Milestone 2

First verify all three model/loss graphs without using BDD100K:

```powershell
roadsense-train smoke-test --output outputs/milestone2_smoke
```

Then verify a complete two-sample BDD100K training/evaluation path:

```powershell
roadsense-train train --config configs/experiments/cpu_smoke_drivable.yaml
roadsense-train evaluate --config configs/experiments/cpu_smoke_drivable.yaml `
  --checkpoint outputs/experiments/cpu_smoke_drivable/best.pt
```

Full experiment profiles live in `configs/experiments`. See
`docs/milestone_2.md` for architecture decisions, every public function's
input/output contract, commands, and metric definitions.
