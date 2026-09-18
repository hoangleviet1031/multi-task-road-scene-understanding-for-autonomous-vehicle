# Milestone 1 — Data and evaluation foundation

## Scope

This milestone establishes a trustworthy input contract before any neural
network is implemented. It answers four questions:

1. Can every image be matched to detection, drivable-area, and lane labels?
2. Can official BDD100K encodings be decoded without silently changing labels?
3. Can the train/development split be reproduced without sequence leakage?
4. Do metrics return known answers on controlled synthetic examples?

Model definition and training belong to Milestone 2 and are deliberately absent.

## Canonical sample contract

`BDD100KDataset[index]` returns a `RoadSceneSample` with:

| Field | Type and shape | Meaning |
|---|---|---|
| `name` | string | Filename stem used as the stable sample ID |
| `image` | `uint8[H,W,3]` | RGB image |
| `boxes` | `float32[N,4]` | Boxes in `x1,y1,x2,y2` pixel coordinates |
| `labels` | `int64[N]` | Zero-based IDs for the 10 detection classes |
| `drivable_mask` | `uint8[H,W]` | 0 direct, 1 alternative, 2 background |
| `lane_mask` | `uint8[H,W]` | 1 lane foreground, 0 background |
| `attributes` | dictionary | Weather, scene, time of day, and available metadata |
| `task_available` | dictionary | Whether each requested task has a label |

When explicit PNG masks are absent, the same contract is produced by lazily
rasterizing the unified label JSON's Scalabel `poly2d` paths. Cubic Bézier
commands are sampled before drawing; drivable regions are filled and lane paths
are stroked as binary foreground. Raw BDD100K files are never rewritten.

The object is framework-neutral. A future PyTorch adapter can convert these
arrays to tensors without coupling data validation to the training framework.

## Public functions and their input/output

### Configuration

- `load_dataset_config(path)`
  - Input: YAML path.
  - Output: validated `DatasetConfig` containing absolute dataset paths.
- `DatasetConfig.for_split(split)`
  - Input: split name such as `train` or `val`.
  - Output: `SplitPaths` with image, JSON, and mask locations.

### BDD100K labels

- `decode_drivable_mask(mask, direct_id, alternative_id, background_id)`
  - Input: one-channel official mask.
  - Output: normalized `uint8[H,W]` mask with IDs 0, 1, and 2.
- `decode_lane_binary_mask(mask, background_bit)`
  - Input: one-channel official bit-packed lane mask.
  - Output: binary `uint8[H,W]` mask.
- `normalize_detection_frame(frame)`
  - Input: one Scalabel/BDD100K JSON frame dictionary.
  - Output: normalized image name, metadata, sequence ID, valid boxes, and
    legacy lane/drivable polygon annotations.
- `rasterize_legacy_labels(frame, image_size, lane_width)`
  - Input: normalized frame plus image width/height.
  - Output: normalized drivable mask and binary lane mask.

### Detection index

- `DetectionAnnotationIndex.ensure_built(rebuild=False)`
  - Input: optional rebuild flag.
  - Output: number of indexed frames; creates a SQLite cache when necessary.
- `DetectionAnnotationIndex.get(stem)`
  - Input: image filename stem.
  - Output: normalized `DetectionFrame`, or `None` if absent.

### Dataset and auditing

- `BDD100KDataset(config, split, required_tasks)`
  - Input: configuration, split name, and required task names.
  - Output: lazy dataset containing only the intersection required by the tasks.
- `BDD100KDataset[index]`
  - Input: integer index.
  - Output: one `RoadSceneSample` following the contract above.
- `audit_dataset(dataset, limit)`
  - Input: dataset and maximum number of samples whose pixels are inspected.
  - Output: serializable report with inventory, invalid boxes, mask values,
    shape errors, class counts, and scene distributions.
- `create_group_stratified_split(records, dev_ratio, seed)`
  - Input: sample metadata records, target ratio, and random seed.
  - Output: train IDs, development IDs, and per-stratum summaries. Samples with
    the same sequence ID are kept together.

### Metrics

- `confusion_matrix(prediction, target, num_classes, ignore_index)`
  - Input: equal-shaped integer masks.
  - Output: matrix whose row is ground truth and column is prediction.
- `segmentation_metrics(...)`
  - Input: prediction and target masks.
  - Output: per-class IoU/Dice plus mean IoU, mean Dice, and pixel accuracy.
- `lane_f1_with_tolerance(prediction, target, tolerance)`
  - Input: binary masks and a pixel tolerance.
  - Output: tolerant precision, recall, and F1.
- `box_iou(boxes_a, boxes_b)`
  - Input: `N×4` and `M×4` boxes.
  - Output: `N×M` pairwise IoU matrix.
- `mean_average_precision(predictions, targets, class_ids, iou_thresholds)`
  - Input: per-image prediction and target records.
  - Output: AP for every class/threshold and their mean.

### Visualization

- `render_sample(sample)`
  - Input: one `RoadSceneSample`.
  - Output: Pillow RGB image containing masks, boxes, labels, and metadata.
- `save_sample_visualization(sample, output_path)`
  - Input: sample and destination path.
  - Output: resolved written file path.

## Validation rules

The audit reports rather than silently fixes:

- Missing task annotations.
- Image/mask shape mismatches.
- Unknown drivable mask values.
- Invalid or out-of-image bounding boxes.
- Missing or malformed scene attributes.

Lane bit packing is decoded using the official layout: category in bits 0–2,
background flag in bit 3, style in bit 4, and direction in bit 5.

## Expected outputs

`self-check` produces:

```text
outputs/milestone1_self_check/
├── report.json
└── synthetic_overlay.jpg
```

`audit` produces a JSON report. `build-split` produces a JSON manifest whose
sample names can later be consumed by the training pipeline.
