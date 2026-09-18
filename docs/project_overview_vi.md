# Tổng quan dự án RoadSense-MTL

## 1. Thông tin chung

**Tên đề tài:** RoadSense-MTL: Conflict-Aware Multi-Task Road Scene Understanding
for Object Detection, Drivable-Area Segmentation and Lane Detection
**Tên mã nguồn:** RoadSense-MTL
**Bài toán:** Nhận thức cảnh đường đa nhiệm từ ảnh camera đơn
**Dataset chính:** BDD100K
**Framework:** Python, PyTorch, Torchvision
**Trạng thái hiện tại:** Milestone 1 và Milestone 2 đã được triển khai; Milestone 3 chưa bắt đầu.
**Input chuẩn:** letterbox canvas `640×384` (`width × height`), tương ứng
`[384,640]` theo thứ tự `height × width` trong YAML/tensor.

Thiết kế chuẩn cho phần nghiên cứu Milestone 3 được quản lý tại
`docs/conflict_aware_research_design_vi.md`. Tài liệu tổng quan này chỉ tóm tắt
thiết kế đó và không thay thế research specification.

RoadSense-MTL nghiên cứu khả năng dùng một hệ thống học sâu để đồng thời thực
hiện ba nhiệm vụ quan trọng trong xe tự hành:

1. Phát hiện đối tượng giao thông.
2. Phân đoạn vùng xe có thể di chuyển.
3. Phát hiện vạch kẻ đường.

Mục tiêu của dự án không chỉ là ghép ba đầu ra vào cùng một model. Trọng tâm
nghiên cứu là xác định khi nào việc chia sẻ biểu diễn giúp các nhiệm vụ hỗ trợ
nhau, khi nào xảy ra **negative transfer**, và chiến lược tối ưu nào cân bằng
được độ chính xác, tốc độ và tài nguyên tính toán.

---

## 2. Bài toán thực sự

Đầu vào của hệ thống là một ảnh RGB từ camera phía trước:

```text
image: uint8 hoặc float tensor [H, W, 3]
```

Từ cùng ảnh đó, hệ thống cần tạo ba loại đầu ra có bản chất khác nhau:

| Nhiệm vụ | Loại bài toán | Đầu ra |
|---|---|---|
| Object detection | Instance-level prediction | Bounding box, class, confidence |
| Drivable-area segmentation | Multi-class semantic segmentation | Mask direct/alternative/background |
| Lane detection | Imbalanced binary segmentation | Mask lane/background |

Các nhiệm vụ có liên hệ nhưng không hoàn toàn đồng nhất:

- Drivable area cần ngữ cảnh không gian rộng để hiểu mặt đường.
- Lane detection cần chi tiết biên và cấu trúc mảnh.
- Object detection cần vừa định vị vừa phân loại từng vật thể.
- Một đặc trưng có lợi cho nhiệm vụ này có thể làm giảm chất lượng nhiệm vụ khác.

Do đó, câu hỏi nghiên cứu cốt lõi là:

> Liệu task-specific adapters kết hợp cơ chế xử lý gradient xung đột có thể
> giảm negative transfer trong multi-task road perception, đặc biệt ở điều
> kiện đêm, mưa và dữ liệu ngoài domain hay không?

---

## 3. Câu hỏi nghiên cứu và giả thuyết

### 3.1. Câu hỏi nghiên cứu

- **RQ1:** Hard parameter sharing gây positive hay negative transfer so với các
  capacity-matched single-task controls?
- **RQ2:** Task-specific adapters có cải thiện specialization của từng head không?
- **RQ3:** Nhiệm vụ nào hỗ trợ nhau và cặp nào tạo gradient cosine âm trên shared parameters?
- **RQ4:** PCGrad có giảm conflict rate và cải thiện Pareto trade-off không?
- **RQ5:** Kết hợp adapters + PCGrad có tốt hơn từng thành phần riêng lẻ không?
- **RQ6:** Lợi ích có ổn định trong điều kiện đêm, mưa, controlled shift và real OOD không?
- **RQ7:** Độ chính xác đạt được phải đánh đổi bao nhiêu parameter, memory và latency?

### 3.2. Giả thuyết

- Drivable area và lane có khả năng hỗ trợ nhau vì cùng phụ thuộc vào hình học mặt đường.
- Detection có thể gây xung đột với dense segmentation ở các feature map độ phân giải cao.
- Task-specific adapters có thể giữ shared representation nhưng cho phép từng
  task học residual feature chuyên biệt với ít parameter bổ sung.
- PCGrad có thể giảm gradient conflict trên shared encoder/FPN, nhưng tăng chi phí huấn luyện.
- Adapters và PCGrad có thể bổ sung cho nhau: một thành phần xử lý specialization,
  thành phần còn lại xử lý optimization conflict.
- MTL có thể giảm tổng parameter và latency so với chạy ba mạng độc lập, ngay cả
  khi độ chính xác của một nhiệm vụ giảm nhẹ.

Các giả thuyết trên là định hướng cho Milestone 3, chưa phải kết luận của dự án.

---

## 4. Phạm vi và ngoài phạm vi

### Trong phạm vi

- Ảnh camera trước dạng monocular.
- Ba nhiệm vụ detection, drivable segmentation và lane segmentation.
- BDD100K làm nguồn dữ liệu chính.
- Baseline đơn nhiệm và mô hình multi-task dùng deep learning.
- Đánh giá accuracy, negative transfer và hiệu quả tính toán.
- Demo suy luận trên ảnh hoặc video sau khi model đã được huấn luyện.

### Ngoài phạm vi hiện tại

- Lập kế hoạch chuyển động và điều khiển xe.
- Sensor fusion với LiDAR, radar hoặc GPS.
- Theo dõi đối tượng qua thời gian.
- Ước lượng depth hoặc 3D detection.
- Chứng nhận an toàn cho hệ thống xe thật.
- Khẳng định kết quả nghiên cứu từ các smoke run 1–2 ảnh.

---

## 5. Trạng thái triển khai

| Milestone | Nội dung | Trạng thái |
|---|---|---|
| 1 | Data foundation, audit, split, metric, visualization | Hoàn thành |
| 2 | Ba baseline đơn nhiệm, training/evaluation/checkpoint | Hoàn thành |
| 3 | Capacity-matched controls, shared network, adapters và PCGrad | Chưa bắt đầu |
| 4 | Ablation, negative-transfer analysis, robustness | Dự kiến |
| 5 | Demo, báo cáo và đóng gói kết quả | Dự kiến |

Nguyên tắc phát triển là không chuyển milestone khi milestone trước chưa được
kiểm thử bằng cả dữ liệu tổng hợp và dữ liệu thật.

---

## 6. Dataset BDD100K

### 6.1. Dữ liệu đang có trong workspace

Cấu hình dữ liệu hiện tại nằm ở `configs/data/bdd100k.yaml` và trỏ tới:

```text
data/raw/
├── bdd100k/bdd100k/images/100k/
│   ├── train/
│   └── val/
└── bdd100k_labels_release/bdd100k/labels/
    ├── bdd100k_labels_images_train.json
    └── bdd100k_labels_images_val.json
```

Các raw file không bị chỉnh sửa. SQLite index và mask rasterized được lưu riêng
trong `data/interim/bdd100k`.

### 6.2. Kết quả inventory thực tế

| Split | Images | Joint samples đủ ba task | Ghi chú |
|---|---:|---:|---|
| Train | 70.000 | 69.863 | 137 ảnh không ghép được với unified label |
| Val | 10.000 | 10.000 | Đủ ba task |

Audit 500 mẫu của mỗi split cho kết quả:

- 500/500 mẫu load được.
- Không có box không hợp lệ hoặc vượt biên.
- Không có lỗi shape ảnh/mask.
- Không có giá trị mask ngoài miền hợp lệ.
- Báo cáo audit có trạng thái `ok: true`.

### 6.3. Train/dev split nghiên cứu

Project không dùng official `val` để điều chỉnh hyperparameter. Từ joint train
set, một manifest train/dev tái lập được đã được tạo:

| Tập | Số mẫu |
|---|---:|
| Train | 62.862 |
| Development | 7.001 |

- Seed: `42`.
- Dev ratio thực tế: khoảng `10,021%`.
- Các ảnh có cùng sequence ID được giữ trong cùng split để giảm leakage.
- Official validation có thể được giữ lại cho đánh giá cuối cùng.

### 6.4. Canonical sample contract

`BDD100KDataset[index]` trả một `RoadSceneSample`:

| Field | Type/shape | Ý nghĩa |
|---|---|---|
| `name` | `str` | ID ổn định từ filename stem |
| `image` | `uint8[H,W,3]` | Ảnh RGB |
| `boxes` | `float32[N,4]` | Bounding box XYXY |
| `labels` | `int64[N]` | ID của 10 lớp detection, bắt đầu từ 0 |
| `drivable_mask` | `uint8[H,W]` | 0 direct, 1 alternative, 2 background |
| `lane_mask` | `uint8[H,W]` | 1 lane, 0 background |
| `attributes` | `dict` | Weather, scene, time of day và metadata |
| `task_available` | `dict` | Trạng thái label của từng task |

Nếu không có mask PNG chính thức, project rasterize `poly2d` từ unified JSON.
Drivable polygon được fill; lane path được vẽ thành binary foreground. Kết quả
được cache atomically để các epoch sau không phải rasterize lại.

---

## 7. Kiến trúc hệ thống hiện tại

```text
                         ┌──────────────────────────┐
                         │      BDD100K raw data     │
                         └─────────────┬────────────┘
                                       │
                         ┌─────────────▼────────────┐
                         │ Milestone 1 data layer    │
                         │ index, decode, rasterize  │
                         └─────────────┬────────────┘
                                       │ RoadSceneSample
                         ┌─────────────▼────────────┐
                         │ aligned transform layer   │
                         │ flip + letterbox          │
                         └──────┬────────┬──────────┘
                                │        │
                 ┌──────────────▼─┐   ┌──▼────────────────┐
                 │ Faster R-CNN   │   │ ResNet18 + FPN     │
                 │ detection      │   │ segmentation       │
                 └──────┬─────────┘   └──┬─────────────┬──┘
                        │                │             │
                    boxes/classes   drivable logits  lane logits
                        │                │             │
                 ┌──────▼────────────────▼─────────────▼──┐
                 │ metrics, checkpoints, JSONL history     │
                 └─────────────────────────────────────────┘
```

Hiện tại các model độc lập, không chia sẻ parameter. Đây là baseline bắt buộc
trước khi nghiên cứu multi-task learning.

---

## 8. Milestone 1 — Data and evaluation foundation

### 8.1. Mục tiêu

Milestone 1 đảm bảo model không học trên dữ liệu sai hoặc split bị leakage.

### 8.2. Thành phần

- Path configuration và validation.
- Streaming parser cho JSON lớn.
- SQLite detection index để random access mà không giữ toàn bộ annotation trong RAM.
- Decoder cho official drivable/lane masks.
- Fallback rasterizer cho legacy unified polygons.
- Joint dataset interface.
- Data audit và sample visualization.
- Group-stratified train/dev split.
- NumPy metrics độc lập với framework.
- Synthetic self-check với prediction hoàn hảo.

### 8.3. Metric foundation

- Bounding-box IoU.
- Per-class AP và mAP.
- Segmentation confusion matrix.
- Per-class IoU và Dice.
- Mean IoU, mean Dice, pixel accuracy.
- Lane precision, recall và F1 có pixel tolerance.

Việc metric không phụ thuộc PyTorch giúp kiểm tra protocol bằng test nhỏ trước
khi kết nối với model.

---

## 9. Milestone 2 — Independent single-task baselines

### 9.1. Detection baseline

**Model:** Faster R-CNN MobileNetV3-Large 320 FPN.

Pipeline cơ bản:

1. Backbone sinh multi-scale feature maps.
2. Region Proposal Network sinh candidate regions.
3. ROI Align trích đặc trưng cho từng proposal.
4. ROI heads phân loại và hồi quy bounding box.
5. Non-maximum suppression loại các dự đoán trùng.

Loss do Torchvision model trả trực tiếp:

- Classification loss.
- Box regression loss.
- Objectness loss.
- RPN box regression loss.

Model có 11 class ở output layer: 10 lớp BDD100K và một background class do
Torchvision quy ước.

### 9.2. Drivable-area baseline

**Model:** ResNet18 encoder + four-level FPN decoder.

Encoder tạo các feature `C2`, `C3`, `C4`, `C5`. FPN chiếu chúng về cùng số
channel, kết hợp top-down, resize về cùng spatial size rồi concatenate. Decoder
trả logits ba lớp ở đúng resolution đầu vào.

Loss:

```text
L_drivable = λ_ce × WeightedCrossEntropy + λ_dice × MacroSoftDice
```

Cross-entropy học phân loại pixel; Dice giảm sự thống trị của background và lớp
có diện tích lớn.

### 9.3. Lane baseline

Lane dùng cùng ResNet18-FPN nhưng chỉ có một output channel.

```text
L_lane = λ_bce × PositiveWeightedBCE + λ_dice × ForegroundDice
```

Positive weight bù cho số pixel lane rất nhỏ so với background. Dice tập trung
vào overlap của foreground mảnh.

### 9.4. Transform

- Horizontal flip chỉ dùng khi training.
- Ảnh, box và mask được flip đồng bộ.
- Letterbox giữ aspect ratio.
- Box được scale, offset, clip và loại nếu chiều rộng/cao còn không quá 1 pixel.
- Drivable padding dùng background ID 2.
- Lane padding dùng background ID 0.
- Ảnh được chuyển về float `[0,1]`; model tự chuẩn hóa theo ImageNet.

### 9.5. Training engine

- Seed Python, NumPy và PyTorch.
- Hỗ trợ CPU, CUDA hoặc tự chọn.
- AdamW hoặc SGD theo YAML.
- CUDA automatic mixed precision khi được bật và CUDA khả dụng.
- Gradient clipping.
- Cosine-annealing learning-rate scheduler.
- Evaluate sau mỗi epoch.
- `latest.pt` cho resume.
- `best.pt` theo mAP, mIoU hoặc lane F1.
- Atomic checkpoint write để tránh file dở nếu tiến trình bị gián đoạn.
- `history.jsonl`, `summary.json` và `evaluation.json` để truy vết.

---

## 10. Thiết kế dự kiến cho Milestone 3

Phần này mô tả hướng nghiên cứu, **chưa được implement**.
Chi tiết có tính quy chuẩn nằm trong `docs/conflict_aware_research_design_vi.md`.

### 10.1. Shared architecture đề xuất

```text
                         Input image
                              │
                     Shared encoder/backbone
                              │
                      Shared feature pyramid
                 ┌────────────┼──────────────┐
                 │            │              │
         Det adapters    Drv adapters    Lane adapters
                 │            │              │
          Detection head  Drivable head   Lane head
                 │            │              │
             boxes/classes  3-class mask  binary mask
```

Kiến trúc chính được đề xuất là shared ResNet18 + FPN để giữ khả năng chạy trong
giới hạn GPU sinh viên. Mỗi task có residual bottleneck adapters và head riêng;
encoder/FPN là phần chia sẻ chịu conflict-aware optimization. ResNet50 chỉ là
mở rộng sau khi pipeline ResNet18 ổn định.

### 10.2. Baseline công bằng bắt buộc

Faster R-CNN MobileNetV3 hiện tại là baseline vận hành tốt, nhưng chưa đủ để kết
luận việc chia sẻ feature gây positive/negative transfer, vì kiến trúc detection
không giống shared model tương lai.

Milestone 3 nên thêm một **capacity-matched single-task baseline**:

- Cùng encoder với MTL.
- Cùng FPN.
- Cùng task head.
- Cùng preprocessing, optimizer, scheduler và số epoch.
- Khác duy nhất ở việc parameter có được chia sẻ hay không.

So sánh này mới cô lập được tác động của multi-task sharing.

### 10.3. Multi-task objective

Với ba task, loss tổng quát:

```text
L_total = w_det L_det + w_drv L_drv + w_lane L_lane
```

Các phương pháp nên thử theo thứ tự:

1. **Fixed weights:** đơn giản, làm control experiment.
2. **Adapters + fixed weights:** cô lập tác động của task specialization.
3. **PCGrad:** chiếu bỏ thành phần gradient xung đột giữa task.
4. **Adapters + PCGrad:** cấu hình đề xuất đầy đủ.
5. **GradNorm/uncertainty weighting:** baseline động nếu còn compute budget.

Gradient surgery chỉ áp dụng cho shared encoder/FPN; adapters và heads nhận
gradient riêng của task tương ứng.

### 10.4. Gradient-conflict logging

Để chứng minh negative transfer thay vì chỉ suy đoán, nên đo:

- Norm gradient của từng task trên shared backbone.
- Cosine similarity giữa từng cặp gradient.
- Tỷ lệ batch có cosine similarity âm.
- Quan hệ giữa gradient conflict và validation metric.

Ba cặp cần theo dõi:

```text
detection ↔ drivable
detection ↔ lane
drivable  ↔ lane
```

---

## 11. Evaluation protocol

### 11.1. Detection

- AP riêng từng class.
- `mAP@0.50`.
- `mAP@0.75`.
- mAP trung bình trên IoU `0.50:0.05:0.95`.
- Với báo cáo công bố, cần đối chiếu evaluator chính thức của BDD100K.

### 11.2. Drivable area

- IoU từng lớp.
- Mean IoU.
- Dice từng lớp.
- Mean Dice.
- Pixel accuracy.
- Confusion matrix.

Mean IoU nên là metric chọn model chính vì ít bị background làm tăng giả tạo hơn
pixel accuracy.

### 11.3. Lane

- Precision.
- Recall.
- F1.
- F1 có symmetric pixel tolerance.

Tolerance được đo ở resolution đã letterbox; mọi báo cáo phải ghi rõ image size
và tolerance để kết quả có thể tái lập.

### 11.4. Multi-task score

Không nên cộng trực tiếp mAP, mIoU và F1 vì thang đo và độ khó khác nhau. Có thể
chuẩn hóa theo single-task baseline:

```text
relative_gain(task) = (MTL_metric - STL_metric) / STL_metric
MTL_score = mean(relative_gain(task))
```

Cần báo cáo cả metric từng task và relative gain. Một mean score tốt không được
che việc một nhiệm vụ bị giảm mạnh.

### 11.5. Efficiency

Ngoài accuracy, nên đo:

- Số parameter trainable.
- FLOPs hoặc MACs tại một resolution cố định.
- Peak GPU memory.
- Training time mỗi epoch.
- Latency batch 1 sau warm-up.
- Throughput FPS.
- Kích thước checkpoint.

Các phép đo latency phải ghi rõ CPU/GPU, precision, batch size, resolution và số
lần warm-up.

---

## 12. Experiment design đề xuất

### 12.1. Ma trận thí nghiệm

| ID | Thiết lập | Mục đích |
|---|---|---|
| E1 | Ba operational single-task baselines | Mốc chạy được của Milestone 2 |
| E2 | Capacity-matched single-task models | Control cho kiến trúc MTL |
| E3 | Hard sharing + fixed equal weights | MTL baseline |
| E4 | Hard sharing + tuned fixed weights | Kiểm tra sensitivity |
| E5 | Uncertainty hoặc GradNorm | Dynamic weighting |
| E6 | PCGrad | Gradient-conflict mitigation |
| E7 | Pairwise tasks | Xác định task interaction |
| E8 | Full three-task model | Kết quả chính |
| E9 | Weather/scene/time slices | Robustness analysis |

### 12.2. Pairwise ablation

- Detection + drivable.
- Detection + lane.
- Drivable + lane.

Pairwise experiments giúp xác định nguồn negative transfer rõ hơn mô hình ba
nhiệm vụ, nơi nhiều tương tác xảy ra đồng thời.

### 12.3. Quy tắc công bằng

- Cùng manifest train/dev.
- Cùng image size và augmentation.
- Cùng pretrained initialization khi có thể.
- Cùng epoch hoặc cùng compute budget.
- Cùng metric implementation.
- Tuning chỉ dùng development split.
- Official validation chỉ dùng sau khi khóa thiết kế.
- Chạy nhiều seed nếu tài nguyên cho phép; tối thiểu 3 seed cho kết luận nghiên cứu.

### 12.4. Báo cáo kết quả

Mỗi experiment nên lưu:

```text
config.yaml
environment.json
history.jsonl
summary.json
best.pt
latest.pt
evaluation.json
qualitative_samples/
```

Nên lập bảng gồm mean ± standard deviation qua các seed, không chỉ báo cáo lần
chạy tốt nhất.

---

## 13. Cấu trúc repository

```text
roadsense/
├── configs/
│   ├── data/
│   │   └── bdd100k.yaml
│   └── experiments/
│       ├── detection_baseline.yaml
│       ├── drivable_baseline.yaml
│       ├── lane_baseline.yaml
│       └── cpu_smoke_*.yaml
├── data/
│   ├── raw/                 # BDD100K gốc, không sửa
│   ├── interim/             # SQLite index và rasterized cache
│   └── splits/              # Reproducible manifests
├── docs/
│   ├── milestone_1.md
│   ├── milestone_2.md
│   └── project_overview_vi.md
├── outputs/
│   ├── audits/
│   ├── experiments/
│   └── milestone*_smoke/
├── src/roadsense/
│   ├── data/                # Dataset, decoder, index, audit, split
│   ├── losses/              # Single-task losses
│   ├── metrics/             # Detection/segmentation metrics
│   ├── models/              # Single-task models
│   ├── training/            # Config, DataLoader, engine, CLI
│   ├── cli.py               # Milestone 1 CLI
│   ├── types.py             # Canonical data types
│   └── visualization.py
├── tests/
├── README.md
└── pyproject.toml
```

---

## 14. Cài đặt môi trường

### CPU

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
```

### CUDA

Trên máy NVIDIA, cần cài PyTorch wheel tương thích CUDA trước, sau đó cài project.
Không nên mặc định rằng wheel CPU hiện tại sẽ sử dụng được GPU.

Kiểm tra môi trường:

```powershell
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print(torch.cuda.is_available())"
python -m pip check
```

---

## 15. Chuẩn bị và kiểm tra dữ liệu

Audit train:

```powershell
roadsense audit `
  --config configs/data/bdd100k.yaml `
  --split train `
  --limit 500 `
  --output outputs/audits/bdd100k_train.json
```

Tạo train/dev manifest:

```powershell
roadsense build-split `
  --config configs/data/bdd100k.yaml `
  --split train `
  --dev-ratio 0.10 `
  --seed 42 `
  --output data/splits/bdd100k_train_dev.json
```

Visualize một mẫu:

```powershell
roadsense visualize `
  --config configs/data/bdd100k.yaml `
  --split train `
  --index 0 `
  --output outputs/samples/train_000000.jpg
```

---

## 16. Huấn luyện và đánh giá

### 16.1. Framework smoke test

Không cần dataset:

```powershell
roadsense-train smoke-test --output outputs/milestone2_smoke
```

Lệnh này thực hiện forward và backward cho cả ba baseline, kiểm tra finite loss
và ghi `report.json`.

### 16.2. Real-data smoke tests

```powershell
roadsense-train train --config configs/experiments/cpu_smoke_detection.yaml
roadsense-train train --config configs/experiments/cpu_smoke_drivable.yaml
roadsense-train train --config configs/experiments/cpu_smoke_lane.yaml
```

Các profile chỉ dùng 1–2 mẫu, một epoch và backbone không pretrained. Chúng kiểm
tra integration, không tạo kết quả dùng trong báo cáo khoa học.

### 16.3. Full baselines

```powershell
roadsense-train train --config configs/experiments/detection_baseline.yaml
roadsense-train train --config configs/experiments/drivable_baseline.yaml
roadsense-train train --config configs/experiments/lane_baseline.yaml
```

Máy hiện tại chỉ có CPU nên các full run sẽ rất chậm. Nên sử dụng GPU cho các
experiment chính thức.

### 16.4. Resume

```powershell
roadsense-train train `
  --config configs/experiments/drivable_baseline.yaml `
  --resume outputs/experiments/drivable_baseline/latest.pt
```

### 16.5. Evaluation độc lập

```powershell
roadsense-train evaluate `
  --config configs/experiments/drivable_baseline.yaml `
  --checkpoint outputs/experiments/drivable_baseline/best.pt
```

---

## 17. Kiểm thử và reproducibility

### Test tự động

```powershell
python -m pytest
```

Test hiện tại bao phủ:

- Label decoding.
- Polygon rasterization.
- Split determinism và sequence grouping.
- Detection/segmentation metrics.
- Synthetic end-to-end self-check.
- Experiment config validation.
- Letterbox box/mask alignment.
- Segmentation model output shapes.
- Loss backward pass.

### Reproducibility controls

- Seed được lưu trong YAML và manifest.
- DataLoader shuffle dùng seeded generator.
- cuDNN deterministic mode được bật.
- Config gốc được lưu trong checkpoint.
- Checkpoint chứa model, optimizer, scheduler, epoch và best metric.
- Fresh run xóa history cũ; resume tiếp tục state đã lưu.

Determinism tuyệt đối giữa các GPU/driver khác nhau không được đảm bảo. Báo cáo
nên lưu thêm phiên bản Python, PyTorch, Torchvision, CUDA và thông tin phần cứng.

---

## 18. Kết quả kiểm tra hiện tại

Milestone 2 đã được xác nhận bằng:

- Toàn bộ test tự động vượt qua.
- `pip check` không phát hiện dependency hỏng.
- Synthetic forward/backward thành công cho cả ba model.
- Real BDD100K smoke training thành công cho detection, drivable và lane.
- Checkpoint `best.pt` và `latest.pt` được tạo.
- Checkpoint detection và drivable đã được load lại bằng evaluation command.

Metric của smoke run không đại diện chất lượng model do chỉ dùng 1–2 ảnh và một
epoch. Chưa có full-baseline result để kết luận accuracy.

---

## 19. Rủi ro và giới hạn kỹ thuật

### 19.1. Compute

- Full BDD100K training trên CPU không thực tế.
- Detection thường là task tốn thời gian và memory nhất.
- Multi-task model có thể cần giảm resolution hoặc batch size.

### 19.2. Label imbalance

- Car và background xuất hiện nhiều hơn các class hiếm.
- Lane foreground chỉ chiếm tỷ lệ pixel rất nhỏ.
- Metric trung bình có thể che chất lượng class hiếm.

### 19.3. Protocol mismatch

- Local AP evaluator có mục tiêu minh bạch và unit testing.
- Kết quả cuối cần kiểm tra lại bằng official BDD100K evaluator.
- Lane F1 phụ thuộc resolution và tolerance.

### 19.4. Fairness của MTL comparison

- So sánh MTL với một baseline khác backbone có thể gây kết luận sai.
- Cần capacity-matched baselines trước khi công bố positive/negative transfer.

### 19.5. Dataset bias

- BDD100K không đại diện mọi quốc gia, camera, biển báo hoặc điều kiện đường.
- Performance trung bình không đồng nghĩa an toàn trong trường hợp hiếm.

---

## 20. Tiêu chí hoàn thành project Advanced

Project có thể được xem là hoàn chỉnh khi đáp ứng đồng thời:

1. Ba single-task baselines được train đầy đủ và có metric tin cậy.
2. Có capacity-matched STL control.
3. Có ít nhất một hard-sharing MTL baseline.
4. Có ít nhất một phương pháp adaptive weighting hoặc gradient balancing.
5. Báo cáo metric từng task, efficiency và relative transfer.
6. Có pairwise hoặc gradient-conflict ablation.
7. Có đánh giá theo weather/scene/time-of-day.
8. Có qualitative failure analysis.
9. Có demo ảnh hoặc video.
10. Toàn bộ experiment có config, seed, checkpoint và environment metadata.

---

## 21. Hướng demo cuối dự án

Demo đề xuất nhận ảnh, video hoặc webcam và hiển thị:

- Bounding box và confidence.
- Drivable-area overlay bán trong suốt.
- Lane overlay màu riêng.
- FPS/latency.
- Công tắc bật/tắt từng task.
- So sánh single-task và multi-task trên cùng frame.

Với demo nghiên cứu, nên có thêm panel hiển thị:

- Metric trên sample có ground truth.
- Model configuration.
- Số parameter.
- Latency và memory.
- Failure cases ở ban đêm hoặc thời tiết xấu.

---

## 22. Tài liệu liên quan trong repository

- `README.md`: hướng dẫn bắt đầu nhanh.
- `docs/README.md`: index, trạng thái và quy tắc đồng bộ tài liệu.
- `docs/milestone_1.md`: data contract, audit, split và metric foundation.
- `docs/milestone_2.md`: kiến trúc baseline, function contracts và lệnh chạy.
- `docs/conflict_aware_research_design_vi.md`: research specification cho
  task-specific adapters, PCGrad, robustness và OOD evaluation.
- `configs/data/bdd100k.yaml`: vị trí và encoding dữ liệu.
- `configs/experiments/*.yaml`: cấu hình huấn luyện tái lập được.

Tài liệu này là bản tổng quan cấp project. Khi Milestone 3 được thiết kế và triển
khai, phần kiến trúc/loss/experiment tương ứng cần được cập nhật từ “dự kiến”
thành “đã triển khai”, kèm kết quả kiểm thử và giới hạn thực tế.
