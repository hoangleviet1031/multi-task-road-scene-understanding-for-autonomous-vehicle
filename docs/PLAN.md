# Kế hoạch dự án RoadSense-MTL

> Tài liệu chuẩn duy nhất cho phạm vi, kiến trúc nghiên cứu, lộ trình và tiêu chí
> nghiệm thu của dự án. Trạng thái triển khai hằng ngày được quản lý riêng tại
> [`AI_PROGRESS.md`](AI_PROGRESS.md).

## 1. Định danh đề tài

**Tên đề tài:** RoadSense-MTL: Conflict-Aware Multi-Task Road Scene Understanding
for Object Detection, Drivable-Area Segmentation and Lane Detection.

**Câu hỏi nghiên cứu trung tâm:** task-specific adapters kết hợp cơ chế xử lý
gradient xung đột có giảm negative transfer trong multi-task road perception,
đặc biệt ở điều kiện đêm, mưa và dữ liệu ngoài domain hay không?

Dự án không chỉ ghép ba head vào một model. Đóng góp cần được chứng minh bằng
thực nghiệm: khi nào sharing có lợi, khi nào gây negative transfer, và adapters
cùng conflict-aware optimization giải quyết vấn đề đến mức nào.

## 2. Phạm vi

### 2.1. Đầu vào và đầu ra

- Đầu vào: một frame RGB từ camera phía trước.
- Tỷ lệ ảnh nguồn BDD100K thường là `16:9`.
- Canvas chuẩn: letterbox `640×384` (`width × height`), tương ứng `[384, 640]`
  theo thứ tự `height × width` trong YAML và tensor.
- Object detection: bounding box, lớp và confidence cho 10 lớp BDD100K.
- Drivable-area segmentation: `direct`, `alternative`, `background`.
- Lane-marking segmentation: foreground/background.

### 2.2. Ngoài phạm vi phiên bản đầu

- Điều khiển xe và lập kế hoạch đường đi.
- Localization, tracking và temporal model.
- 3D detection, depth estimation và sensor fusion.
- Chứng nhận an toàn cho hệ thống xe thật.

Các nội dung trên chỉ được đưa vào future work, không làm loãng câu hỏi nghiên
cứu chính.

### 2.3. Giả định tài nguyên

- Mục tiêu: 1 GPU có 16–24 GB VRAM, 12–16 tuần phát triển.
- Với 8 GB VRAM: ưu tiên ResNet18, giảm input về `512×288` hoặc dùng gradient
  accumulation.
- Smoke run 1–2 ảnh chỉ kiểm tra plumbing, không được dùng làm kết quả nghiên cứu.

## 3. Dữ liệu và data contract

### 3.1. Dataset chính

BDD100K là nguồn dữ liệu chính. Thí nghiệm cốt lõi bắt đầu từ giao của các ảnh có
đủ detection, drivable và lane label. Partial-label training chỉ là mở rộng sau
khi pipeline chính ổn định.

Mỗi sample chuẩn gồm:

| Trường | Kiểu/shape | Ý nghĩa |
|---|---|---|
| `name` | `str` | ID ổn định từ filename |
| `image` | `uint8[H,W,3]` | Ảnh RGB |
| `boxes` | `float32[N,4]` | Box XYXY |
| `labels` | `int64[N]` | ID 10 lớp detection, bắt đầu từ 0 |
| `drivable_mask` | `uint8[H,W]` | 0 direct, 1 alternative, 2 background |
| `lane_mask` | `uint8[H,W]` | 1 lane, 0 background |
| `attributes` | `dict` | Weather, scene, time of day |
| `task_available` | `dict` | Cờ availability cho từng task |

Raw data không được sửa. Index, cache mask và artifact trung gian phải nằm trong
`data/interim` hoặc `outputs` và không commit vào Git.

### 3.2. Split

- Tạo train/dev từ official train bằng split theo sequence để tránh leakage.
- Official validation được giữ cho đánh giá cuối.
- Manifest phải lưu seed, số lượng và phân bố các condition.
- Frame từ cùng video/sequence không được xuất hiện ở hai split khác nhau.

### 3.3. Augmentation

Mọi geometric transform phải đồng bộ trên ảnh, boxes và cả hai masks.

Nên dùng:

- Letterbox resize, horizontal flip.
- Scale/crop có kiểm soát.
- Color jitter, brightness/contrast, Gaussian blur nhẹ.
- Synthetic rain/fog vừa phải và random shadow dưới dạng experiment riêng.

Chưa dùng trong baseline chính: mạnh tay với Mosaic, MixUp giữa cảnh hoặc
perspective transform lớn, vì có thể làm mask lane/drivable thiếu tự nhiên.

## 4. Kiến trúc nghiên cứu mục tiêu

```text
RGB 640×384
     │
Shared ResNet encoder → C2, C3, C4, C5
     │
Shared FPN → P2, P3, P4, P5 (+ P6/P7 nếu detection cần)
     │
     ├──────────────┬──────────────────┐
     │              │                  │
Detection adapter  Drivable adapter   Lane adapter
     │              │                  │
FCOS-style head    3-class head       lane-aware head
```

### 4.1. Shared encoder và FPN

- ResNet18 là backbone đầu tiên để ổn định pipeline và tạo control phù hợp tài
  nguyên; ResNet50 là cấu hình nâng cấp/ablation khi có đủ GPU.
- FPN tạo `P2..P5` cùng số channel, mặc định nghiên cứu là 128 hoặc 256.
- Lane ưu tiên `P2/P3`; detection dùng `P3..P5` và có thể thêm `P6/P7`;
  drivable kết hợp chi tiết tầng nông với context tầng sâu.

### 4.2. Task-specific gated adapters

Mỗi task có residual bottleneck adapter tại các FPN level được chọn:

```text
A_t(x) = x + sigmoid(g_t) ⊙ W_up,t(DWConv(activation(W_down,t(Norm(x)))))
adapter_channels = shared_channels / r,  r ∈ {4, 8, 16}
```

Adapters thuộc task-specific parameters. Cần log gate strength và chi phí
parameter/latency; không được gộp chúng vào nhóm shared khi chạy PCGrad.

### 4.3. Detection head

- Kiến trúc nghiên cứu mục tiêu: FCOS-style anchor-free head trên multi-scale FPN.
- Mỗi vị trí dự đoán class probability, box offsets và centerness.
- Loss: focal classification + GIoU/IoU box loss + centerness loss.
- Faster R-CNN MobileNetV3-FPN hiện có là operational baseline của Milestone 2,
  không phải capacity-matched control duy nhất để kết luận negative transfer.

### 4.4. Drivable head

- Resize `P2..P5` về resolution của `P2`, fuse và decode về kích thước input.
- Output `float[B,3,H,W]`.
- Loss: weighted cross-entropy + macro soft Dice.

### 4.5. Lane head

- Ưu tiên `P2/P3`, lấy thêm context từ `P4/P5`.
- Có thể dùng asymmetric `1×7`/`7×1` convolution; refinement phức tạp chỉ thêm
  sau ablation cơ bản.
- Output `float[B,1,H,W]`.
- Loss baseline: positive-weighted BCE hoặc focal/Tversky + Dice; boundary loss
  là mở rộng cần ablation riêng.

## 5. Multi-task optimization

### 5.1. Baseline

```text
L_total = w_det L_det + w_drv L_drv + w_lane L_lane
```

Bắt đầu bằng fixed/equal weights hoặc weights chuẩn hóa theo initial loss. Sau
khi baseline ổn định mới bật uncertainty weighting:

```text
L_weighted = Σ exp(-s_t) L_t + s_t
```

### 5.2. PCGrad

Với shared parameters `θ_s`, hai task xung đột khi:

```text
cos(g_i, g_j) < 0,  với g_t = ∇θ_s L_t
```

PCGrad chiếu bỏ thành phần xung đột:

```text
g_i ← g_i - ((g_i · g_j) / ||g_j||²) g_j
```

PCGrad chỉ áp dụng cho shared encoder và shared FPN. Adapters và heads nhận
gradient thông thường từ task tương ứng.

Mỗi epoch hoặc theo interval cố định phải log:

- Gradient norm theo task.
- Pairwise cosine cho detection–drivable, detection–lane, drivable–lane.
- Conflict rate, mean/min cosine.
- Tỷ lệ gradient bị projection.
- Trọng số uncertainty nếu được bật.

## 6. Training và inference

### 6.1. Training pipeline

1. Load pretrained backbone và khởi tạo FPN/adapters/heads.
2. Warm-up 3–5 epoch: freeze một phần backbone, fixed weights, chưa bật PCGrad.
3. Joint training: unfreeze, bật weighting/PCGrad theo experiment.
4. Dùng gradient clipping, cosine learning-rate schedule và mixed precision khi
   implementation tương thích.
5. Theo dõi multi-task development score và từng task metric.
6. Lưu `last`, `best overall` và `best per task` checkpoint.
7. Mọi kết quả phải truy được về config, commit, seed và checkpoint.

### 6.2. Inference pipeline

```text
frame → letterbox/normalize → one shared forward
      → boxes + NMS
      → resize drivable mask
      → resize/thin lane mask
      → overlay + confidence + timing
```

Latency end-to-end phải bao gồm preprocessing và post-processing nếu báo cáo là
end-to-end FPS.

## 7. Evaluation protocol

### 7.1. Task metrics

- Detection: mAP@[0.5:0.95], AP50, AP small/medium/large, per-class AP, recall,
  false positives per scene.
- Drivable: mIoU, per-class IoU, Dice/F1 và boundary F1.
- Lane: precision, recall, tolerance F1, lane IoU, boundary F1 và fragmentation.

Protocol lane phải cố định độ dày ground truth, threshold và pixel tolerance.

### 7.2. Negative transfer và aggregate score

```text
Δ_t = (M_t^MTL - M_t^STL) / M_t^STL
Δ_MTL = mean_t(Δ_t)
```

Mẫu số chính phải là capacity-matched single-task control. Báo cáo cả từng
`Δ_t`, không chỉ giá trị trung bình. Dùng Pareto plot giữa normalized multi-task
score và latency/FLOPs.

### 7.3. Robustness và domain shift

Đánh giá theo day/night/dawn, clear/rain/fog, highway/city/residential, small và
occluded objects.

Tập Vietnam Road Domain dự kiến khoảng 300 frame từ các video tách biệt:

- 100 adaptation-train.
- 50 development.
- 150 held-out test.

Hai protocol: zero-shot từ BDD100K và few-shot adaptation. Nếu thiếu nguồn lực
gán nhãn, ưu tiên held-out test; adaptation chuyển thành future work.

```text
D_drop = (M_BDD - M_VN) / M_BDD
```

### 7.4. Efficiency

Đo batch size 1: FPS, mean/p50/p95 latency, parameter, FLOPs, peak VRAM, model
size và thời gian pre/model/post-processing. Ghi rõ hardware, resolution, FP32
hay FP16, warm-up và việc timing có bao gồm visualization hay không.

## 8. Experiment matrix

| ID | Shared | Adapters | Weighting/gradient | Mục tiêu |
|---|---|---|---|---|
| E0 | Không | Theo task | Standard | Ba capacity-matched STL controls |
| E1 | Có | Không | Fixed weighted sum | Hard-sharing baseline |
| E2 | Có | Không | Uncertainty | Tác động loss weighting |
| E3 | Có | Không | PCGrad | Tác động gradient surgery |
| E4 | Có | Có | Fixed weighted sum | Tác động adapters |
| E5 | Có | Có | Uncertainty | Adapters + adaptive weighting |
| E6 | Có | Có | PCGrad (+ uncertainty nếu ablate) | Proposed model |

Quy tắc công bằng: giữ nguyên input, split, initialization, augmentation, batch
hiệu dụng, số optimizer step, LR schedule, evaluation code, threshold và NMS.

Nếu compute hạn chế:

1. Screening E1–E6 trên subset 20.000 ảnh với một seed.
2. Chọn E1, biến thể trung gian tốt nhất và E6.
3. Chạy ba cấu hình đó trên full data với ba seed.

Final run báo cáo mean ± standard deviation và paired bootstrap confidence
interval; không chỉ báo cáo checkpoint tốt nhất của một run.

## 9. Lộ trình và exit criteria

| Giai đoạn | Nội dung | Exit criteria chính |
|---|---|---|
| M1 — Data foundation | Taxonomy, audit, intersection, split, visualization, metric | Sample joint đúng shape/mapping; split không leakage; metric qua synthetic test |
| M2 — STL baselines | Detection, drivable, lane model độc lập | Loss giảm, prediction có nghĩa, metric/timing/memory baseline có thể tái lập |
| M3A — Shared baseline | Shared ResNet-FPN, 3 heads, weighted sum, STL controls | Ba task cùng học; không collapse; có STL-vs-MTL và gradient norm |
| M3B — Adapters | Gated adapters và placement/reduction ablation | Overhead được đo; gate được log; ít nhất một task/aggregate score cải thiện |
| M3C — Conflict awareness | Per-task gradients, diagnostics, PCGrad/weighting | PCGrad chỉ chạm shared params; conflict được log; chọn final strategy |
| M4 — Full experiments | Full data, 3 seeds, efficiency, robustness/OOD | Mean±std, Pareto, domain degradation và failure taxonomy |
| M5 — Demo/report | Video/UI, ablation, failure gallery, báo cáo | Demo tái lập và trả lời rõ các research questions |

Không bắt đầu milestone mới chỉ vì file skeleton đã tồn tại. Milestone được coi là
hoàn thành khi code, test, lệnh tái lập và evidence đều có đủ.

## 10. Tiêu chí thành công mục tiêu

Đây là mục tiêu nghiên cứu, không phải kết quả được đảm bảo:

- Không task nào giảm quá 5% tương đối so với capacity-matched STL.
- Joint model giảm ít nhất 35% tổng parameter so với ba model riêng.
- Proposed model tốt hơn hard sharing về normalized multi-task score.
- PCGrad làm giảm measured conflict rate.
- Adapter tăng latency không quá 10%.
- Proposed model có domain drop thấp hơn hard-sharing baseline.

## 11. Cấu trúc repository mục tiêu

```text
configs/{data,model,optimization,experiments}/
data/{raw,interim,processed,splits}/
src/roadsense/
  data/
  models/{backbones,necks,adapters,heads,multitask}/
  losses/{detection,segmentation,weighting,multitask}/
  optimization/{gradient_surgery,schedulers}/
  training/
  metrics/
  analysis/{gradient_conflict,robustness,domain_shift,pareto,failure_cases}/
  inference/
scripts/{data,train,evaluate,inference,analysis}/
tests/{data,models,losses,metrics,integration}/
docs/
outputs/{checkpoints,logs,metrics,predictions,figures,demo}/
```

Chỉ tạo thư mục/file khi milestone tương ứng bắt đầu. Không tạo skeleton trống chỉ
để giống cây mục tiêu. Notebook chỉ dùng khám phá; logic chính phải nằm trong
`src/roadsense` và có test.

## 12. Deliverables cuối

- Ba capacity-matched single-task baselines.
- Hard-sharing multi-task baseline.
- Proposed model có gated adapters và conflict-aware optimization.
- Training/evaluation protocol tái lập được.
- Bảng accuracy, efficiency và robustness.
- Gradient-conflict analysis và ablation study.
- Tập đánh giá nhỏ cho giao thông Việt Nam.
- Video demo, failure-case report và báo cáo trả lời research questions.

## 13. Quy tắc quản trị tài liệu

- File này sở hữu phạm vi, kế hoạch và thiết kế tương lai.
- `milestone_1.md` và `milestone_2.md` mô tả chi tiết code đã triển khai.
- `AI_PROGRESS.md` sở hữu trạng thái thực tế, evidence và next actions.
- Khi code thay đổi, cập nhật tracker trong cùng commit; chỉ sửa kế hoạch khi
  phạm vi hoặc quyết định kiến trúc thực sự thay đổi.
- Không mô tả planned component như thể đã hoạt động.
