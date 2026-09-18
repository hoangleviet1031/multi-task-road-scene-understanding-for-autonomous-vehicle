# RoadSense-MTL: Conflict-Aware Multi-Task Road Scene Understanding

## Object Detection, Drivable-Area Segmentation and Lane Detection

## 1. Định danh đề tài

**Tên đầy đủ:** RoadSense-MTL: Conflict-Aware Multi-Task Road Scene
Understanding for Object Detection, Drivable-Area Segmentation and Lane
Detection.

**Câu hỏi nghiên cứu trung tâm:**

> Liệu task-specific adapters kết hợp cơ chế xử lý gradient xung đột có thể
> giảm negative transfer trong multi-task road perception, đặc biệt ở điều
> kiện đêm, mưa và dữ liệu ngoài domain hay không?

Đề tài nghiên cứu một mô hình nhận thức cảnh đường đa nhiệm từ camera phía
trước. Ba nhiệm vụ được giải quyết đồng thời gồm:

1. Phát hiện đối tượng giao thông.
2. Phân đoạn vùng có thể di chuyển.
3. Phát hiện vạch kẻ đường.

Điểm khác biệt của đề tài không nằm ở việc chỉ gắn ba head vào một backbone.
Mô hình đề xuất phải giải quyết hai vấn đề cốt lõi của multi-task learning:

- Các nhiệm vụ cần đặc trưng chung nhưng vẫn có nhu cầu biểu diễn riêng.
- Gradient của các nhiệm vụ có thể xung đột trên phần parameter được chia sẻ.

Task-specific adapters xử lý vấn đề thứ nhất. Conflict-aware optimization xử lý
vấn đề thứ hai.

---

## 2. Phạm vi và giả định

### 2.1. Đầu vào

- Một frame RGB từ camera phía trước.
- Ảnh gốc BDD100K có tỷ lệ gần 16:9.
- Canvas đầu vào model là **640×384 pixel**, theo thứ tự `width × height`.
- Trong YAML và tensor shape, kích thước được biểu diễn là `[384, 640]`, theo
  thứ tự `height × width`.
- Ảnh được letterbox để giữ nguyên tỷ lệ hình học, không resize kéo giãn.

Với ảnh nguồn kích thước `W×H`, tỷ lệ letterbox là:

```text
scale = min(640 / W, 384 / H)
new_W = round(W × scale)
new_H = round(H × scale)
```

Phần còn lại được padding đối xứng. Cùng phép biến đổi phải được áp dụng cho
bounding box, drivable mask và lane mask để bảo toàn alignment.

### 2.2. Đầu ra

| Nhiệm vụ | Đầu ra model | Post-processing |
|---|---|---|
| Object detection | Box regression, class score, object confidence | Threshold và NMS |
| Drivable area | Logits `[B,3,384,640]` | `argmax` theo channel |
| Lane detection | Logits `[B,1,384,640]` | `sigmoid` và threshold |

Ba lớp drivable:

```text
0 = direct drivable
1 = alternative drivable
2 = background
```

Lane là bài toán nhị phân:

```text
0 = background
1 = lane foreground
```

Detection dùng 10 lớp BDD100K. Tùy framework, class `0` có thể được dành cho
background và 10 class thật được dịch thành `1..10`.

### 2.3. Giả định kỹ thuật

- Dữ liệu train có annotation cho cả ba nhiệm vụ.
- Một sample chỉ được dùng khi đáp ứng data contract của các task được yêu cầu.
- Metric được tính trong cùng hệ tọa độ letterbox hoặc được ánh xạ ngược nhất quán.
- Thử nghiệm chính thức được thực hiện trên GPU; CPU chỉ dùng kiểm thử pipeline.
- Mọi so sánh MTL/STL dùng cùng split, input resolution và protocol đánh giá.
- Official validation không được dùng để tuning sau khi đã chọn mô hình cuối.

### 2.4. Ngoài phạm vi

- LiDAR/radar fusion.
- 3D object detection.
- Object tracking theo video.
- Motion planning và vehicle control.
- Chứng nhận an toàn cho triển khai trên xe thật.

---

## 3. Mục tiêu nghiên cứu

### 3.1. Mục tiêu tổng quát

Thiết kế và đánh giá một mô hình multi-task road perception có khả năng chia sẻ
đặc trưng hiệu quả, giữ được tính chuyên biệt của từng nhiệm vụ và giảm negative
transfer trong điều kiện thông thường lẫn điều kiện khó.

### 3.2. Mục tiêu cụ thể

1. Xây dựng single-task baselines có protocol tái lập được.
2. Xây dựng hard-sharing multi-task baseline.
3. Thêm task-specific adapters vào các tầng feature dùng chung.
4. Đo gradient norm và cosine similarity giữa từng cặp task.
5. Áp dụng conflict-aware optimizer trên shared parameters.
6. Đánh giá theo điều kiện ban ngày, ban đêm, trời quang và mưa.
7. Thiết lập một protocol out-of-domain có mapping label rõ ràng.
8. So sánh accuracy, robustness, parameter count, memory và latency.

### 3.3. Giả thuyết

- `H1`: Hard sharing tiết kiệm compute nhưng có thể gây negative transfer.
- `H2`: Task-specific adapters cải thiện từng task bằng cách tách phần biểu diễn
  chuyên biệt khỏi shared features.
- `H3`: Gradient surgery giảm tỷ lệ gradient conflict trên shared backbone.
- `H4`: Adapters và gradient surgery bổ sung cho nhau; kết hợp hai thành phần
  tốt hơn việc chỉ dùng một thành phần.
- `H5`: Lợi ích của conflict-aware learning thể hiện rõ hơn ở night/rain/OOD,
  nơi các nhiệm vụ có độ khó và tốc độ hội tụ khác nhau.

---

## 4. Dữ liệu và data contract

### 4.1. Dataset chính

BDD100K được dùng cho huấn luyện và đánh giá in-domain. Workspace hiện có:

| Split nguồn | Số ảnh | Joint samples đủ ba task |
|---|---:|---:|
| Train | 70.000 | 69.863 |
| Validation | 10.000 | 10.000 |

Train source đã được chia thành:

| Split nghiên cứu | Số mẫu |
|---|---:|
| Train | 62.862 |
| Development | 7.001 |

Manifest dùng seed `42` và giữ các frame cùng sequence trong cùng split nhằm
giảm leakage.

### 4.2. Canonical sample

Mỗi sample gồm:

```text
name             string
image            uint8[H,W,3]
boxes            float32[N,4], XYXY
labels           int64[N]
drivable_mask    uint8[H,W]
lane_mask        uint8[H,W]
attributes       weather, scene, time-of-day, ...
task_available   availability của từng task
```

Unified JSON polygons được rasterize khi official PNG masks không có. Raw data
không bị sửa; mask kết quả được cache trong `data/interim`.

### 4.3. Condition slices

BDD attributes được dùng để tạo các evaluation slice:

- `timeofday`: daytime, dawn/dusk, night.
- `weather`: clear, overcast, rainy, snowy, foggy.
- `scene`: city street, highway, residential và các nhóm có đủ sample.

Mỗi slice phải báo cáo cả số sample. Không kết luận từ nhóm quá nhỏ.

### 4.4. Out-of-domain protocol

OOD evaluation cần được định nghĩa trước khi chạy thí nghiệm:

1. Chọn dataset camera đường phố ngoài BDD100K.
2. Lập bảng mapping class và mask ontology.
3. Chỉ tính metric cho các class/task ánh xạ hợp lệ.
4. Không fine-tune trên OOD test set.
5. Báo cáo riêng zero-shot và domain-adapted nếu có adaptation.

Nếu chưa có một dataset ngoài domain chứa đủ cả ba task, có thể dùng hai protocol:

- **Real OOD:** đánh giá từng task trên dataset tương thích label.
- **Controlled shift:** áp dụng corruption cố định lên BDD development/validation
  như giảm sáng, mưa giả lập, blur, noise hoặc color shift, trong khi giữ nguyên
  ground truth.

Controlled shift không thay thế real OOD nhưng giúp đánh giá ba task trên cùng
sample và cùng annotation.

---

## 5. Kiến trúc nghiên cứu đề xuất

Phần này là thiết kế cho milestone tiếp theo, chưa được triển khai trong code
hiện tại.

```text
RGB 640×384
     │
     ▼
Shared encoder: ResNet18/ResNet50
     │ C2, C3, C4, C5
     ▼
Shared Feature Pyramid: P2, P3, P4, P5
     │
     ├──────────────┬──────────────────┐
     │              │                  │
Detection        Drivable           Lane
adapters         adapters            adapters
     │              │                  │
Detection head   Segmentation head  Segmentation head
     │              │                  │
boxes/classes    3-class logits     binary logits
```

### 5.1. Shared encoder

ResNet18 là lựa chọn đầu tiên vì:

- Đã được dùng trong segmentation baseline hiện tại.
- Chi phí phù hợp project sinh viên.
- Có bốn stage feature tự nhiên cho FPN.
- Dễ tạo capacity-matched single-task controls.

ResNet50 chỉ nên dùng sau khi pipeline ResNet18 ổn định và có đủ GPU.

### 5.2. Shared Feature Pyramid Network

FPN tổng hợp semantic context từ tầng sâu và spatial detail từ tầng nông:

```text
P5 = lateral(C5)
P4 = lateral(C4) + upsample(P5)
P3 = lateral(C3) + upsample(P4)
P2 = lateral(C2) + upsample(P3)
```

- Detection chủ yếu dùng multi-scale `P3..P5`, có thể mở rộng `P6/P7`.
- Drivable cần context từ tầng sâu và biên từ tầng nông.
- Lane phụ thuộc mạnh vào `P2/P3` để giữ cấu trúc mảnh.

### 5.3. Task-specific adapters

Adapter được đặt sau các shared FPN levels hoặc tại các stage encoder được chọn.
Một residual bottleneck adapter có dạng:

```text
A_t(x) = x + W_up,t( activation( W_down,t( Norm(x) ) ) )
```

Trong đó:

- `t` là detection, drivable hoặc lane.
- `W_down` giảm số channel.
- `W_up` đưa feature về số channel ban đầu.
- Residual connection giữ shared representation làm đường chính.

Adapter ratio dự kiến:

```text
adapter_channels = shared_channels / r
r ∈ {4, 8, 16}
```

Lý do chọn adapter:

- Ít parameter hơn tách riêng toàn bộ backbone.
- Cho phép mỗi task chỉnh feature theo nhu cầu riêng.
- Có thể ablate vị trí, độ rộng và số adapter.
- Dễ tách gradient shared và task-specific để phân tích.

### 5.4. Detection head

Target architecture nên dùng một detection head nhận trực tiếp shared FPN. Một
head anchor-free kiểu FCOS là ứng viên phù hợp vì dùng multi-scale dense feature
và tránh thiết kế anchor thủ công.

Để kết luận công bằng, cần huấn luyện thêm single-task detection model có cùng:

- Encoder.
- FPN.
- Adapter/head.
- Input resolution.
- Training schedule.

Faster R-CNN MobileNetV3 của Milestone 2 vẫn hữu ích như operational baseline,
nhưng không phải control duy nhất cho negative-transfer analysis.

### 5.5. Drivable head

Các feature đã qua drivable adapters được resize về `P2`, concatenate hoặc cộng
có trọng số, sau đó qua convolution decoder và upsample về `384×640`.

Output:

```text
drivable_logits: float[B,3,384,640]
```

### 5.6. Lane head

Lane head có cấu trúc tương tự nhưng ưu tiên feature resolution cao. Có thể thêm
dilated convolution hoặc refinement block nếu lane bị đứt đoạn.

Output:

```text
lane_logits: float[B,1,384,640]
```

Refinement chỉ được thêm sau ablation cơ bản để không trộn tác động của adapter
với tác động của head phức tạp hơn.

---

## 6. Loss functions

### 6.1. Detection loss

Nếu dùng FCOS-style head:

```text
L_det = L_classification + λ_box L_box + λ_ctr L_centerness
```

Classification có thể dùng focal loss để xử lý foreground/background imbalance.
Box loss dùng IoU/GIoU tùy implementation đã chọn.

### 6.2. Drivable loss

```text
L_drv = λ_ce WeightedCrossEntropy + λ_dice MacroSoftDice
```

Cross-entropy xử lý phân loại từng pixel. Dice giảm bias về background và vùng
có diện tích lớn.

### 6.3. Lane loss

```text
L_lane = λ_bce PositiveWeightedBCE + λ_dice ForegroundDice
```

Positive weighting cần thiết vì lane foreground rất thưa.

### 6.4. Multi-task loss

Baseline cố định:

```text
L_total = w_det L_det + w_drv L_drv + w_lane L_lane
```

Thí nghiệm đầu tiên dùng equal weights hoặc weights chuẩn hóa theo initial loss.
Dynamic weighting chỉ được thêm sau khi fixed-weight baseline ổn định.

---

## 7. Conflict-aware optimization

### 7.1. Định nghĩa gradient xung đột

Với shared parameters `θ_s`, gradient của task `i` là:

```text
g_i = ∇θ_s L_i
```

Hai task được xem là xung đột tại một update nếu:

```text
cos(g_i, g_j) = (g_i · g_j) / (||g_i|| ||g_j||) < 0
```

Cosine âm nghĩa là cập nhật tối ưu cho task này có thành phần làm tăng loss của
task kia trong xấp xỉ bậc nhất.

### 7.2. Phạm vi gradient surgery

Gradient surgery chỉ áp dụng lên:

- Shared encoder parameters.
- Shared FPN parameters.

Không áp dụng lên:

- Task-specific adapters.
- Detection head.
- Drivable head.
- Lane head.

Các parameter riêng chỉ nhận gradient từ task tương ứng.

### 7.3. PCGrad làm phương pháp chính

Nếu `g_i · g_j < 0`, PCGrad chiếu gradient `g_i` khỏi hướng xung đột với `g_j`:

```text
g_i ← g_i - ((g_i · g_j) / ||g_j||²) g_j
```

Sau khi xử lý các cặp task, gradient được tổng hợp và gán cho shared parameters.

Lý do chọn PCGrad làm conflict-aware baseline đầu tiên:

- Cơ chế trực quan.
- Có thể log trước/sau projection.
- Không yêu cầu thêm network học trọng số.
- Phù hợp trực tiếp với câu hỏi nghiên cứu về gradient conflict.

### 7.4. Phương pháp so sánh

- Naive weighted-sum gradients.
- Uncertainty weighting.
- GradNorm.
- PCGrad.
- Adapter + PCGrad, là cấu hình đề xuất chính.

Không cần triển khai mọi phương pháp cùng lúc. Thứ tự khuyến nghị:

1. Hard sharing + fixed weights.
2. Adapters + fixed weights.
3. Hard sharing + PCGrad.
4. Adapters + PCGrad.
5. Một dynamic weighting baseline nếu còn tài nguyên.

### 7.5. Gradient diagnostics

Mỗi epoch hoặc mỗi số batch cố định cần log:

- Gradient norm theo task.
- Pairwise cosine similarity.
- Conflict rate: tỷ lệ cặp có cosine âm.
- Mean/min cosine theo epoch.
- Tỷ lệ gradient bị thay đổi bởi projection.
- Validation metric theo cùng timeline.

Ba cặp chính:

```text
detection ↔ drivable
detection ↔ lane
drivable  ↔ lane
```

---

## 8. Training pipeline

```text
Load joint sample
      │
Aligned augmentation + letterbox 640×384
      │
Shared forward pass
      │
Task-specific adapters and heads
      │
Compute L_det, L_drv, L_lane
      │
Compute per-task gradients on shared parameters
      │
Measure conflicts
      │
Optional PCGrad projection
      │
Combine shared gradients
      │
Apply ordinary task-specific gradients
      │
Optimizer step
      │
Validation, logging and checkpoint
```

### 8.1. Hai giai đoạn huấn luyện đề xuất

**Giai đoạn A — stabilization**

- Khởi tạo encoder từ pretrained weights.
- Train heads/adapters với shared backbone freeze hoặc learning rate nhỏ.
- Mục tiêu là tránh một task có loss lớn phá hỏng representation ban đầu.

**Giai đoạn B — joint fine-tuning**

- Unfreeze shared layers.
- Bật gradient diagnostics.
- Bật fixed weighting hoặc PCGrad theo experiment.
- Chọn checkpoint theo bộ metric đa nhiệm đã định nghĩa trước.

### 8.2. Reproducibility

- Cố định seed.
- Lưu exact YAML config.
- Lưu dataset manifest hash.
- Lưu code commit hash.
- Lưu Python/PyTorch/CUDA versions.
- Lưu model, optimizer, scheduler và scaler state.
- Chạy ít nhất ba seed cho kết luận chính nếu compute cho phép.

---

## 9. Evaluation protocol

### 9.1. Task metrics

| Task | Primary metric | Secondary metrics |
|---|---|---|
| Detection | mAP 0.50:0.05:0.95 | AP50, AP75, per-class AP |
| Drivable | mIoU | Per-class IoU/Dice, mean Dice, pixel accuracy |
| Lane | Tolerance F1 | Precision, recall, exact-pixel IoU/F1 |

Lane tolerance phải được báo cáo cùng resolution. Ví dụ tolerance 2 pixel ở
`384×640` không tương đương tolerance 2 pixel ở ảnh gốc.

### 9.2. Negative transfer

Với metric càng cao càng tốt:

```text
relative_transfer_t = 100 × (MTL_t - STL_t) / STL_t
```

- Giá trị dương: positive transfer.
- Giá trị âm: negative transfer.
- Giá trị gần 0: không thay đổi đáng kể.

Phải dùng capacity-matched STL làm mẫu số chính. Operational Milestone 2
baselines có thể được báo cáo thêm nhưng không thay thế control này.

### 9.3. Aggregate score

```text
average_relative_transfer = mean(relative_transfer_t)
worst_task_transfer = min(relative_transfer_t)
```

Cần báo cáo cả hai. Average tốt không được che việc một task giảm mạnh.

### 9.4. Robustness metrics

Cho từng configuration, báo cáo:

- Overall development/validation metric.
- Day metric.
- Night metric.
- Clear metric.
- Rainy metric.
- Worst-group metric.
- Clean-to-corruption degradation.
- Real-OOD metric nếu có label mapping hợp lệ.

### 9.5. Efficiency metrics

- Tổng parameter và shared parameter.
- Parameter riêng của từng adapter/head.
- Peak GPU memory khi train và infer.
- FLOPs/MACs ở `384×640`.
- Latency batch 1 sau warm-up.
- Throughput FPS.
- Kích thước checkpoint.

---

## 10. Experiment matrix

| ID | Shared backbone | Adapters | Conflict handling | Mục đích |
|---|---|---|---|---|
| E0 | Không | Không | Không | Operational STL baselines hiện tại |
| E1 | Không | Theo task | Không | Capacity-matched STL controls |
| E2 | Có | Không | Weighted sum | Hard-sharing baseline |
| E3 | Có | Có | Weighted sum | Đo riêng tác động adapters |
| E4 | Có | Không | PCGrad | Đo riêng tác động conflict handling |
| E5 | Có | Có | PCGrad | Mô hình đề xuất đầy đủ |
| E6 | Có | Có | GradNorm/uncertainty | So sánh adaptive weighting |

### 10.1. Pairwise ablation

- Detection + drivable.
- Detection + lane.
- Drivable + lane.

Pairwise runs giúp xác định task nào là nguồn conflict thay vì chỉ quan sát mô
hình ba task.

### 10.2. Adapter ablation

- Không adapter.
- Adapter chỉ ở P2/P3.
- Adapter ở P2..P5.
- Reduction ratio `4`, `8`, `16`.
- Adapter trước FPN so với sau FPN.

### 10.3. Robustness ablation

- Train clean, test day/night/rain.
- Train clean, test controlled corruptions.
- Train với condition-aware sampling.
- So sánh mức giảm metric của E2, E3, E4 và E5.

### 10.4. Quy tắc công bằng

- Cùng data manifest.
- Cùng image size `384×640`.
- Cùng augmentation.
- Cùng pretrained initialization.
- Cùng optimizer/scheduler hoặc compute budget.
- Cùng evaluation code.
- Tuning chỉ trên development split.

---

## 11. Tiêu chí trả lời câu hỏi nghiên cứu

Kết luận “adapters + conflict handling giảm negative transfer” chỉ được đưa ra
khi thỏa các điều kiện sau:

1. So sánh với capacity-matched STL và hard-sharing baselines.
2. Cải thiện average relative transfer qua nhiều seed.
3. Không chỉ cải thiện trung bình bằng cách hy sinh nghiêm trọng một task.
4. Conflict rate hoặc negative cosine giảm theo gradient diagnostics.
5. Night/rain/OOD degradation thấp hơn baseline ở mức nhất quán.
6. Chi phí parameter/latency tăng được định lượng rõ.

Nếu adapters tăng metric nhưng cosine conflict không đổi, kết luận phù hợp là
adapters cải thiện specialization, không phải chúng trực tiếp xử lý gradient
conflict. Nếu PCGrad giảm conflict nhưng metric không tăng, cần báo cáo rằng
gradient alignment chưa chuyển thành performance gain.

---

## 12. Failure analysis

Các failure case cần được lưu và phân nhóm:

- Xe nhỏ hoặc bị che khuất.
- Biển báo/đèn giao thông ở xa.
- Drivable boundary không rõ.
- Lane bị mờ, đứt hoặc phản sáng.
- Night glare.
- Rain droplets và motion blur.
- Scene có road geometry khác training domain.

Mỗi failure sample nên hiển thị:

- Ảnh gốc.
- Ground truth của ba task.
- STL predictions.
- Hard-sharing predictions.
- Adapter + PCGrad predictions.
- Metadata weather/scene/time-of-day.

---

## 13. Quan hệ với code hiện tại

### Đã có

- BDD100K joint data contract.
- SQLite detection index.
- Polygon-to-mask rasterization và cache.
- Sequence-aware train/dev manifest.
- Data audit và visualization.
- Detection, drivable và lane metrics.
- Operational single-task baselines.
- Letterbox `384×640` và aligned transforms.
- Train/evaluate/checkpoint/resume engine.
- CPU framework và real-data smoke tests.

### Chưa có

- Shared multi-task encoder/FPN.
- Task-specific adapters.
- Capacity-matched STL controls.
- FCOS-style shared-FPN detection head.
- Multi-task batch/loss orchestration.
- Per-task shared-gradient extraction.
- PCGrad/GradNorm.
- Gradient conflict logging.
- Condition-sliced evaluation CLI.
- OOD/corruption protocol implementation.

Phần “chưa có” chính là phạm vi implementation của các milestone tiếp theo; tài
liệu này không khẳng định chúng đã hoạt động.

---

## 14. Cấu trúc code dự kiến

```text
src/roadsense/
├── models/
│   ├── single_task.py
│   ├── shared_backbone.py       # planned
│   ├── adapters.py              # planned
│   ├── multitask.py             # planned
│   └── heads/                   # planned
│       ├── detection.py
│       ├── drivable.py
│       └── lane.py
├── losses/
│   ├── single_task.py
│   └── multitask.py             # planned
├── optimization/
│   ├── gradient_diagnostics.py  # planned
│   ├── pcgrad.py                # planned
│   └── weighting.py             # planned
├── evaluation/
│   ├── slices.py                # planned
│   └── transfer.py              # planned
└── training/
    ├── engine.py
    └── multitask_engine.py      # planned
```

Các file `planned` chỉ được tạo khi Milestone 3 bắt đầu.

---

## 15. Roadmap triển khai

### Milestone 1 — hoàn thành

- Data contract.
- Audit và split.
- Metrics và visualization.

### Milestone 2 — hoàn thành

- Ba operational STL baselines.
- Training/evaluation/checkpoint.
- Smoke tests trên synthetic và BDD100K thật.

### Milestone 3A — shared baseline

- Shared ResNet-FPN.
- Ba task heads.
- Weighted-sum training.
- Capacity-matched STL controls.

### Milestone 3B — adapters

- Residual bottleneck adapters.
- Adapter placement/reduction ablation.

### Milestone 3C — conflict awareness

- Per-task gradient extraction.
- Cosine/norm logging.
- PCGrad.
- So sánh adapters, PCGrad và kết hợp.

### Milestone 4 — robustness research

- Day/night/rain slices.
- Controlled corruptions.
- Real OOD protocol nếu ontology mapping phù hợp.
- Failure analysis và statistical reporting.

### Milestone 5 — demo và báo cáo

- Overlay cả ba task.
- So sánh STL/MTL trên cùng frame.
- Latency/FPS panel.
- Final tables, plots và thesis report.

---

## 16. Deliverables cuối cùng

- Source code tái lập được.
- Dataset/config manifests, không chứa raw BDD100K trong Git.
- Single-task và multi-task experiment YAMLs.
- Best/latest checkpoints ngoài Git hoặc qua artifact storage.
- Metric tables theo task và condition.
- Gradient conflict plots.
- Adapter/optimizer ablation tables.
- Qualitative failure cases.
- Demo ảnh/video.
- Báo cáo trả lời trực tiếp câu hỏi nghiên cứu trung tâm.

Tài liệu này là research design cho phiên bản Advanced của RoadSense-MTL. Mọi
thay đổi về backbone, detection head, adapter placement hoặc OOD dataset cần
được cập nhật tại đây trước khi dùng làm cơ sở so sánh thực nghiệm.
