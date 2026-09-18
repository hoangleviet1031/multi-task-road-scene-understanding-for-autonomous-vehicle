# RoadSense-MTL — AI progress tracker

> File handoff dành cho AI/cộng tác viên. Đọc file này và
> [`PLAN.md`](PLAN.md) trước khi sửa code.

## Snapshot

| Trường | Giá trị |
|---|---|
| Cập nhật lần cuối | 2026-09-18 |
| Milestone hiện tại | M2 — independent single-task baselines |
| Phạm vi được phép hiện tại | Duy trì/kiểm chứng M1–M2 và chuẩn bị tài liệu |
| Milestone 3 | Chưa bắt đầu; không triển khai nếu chưa có yêu cầu mới |
| Input chuẩn | `[384, 640]` trong YAML/tensor (`640×384` width×height) |
| Dataset chính | BDD100K |
| Split manifest | `data/splits/bdd100k_train_dev.json` |

## Quy ước trạng thái

- `NOT_STARTED`: chưa có implementation.
- `IN_PROGRESS`: đang làm, chưa đủ acceptance evidence.
- `BLOCKED`: không thể tiếp tục nếu thiếu input/quyền/tài nguyên cụ thể.
- `IMPLEMENTED`: code và automated test đã có; chưa hàm ý full experiment xong.
- `VERIFIED`: đã chạy protocol/evidence yêu cầu trên dữ liệu mục tiêu.

Không dùng từ “hoàn thành” nếu chỉ mới tạo file hoặc smoke test.

## Milestone board

| ID | Workstream | Trạng thái | Evidence hiện có | Việc còn lại |
|---|---|---|---|---|
| M1.1 | Joint BDD100K data contract | VERIFIED | `src/roadsense/data`, `tests/test_self_check.py` | Duy trì khi format data đổi |
| M1.2 | Rasterize/cache lane + drivable | VERIFIED | `rasterize.py`, `tests/test_rasterize.py` | Cross-check thêm official masks khi có |
| M1.3 | Sequence-safe train/dev split | VERIFIED | `split.py`, `tests/test_split.py`, manifest | Theo dõi condition distribution |
| M1.4 | Task metrics foundation | IMPLEMENTED | `src/roadsense/metrics`, `tests/test_metrics.py` | Cross-check official evaluator trước publication |
| M2.1 | Detection operational baseline | IMPLEMENTED | Faster R-CNN config/model/engine | Chạy full training và lưu evidence nếu chưa có artifact |
| M2.2 | Drivable STL baseline | IMPLEMENTED | ResNet18-FPN + CE/Dice | Chạy/ghi metric full experiment |
| M2.3 | Lane STL baseline | IMPLEMENTED | ResNet18-FPN + BCE/Dice | Chạy/ghi metric full experiment |
| M2.4 | Checkpoint/resume/evaluation | IMPLEMENTED | `training/engine.py`, Milestone 2 tests | Thêm integration evidence trên GPU run |
| M3A | Shared MTL baseline | NOT_STARTED | Thiết kế trong project plan | Chờ yêu cầu bắt đầu M3 |
| M3B | Gated adapters | NOT_STARTED | Thiết kế trong project plan | Phụ thuộc M3A |
| M3C | Gradient diagnostics + PCGrad | NOT_STARTED | Thiết kế trong project plan | Phụ thuộc M3A/M3B |
| M4 | Robustness/OOD/ablation | NOT_STARTED | Protocol trong project plan | Phụ thuộc final M3 model |
| M5 | Demo và final report | NOT_STARTED | Deliverables trong project plan | Phụ thuộc M4 |

## Repo inventory hiện tại

### Đã triển khai

- `src/roadsense/data`: config, BDD100K dataset, label normalization, SQLite
  detection index, rasterization, audit và split.
- `src/roadsense/metrics`: detection và segmentation/lane metrics.
- `src/roadsense/models/single_task.py`: Faster R-CNN MobileNetV3-FPN và
  ResNet18-FPN segmenter.
- `src/roadsense/losses/single_task.py`: drivable/lane losses.
- `src/roadsense/training`: config, aligned transforms, DataLoader, train,
  evaluate, checkpoint, resume và smoke tests.
- `configs/experiments`: 3 full baseline configs và 3 CPU smoke configs.

### Cố ý chưa có

- Shared multi-task encoder/FPN và joint batch orchestration.
- FCOS-style capacity-matched detection control.
- Gated adapters.
- Uncertainty weighting/GradNorm/PCGrad.
- Shared-gradient diagnostics.
- Condition-sliced/OOD evaluation và video demo.

Nếu một AI thấy planned file chưa tồn tại, không tự động tạo nó khi milestone đó
chưa được kích hoạt.

## Quyết định đã chốt

| ID | Quyết định | Lý do |
|---|---|---|
| D001 | Dùng letterbox, không stretch ảnh | Giữ hình học đường và box/mask alignment |
| D002 | Main experiment dùng intersection đủ 3 label | Baseline nghiên cứu rõ và ít confound |
| D003 | Giữ official val cho final evaluation | Tránh tuning trên test-like split |
| D004 | Faster R-CNN M2 là operational baseline | Chạy ổn định nhưng chưa capacity-matched với future MTL |
| D005 | M3 bắt đầu bằng ResNet18 | Phù hợp tài nguyên và baseline hiện tại; ResNet50 là ablation/nâng cấp |
| D006 | PCGrad chỉ áp dụng shared encoder/FPN | Private heads/adapters không có gradient conflict liên-task có ý nghĩa |
| D007 | Không triển khai M3 ở thời điểm snapshot | Yêu cầu hiện tại chỉ chuẩn hóa tài liệu |

## Cách AI tiếp tục công việc

1. Đọc `PLAN.md`, file này và milestone document liên quan.
2. Chạy `git status --short`; không ghi đè thay đổi không thuộc task.
3. Xác nhận milestone được người dùng cho phép.
4. Chuyển đúng item sang `IN_PROGRESS`; ghi rõ assumptions nếu có.
5. Chỉ triển khai phần nhỏ nhất đáp ứng exit criteria hiện tại.
6. Chạy test/smoke phù hợp và lưu command + kết quả vào mục Evidence log.
7. Cập nhật board, quyết định mới, rủi ro và next action trong cùng thay đổi.
8. Không đánh dấu `VERIFIED` nếu chưa chạy trên dữ liệu/protocol được yêu cầu.

## Verification commands

```powershell
# M1 synthetic end-to-end
roadsense self-check --output outputs/milestone1_self_check

# Unit tests
pytest

# M2 framework-only smoke
roadsense-train smoke-test --output outputs/milestone2_smoke

# M2 real-data smoke example
roadsense-train train --config configs/experiments/cpu_smoke_drivable.yaml
roadsense-train evaluate --config configs/experiments/cpu_smoke_drivable.yaml `
  --checkpoint outputs/experiments/cpu_smoke_drivable/best.pt
```

Smoke score không được ghi vào bảng kết quả nghiên cứu.

## Evidence log

| Ngày | Phạm vi | Command/evidence | Kết quả |
|---|---|---|---|
| 2026-09-18 | Documentation consolidation | `docs/PLAN.md`, `docs/AI_PROGRESS.md` | Kế hoạch và tracker được tách rõ; M3 vẫn chưa triển khai |

Thêm dòng mới; không sửa lịch sử cũ trừ khi đính chính sai sót và nêu lý do.

## Known risks

- Full baseline artifact có thể chưa nằm trong Git vì checkpoints/outputs không
  được commit; cần kiểm tra máy chạy experiment trước khi tuyên bố metric.
- Local NumPy mAP cần cross-check với official BDD100K evaluator trước báo cáo.
- Lane tolerance F1 phụ thuộc resolution, threshold và tolerance; mọi báo cáo phải
  ghi đủ ba giá trị.
- So sánh STL–MTL không hợp lệ nếu backbone/head/training budget không matched.
- PCGrad tăng memory/compute do phải lấy gradient từng task.
- Vietnam OOD set cần license, tách video theo split và kiểm soát chất lượng nhãn.

## Next authorized actions

Hiện không có task code mới được ủy quyền. Khi có yêu cầu tiếp theo, ưu tiên:

1. Chạy lại toàn bộ test M1–M2.
2. Kiểm kê artifact của ba full STL baselines và ghi evidence thực tế.
3. Chỉ sau khi M2 được nghiệm thu và người dùng cho phép: lập implementation
   checklist cho M3A, chưa gộp adapters/PCGrad vào bước đầu.

## Handoff template

Khi kết thúc một lượt làm việc, thêm hoặc cập nhật:

```text
Active item:
Status:
Files changed:
Tests/evidence:
Open risks:
Exact next action:
```
