# Huấn luyện Milestone 2 trên Kaggle

Pipeline Kaggle được thiết kế để không sửa config đã commit. Mỗi run tự phát hiện
BDD100K trong `/kaggle/input`, tạo config tuyệt đối trong `/kaggle/working`, chạy
preflight trên một sample thật, training, evaluation và ghi provenance.

## 1. Artifact được tạo

Mỗi task ghi vào `/kaggle/working/roadsense/outputs/<task>_<mode>/`:

```text
best.pt                    checkpoint có validation metric tốt nhất
latest.pt                  checkpoint cuối epoch, dùng để resume
summary.json               tổng kết training
history.jsonl              loss và validation metric theo epoch
evaluation.json            đánh giá checkpoint được chọn
run_manifest.json          commit, GPU, dataset path, command, trạng thái, lỗi
resolved_dataset.yaml      dataset config thực tế
resolved_experiment.yaml   experiment config thực tế
split_manifest.json        snapshot train/dev split
environment.txt            pip freeze của môi trường
```

`run_manifest.json` được ghi từ trước khi training và cập nhật atomically. Nếu
run lỗi bình thường, manifest chuyển sang `failed` và lưu thông báo lỗi cùng các
artifact đã tạo được. Nếu Kaggle hard-timeout, trạng thái có thể còn `running`,
nhưng `latest.pt` vẫn dùng được nếu epoch gần nhất đã hoàn tất.

## 2. Cell cài đặt

Trong Kaggle Notebook, bật GPU accelerator rồi chạy:

```bash
git clone https://github.com/hoangleviet1031/multi-task-road-scene-understanding-for-autonomous-vehicle.git /kaggle/working/roadsense
cd /kaggle/working/roadsense
python -m pip install -q -e ".[train]"
```

Nếu training một commit/tag đã chốt, checkout trước khi cài:

```bash
git checkout <tag-or-commit>
```

Không commit Kaggle token, dataset, cache hoặc checkpoint vào Git.

## 3. Pilot bắt buộc

Chạy từng task để lỗi được cô lập rõ:

```bash
python -m roadsense.kaggle --task drivable --mode pilot
python -m roadsense.kaggle --task lane --mode pilot
python -m roadsense.kaggle --task detection --mode pilot
```

Pilot mặc định dùng 256 train sample, 64 dev sample và 1 epoch. Có thể chạy tuần
tự cả ba task:

```bash
python -m roadsense.kaggle --task all --mode pilot
```

Chỉ chuyển sang full khi cả ba pilot có `run_manifest.json` với
`"status": "succeeded"`.

## 4. Full-dataset fine-tuning

Flag xác nhận là bắt buộc để tránh vô tình bắt đầu job dài:

```bash
python -m roadsense.kaggle --task drivable --mode full --confirm-full --hash-checkpoints
python -m roadsense.kaggle --task lane --mode full --confirm-full --hash-checkpoints
python -m roadsense.kaggle --task detection --mode full --confirm-full --hash-checkpoints
```

Nên chạy mỗi task trong một Kaggle Notebook Version riêng. Cách này giúp mỗi
checkpoint có output, log và trạng thái độc lập; một task lỗi không làm mất hai
task còn lại.

## 5. Resume

Nếu `latest.pt` đã nằm trong output directory hiện tại, runner tự resume. Khi
checkpoint đến từ một Kaggle Dataset đã attach ở `/kaggle/input`, chỉ rõ path:

```bash
python -m roadsense.kaggle \
  --task lane \
  --mode full \
  --confirm-full \
  --resume /kaggle/input/roadsense-lane-checkpoint/latest.pt
```

Runner không cho dùng một `--resume` chung với `--task all` vì checkpoint của ba
kiến trúc không tương thích.

## 6. Auto-discovery và override

Auto-discovery chỉ chấp nhận đúng một cây `images/100k/{train,val}` và đúng một
cặp label JSON. Nếu có nhiều Kaggle Dataset cùng chứa BDD100K, runner dừng với
danh sách candidate thay vì chọn ngẫu nhiên.

Khi cần, truyền path tuyệt đối:

```bash
python -m roadsense.kaggle \
  --task lane --mode pilot \
  --train-images /kaggle/input/<dataset>/bdd100k/images/100k/train \
  --val-images /kaggle/input/<dataset>/bdd100k/images/100k/val \
  --train-labels /kaggle/input/<labels>/bdd100k_labels_images_train.json \
  --val-labels /kaggle/input/<labels>/bdd100k_labels_images_val.json
```

## 7. Pretrained weights và Internet

Baseline chuẩn dùng ImageNet-pretrained backbone. Runner tìm các file weights
Torchvision có đúng filename trong dataset attach và tự copy vào
`/kaggle/working/roadsense/torch_cache`. Nếu không có, Torchvision sẽ thử tải.

Nếu Kaggle Internet bị tắt, attach một private Kaggle Dataset chứa weights. Chỉ
dùng `--no-pretrained` khi chủ ý chạy experiment from-scratch; kết quả đó không
thay thế baseline fine-tuning.

## 8. Các guardrail

- Mặc định yêu cầu CUDA; không âm thầm chạy full trên CPU.
- `--allow-cpu` chỉ dành cho pilot nhỏ.
- Full mode yêu cầu `--confirm-full`.
- AMP tự bật khi CUDA khả dụng; dùng `--no-amp` để chẩn đoán numerical issue.
- `num_workers=0` là mặc định an toàn cho SQLite/cache; chỉ tăng sau khi pilot.
- Không ghi vào `/kaggle/input`; config, cache và output đều ở working directory.
- Nếu dataset discovery mơ hồ hoặc label thiếu/rỗng, preflight dừng trước training.
- Nếu output đã hoàn tất đủ epoch, runner không train lại mà chạy evaluation.

## 9. Tùy chỉnh có kiểm soát

```bash
# Pilot lớn hơn
python -m roadsense.kaggle --task lane --mode pilot \
  --train-limit 1000 --val-limit 200 --epochs 2

# Giảm batch nếu hết VRAM
python -m roadsense.kaggle --task detection --mode full --confirm-full \
  --batch-size 1

# Chỉ chuẩn bị, validate dữ liệu/model và tạo manifest
python -m roadsense.kaggle --task drivable --mode pilot --prepare-only
```

Sau mỗi full run, tải toàn bộ thư mục output về. Khi ghi kết quả vào báo cáo,
dùng `run_manifest.json`, `resolved_experiment.yaml`, `summary.json` và
`evaluation.json`; không suy luận config chỉ từ tên checkpoint.
