# Experiment records

Checkpoint lớn không được commit vào Git. Mỗi full training run cần một bản ghi
Markdown hoặc JSON nhỏ trong thư mục này, trích từ Kaggle `run_manifest.json`.

Tên đề xuất:

```text
m2_<task>_<YYYYMMDD>_<short-commit>.md
```

Nội dung tối thiểu:

```text
Task và mục tiêu:
Git commit/tag:
Kaggle Notebook URL/version:
Kaggle Dataset slug/version:
Config snapshot:
Seed:
GPU và software versions:
Start/end time:
Resume lineage:
Best epoch/checkpoint:
Validation metrics:
Checkpoint SHA256:
Known warnings/failures:
```

Smoke/pilot metrics chỉ chứng minh pipeline hoạt động, không đưa vào bảng kết quả
nghiên cứu cuối.
