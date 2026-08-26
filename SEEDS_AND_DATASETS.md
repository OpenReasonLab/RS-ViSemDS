# Random Seeds and Dataset Sources

## Dataset sources

### AID

- Official project page: https://captain-whu.github.io/AID/
- Official OneDrive download: https://1drv.ms/u/s!AthY3vMZmuxChNR0Co7QHpJ56M-SvQ
- Dataset: 10,000 images, 30 scene classes.
- This project evaluates the 10 classes listed in `configs/aid.json`.
- Required local path: `data_raw/AID_dataset/`.

### NWPU-RESISC45

- Author dataset page: https://gcheng-nwpu.github.io/#Datasets
- Author-provided OneDrive download: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
- Dataset: 31,500 images, 45 classes, 700 images per class.
- This project evaluates the 8 fine-grained urban classes listed in `configs/nwpu_fg_urban.json`.
- Required local path: `data_raw/NWPU-RESISC45/`.

Observe the licenses and citation requirements shown on the official pages.

## Seed registry

| Component | Seed(s) | Meaning |
|---|---:|---|
| Fixed eval100 manifest | 42 | Selects exactly 100 evaluation images per class. |
| MLLM random demonstrations | 42–51 | Independently resamples support examples in each of the ten formal runs. |
| MLLM RemoteCLIP kNN | deterministic | Ranking is derived from the fixed support pool and frozen RemoteCLIP embeddings. |
| MLLM generation | deterministic decoding | Temperature is 0.0; local Transformers generation uses `do_sample=False`. |
| Metrics bootstrap | 42 | Uses 10,000 bootstrap samples for the accuracy confidence interval. |
| Traditional few-shot | 42–51 | Ten independent head-training runs; RemoteCLIP retrieval is fixed for a target/manifest. |
| Full-data development | 42 | One support-only 90/10 split used to fix the training schedule. |
| Full-data final runs | 42–51 | Ten independent full-support head initializations; final epoch is 10. |
| RS-ViSemDS selection | deterministic per fixed input | Uses the seed-42 manifest and deterministic RemoteCLIP visual-semantic ranking. |
| RS-ViSemDS metrics bootstrap | 42 | Uses 10,000 bootstrap samples. |

## Fixed split sizes

| Dataset subset | Classes | Evaluation per class | Evaluation total | Support total |
|---|---:|---:|---:|---:|
| AID-10 | 10 | 100 | 1000 | 2210 |
| NWPU-Urban-8 | 8 | 100 | 800 | 4800 |

The exact file paths and class-wise counts are recorded in each manifest's `evaluation.csv`, `support.csv`, `class_order.json`, and `summary.json`.

For RemoteCLIP kNN reproduction, the seed alone is not sufficient: use the same `RemoteCLIP-ViT-B-32.pt` checkpoint and record its SHA-256. Different model weights or library kernels can change near-tie ordering even when the manifest is identical.
