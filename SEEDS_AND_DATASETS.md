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
| MLLM random demonstrations | 42 | Samples support examples for 1/3/5/10-shot Random prompts. |
| MLLM RemoteCLIP kNN | deterministic | Ranking is derived from the fixed support pool and frozen RemoteCLIP embeddings. |
| MLLM generation | deterministic decoding | Temperature is 0.0; local Transformers generation uses `do_sample=False`. |
| Metrics bootstrap | 42 | Uses 10,000 bootstrap samples for the accuracy confidence interval. |
| Traditional few-shot | 42 | Base training seed; the runner derives stable model/target-specific seeds from it. |
| Full-data baselines | 42, 43, 44 | Three independent runs; maximum/final training epoch is 10. |
| RS-ViSemDS selection | deterministic | Uses the seed-42 manifest and deterministic RemoteCLIP visual-semantic ranking. |
| RS-ViSemDS metrics bootstrap | 42 | Uses 10,000 bootstrap samples. |

## Fixed split sizes

| Dataset subset | Classes | Evaluation per class | Evaluation total | Support total |
|---|---:|---:|---:|---:|
| AID-10 | 10 | 100 | 1000 | 2210 |
| NWPU-Urban-8 | 8 | 100 | 800 | 4800 |

The exact file paths and class-wise counts are recorded in each manifest's `evaluation.csv`, `support.csv`, `class_order.json`, and `summary.json`.

