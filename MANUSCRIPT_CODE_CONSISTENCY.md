# Manuscript-Code Consistency Audit

Reference manuscript: `RS-ViSemDS_Manuscript.pdf`, modified 2026-07-20.

## Aligned Protocols

| Item | Manuscript protocol | Packaged code |
|---|---|---|
| Test split | 100 images per class, seed 42 | Fixed seed-42 manifests; validator checks class counts and support/test disjointness |
| AID | 10 classes, 1000 test, 2210 support | `configs/aid.json` and `manifests/aid_eval100_seed42/` |
| NWPU-Urban | 8 classes, 800 test, 4800 support | `configs/nwpu_fg_urban.json` and `manifests/nwpu_eval100_seed42/` |
| MLLM inference | Frozen parameters, bfloat16, automatic device mapping, greedy decoding, 256 new tokens | Shared Transformers backend and all open-model launchers |
| Context isolation | One fresh context per target; no filename or metadata | Each target builds a new message list; images are decoded and metadata is removed for API calls |
| Random few-shot | Total k examples sampled from support | `build_examples.py` plus `run_random_fewshot_mllm.py` |
| Visual kNN MLLM | Global top-k support images from normalized RemoteCLIP image similarity | `build_examples.py --feature-backend remoteclip` plus `run_knn_totalshot_mllm.py` |
| Conventional few-shot | Per-class kNN with the corresponding frozen ImageNet backbone | `build_backbone_knn_examples.py`; one model-specific selection directory per backbone |
| Conventional training | Head only, 10 epochs, batch 16, Adam, lr 0.001, cross entropy, 224x224, seed 42 | `run_strict_baselines.py` and `run_all_per_class_fewshot.py` defaults |
| Full data | Seed 42; 90/10 support validation; reinitialize head; full-support retraining for 10 epochs; final test once | `run_full_data_fixed_eval.py` and `run_full_data_fixed_eval_all.py` default to seed 42 and enforce 10 epochs |
| RS-ViSemDS retrieval | RemoteCLIP, r=3 per class, final total k=3 | `build_rs_visemds_examples.py` and main runner defaults |
| RS-ViSemDS weights | alpha=0.6, beta=0.2, gamma=0.2 | Builder defaults and main runner explicit arguments |
| RS-ViSemDS prompt | Demonstrations as positive evidence; Stage A P/R; Stage B one conservative boundary check | `manuscript_v1`, explicitly passed by the main runner |
| GPT-4o | Official OpenAI API | Official API base is the default; compatible endpoints require an explicit override |
| Invalid MLLM output | Counted as incorrect | Main evaluator defaults use zero invalid-output retries |
| Metrics | Accuracy and macro precision, recall, F1 | Shared metric summaries use macro averaging |
| Timing | Full-data inference uses batch 1 end-to-end; retrieval methods include retrieval/selection/prompt/generation | Timing fields and run metadata encode these scopes |

## Corrections Made During This Audit

1. Replaced the incorrect traditional RemoteCLIP per-class retrieval path with model-specific ImageNet-backbone retrieval.
2. Added the exact manuscript-oriented RS-ViSemDS prompt mode and made it the explicit main-suite default.
3. Changed full-data default seeds from three seeds to manuscript seed 42.
4. Changed GPT-4o's default from a third-party compatible endpoint to the official OpenAI endpoint.
5. Removed all precomputed traditional retrieval CSVs and made generated-example directories output-only.
6. Strengthened package validation to reject results, logs, caches, weights, archives, and generated example files.

## Disclosure Note

The manuscript states the full-data split, seed, epoch count, and initial learning rate, but does not enumerate every implementation detail of that protocol. The code additionally records its batch size, AdamW optimizer, weight decay, augmentation, and validation-driven learning-rate schedule in each run configuration. These are implementation details, not hidden packaged results. For exact reproducibility, they should either remain frozen in code or be added to the manuscript/supplementary material.

This package contains no experiment outputs or numerical performance tables.
