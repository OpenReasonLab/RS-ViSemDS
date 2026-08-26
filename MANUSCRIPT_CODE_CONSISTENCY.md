# Manuscript–Code Consistency Audit

Reference manuscript: `2027IEEE顶刊遥感图像分类 (5).pdf`.

## Aligned protocols

| Item | Manuscript protocol | Packaged implementation |
|---|---|---|
| Fixed evaluation split | Seed 42; 100 test images per class | Validator checks class counts, recorded seed, and support/test disjointness |
| AID | 10 classes; 1,000 test; 2,210 support | `configs/aid.json`; `manifests/aid_eval100_seed42/` |
| NWPU-Urban | 8 classes; 800 test; 4,800 support | `configs/nwpu_fg_urban.json`; `manifests/nwpu_eval100_seed42/` |
| Repeated experiments | Arithmetic mean over ten runs | Formal launchers use ten distinct run directories; `aggregate_paper_runs.py` requires exactly ten |
| MLLM inference | Frozen parameters, bfloat16, automatic device mapping, greedy decoding, at most 256 new tokens | Shared local backend and `run_mllm_paper_runs.py` enforce these settings |
| Context isolation | One fresh context per target; no filename, metadata, or external information | Each target builds a new message list and supplies only image content and prompt text |
| Baseline prompts | Figure 8 role, task, label set, demonstrations, target, query, and JSON output contract | Shared prompt builders and public text templates under `prompts/` |
| Invalid output | A unique candidate-label match is accepted; otherwise counted incorrect | Shared parser; formal launcher disables invalid-output regeneration |
| Random few-shot | Total `k` support examples; independently resampled every run | Run-index 0–9 rebuilds examples with seeds 42–51 |
| Visual kNN MLLM | Global top-`k` support images from normalized RemoteCLIP image similarity | `build_examples.py --feature-backend remoteclip`; `run_knn_totalshot_mllm.py` |
| Conventional few-shot | Class-balanced RemoteCLIP retrieval; frozen ImageNet backbone; head-only fitting | `run_all_per_class_fewshot.py` builds one shared per-class RemoteCLIP selection and runs seeds 42–51 |
| Conventional training | 10 epochs, batch 16, Adam, lr 0.001, cross entropy, 224×224 | `run_strict_baselines.py` defaults |
| Full data | One seed-42 support-only 90/10 development split; reinitialize head; train complete support for 10 epochs | `run_full_data_fixed_eval.py`; final runs use seeds 42–51 |
| RS-ViSemDS retrieval | Complete class support, RemoteCLIP, `r=3` per class, global final `k=3`, exclude held-out query from candidates and visual reference | Builder and runner validation enforce these rules |
| RS-ViSemDS fusion | Separately calibrated visual/semantic temperatures; JSD; logit fusion; base prior `(0.6, 0.2, 0.2)` | Formal defaults and run metadata record temperatures, divergence, logits, and weights |
| RS-ViSemDS prompt | Demonstrations as positive evidence; Stage A prototype/relevance; Stage B one conservative boundary check | `manuscript_v1` prompt mode |
| GPT-4o | Official OpenAI API | Official API base is the default; compatible endpoints require explicit override |
| Metrics | Accuracy, macro Precision, macro Recall, macro F1 | All baseline and method summaries expose the same four measures |
| Retrieval timing | Retrieval, selection, prompt construction, and generation are included | RemoteCLIP builder records per-target retrieval time; kNN evaluator adds it to end-to-end time |

## Corrections implemented

1. Added one canonical ten-run MLLM launcher and independent Random resampling.
2. Replaced stale backbone-specific conventional retrieval in the manuscript entry point with class-balanced RemoteCLIP retrieval.
3. Changed conventional and full-data final evaluation from a single run to ten independent seeds and four-metric arithmetic-mean aggregation.
4. Matched the Figure 8 prompt order and output contract; corrected the invalid-output retry path.
5. Added macro Precision and macro Recall to all shared summaries.
6. Included RemoteCLIP retrieval cost in retrieval-based MLLM timing.
7. Made the full-data development split one shared seed-42 support-only procedure, followed by complete-support retraining.
8. Preserved automatic device mapping for local MLLMs and enabled the InternVL tokenizer compatibility option.
9. Corrected public category-description punctuation to the manuscript wording and documented the fixed RS-ViSemDS prior.
10. Kept API credentials environment-only and removed results, caches, logs, checkpoints, and generated selections from the public package.

## Manuscript ambiguities recorded explicitly

- The full-data prose mentions five seeds, while the table and global reporting statement specify ten runs. The formal code follows ten runs (seeds 42–51) and records all per-run results.
- The traditional few-shot equation can be read as using each backbone for retrieval, while the experiment prose explicitly names RemoteCLIP. The canonical manuscript entry follows the experiment prose; the older backbone-specific builder is retained only as non-paper historical/ablation code.
- The manuscript requires separate temperature calibration but does not disclose its optimization objective or subset size. The code uses a support-only class-balanced subset and NLL, and records the subset, bounds, objective, and fitted values. It does not claim these missing choices were specified by the manuscript.
- The category-description generation workflow does not disclose a complete selection objective. The public descriptions are fixed inputs matching the manuscript; no undisclosed generation objective is invented.

The package contains no experiment outputs or numerical result tables.
