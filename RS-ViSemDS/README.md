# RS-ViSemDS

This directory is an isolated implementation plan and runnable scaffold for the paper's
Remote Sensing Visual-Semantic Demonstration Selection framework. It does not modify the
existing zero-shot, random few-shot, global-kNN, Skill Prompt, or traditional visual baselines.

## 1. Fixed experimental contract

Use only the formal manifests below:

- `manifests/aid_eval100_seed42`: 10 classes, 1000 evaluation images, 2210 support images.
- `manifests/nwpu_eval100_seed42`: 8 classes, 800 evaluation images, 4800 support images.

The older 240-image AID and 192-image NWPU manifests and example CSVs are not valid for the
paper's final tables. Test images are used only for final inference. They must never enter the
support pool, category-text development, boundary-rule development, or hyperparameter tuning.

## 2. Code map

- `rs_visemds/category_texts.py`: ten short positive descriptions per class plus separate
  boundary-aware rules. Freeze this file before the final run.
- `rs_visemds/embedding_backend.py`: RemoteCLIP image/text encoding, normalized category
  prototype construction, cache hashes, and manifest-aware cache invalidation.
- `rs_visemds/selector.py`: class-balanced top-r candidate retrieval, three-component scoring,
  candidate-pool min-max normalization, and deterministic final top-k selection.
- `build_rs_visemds_examples.py`: writes selected examples and a complete candidate-score audit.
- `rs_visemds/prompt_builder.py`: paper-aligned sequential multimodal prompt construction.
- `run_rs_visemds_mllm.py`: frozen local MLLM inference, resume protection, parsing, metrics,
  confusion matrices, and timing.
- `run_rs_visemds_all.py`: main AID/NWPU and Gemma/Qwen/InternVL batch runner.

## 3. Algorithm implemented

For each category, encode and L2-normalize ten short descriptions. Average the ten vectors and
normalize the mean to obtain the category prototype. For each target, retrieve the visually
nearest `r=3` support images independently from every class. This creates 30 AID candidates or
24 NWPU-Urban candidates.

For candidate `i` with label `y_i`, compute:

```text
S_img = cos(target_image, candidate_image)
S_typ = cos(candidate_image, category_prototype[y_i])
S_sem = cos(target_image, category_prototype[y_i])
```

Normalize each component independently over the complete `r*C` candidate pool, then compute:

```text
R = alpha*S_img_norm + beta*S_typ_norm + gamma*S_sem_norm
alpha = 0.6, beta = 0.2, gamma = 0.2
```

These `(0.6, 0.2, 0.2)` weights are the paper's main setting and are also the code defaults. The main runner passes them explicitly to the builder. Uniform `(1/3, 1/3, 1/3)` weighting is not the main experiment and must be treated as a separate ablation.

Select the overall top `k=3` candidates. The final list is not forced to be class-balanced and
may contain repeated labels. Demonstrations are ordered by descending `R` in the prompt.

## 4. Build selections

Run from the parent project root:

```bash
python RS-ViSemDS/build_rs_visemds_examples.py \
  --dataset aid \
  --manifest-dir manifests/aid_eval100_seed42 \
  --out-dir RS-ViSemDS/examples/aid_eval100_seed42_a0p6_b0p2_g0p2 \
  --r 3 --k 3 \
  --alpha 0.6 --beta 0.2 --gamma 0.2 \
  --remoteclip-cache checkpoints

python RS-ViSemDS/build_rs_visemds_examples.py \
  --dataset nwpu_fg_urban \
  --manifest-dir manifests/nwpu_eval100_seed42 \
  --out-dir RS-ViSemDS/examples/nwpu_fg_urban_eval100_seed42_a0p6_b0p2_g0p2 \
  --r 3 --k 3 \
  --alpha 0.6 --beta 0.2 --gamma 0.2 \
  --remoteclip-cache checkpoints
```

Each output directory contains:

- `examples_rs_visemds_shot_3.csv`: final three demonstrations per target.
- `candidate_scores.csv`: all `r*C` candidates and every raw/normalized score.
- `selection_config.json`: weights, hashes, checkpoint identity, and cache metadata.

## 5. Run one model

```bash
python RS-ViSemDS/run_rs_visemds_mllm.py \
  --dataset nwpu_fg_urban \
  --manifest-dir manifests/nwpu_eval100_seed42 \
  --selected-examples-csv RS-ViSemDS/examples/nwpu_fg_urban_eval100_seed42_a0p6_b0p2_g0p2/examples_rs_visemds_shot_3.csv \
  --model /root/autodl-tmp/models/Qwen3-VL-8B \
  --out-dir RS-ViSemDS/results_eval100_seed42/qwen3vl_8b_nwpu_fg_urban \
  --torch-dtype bfloat16 --device-map auto --max-tokens 256 \
  --prompt-mode manuscript_v1 --resume
```

The runner reuses the parent project's `TransformersVisionLLM` and prediction parser. This is
intentional: changing the parser only for RS-ViSemDS would make comparisons with existing
zero/random/kNN baselines unfair.

## 6. Run the paper's main suite

```bash
python RS-ViSemDS/run_rs_visemds_all.py \
  --datasets aid nwpu_fg_urban \
  --models gemma3_12b qwen3vl_8b internvl35_14b \
  --alpha 0.6 --beta 0.2 --gamma 0.2 \
  --prompt-mode manuscript_v1
```

Selections are stored in weight-tagged directories such as `a0p6_b0p2_g0p2`, and result paths additionally include the prompt mode. The manuscript main suite explicitly uses `manuscript_v1`; the legacy and fallback modes are auxiliary analyses. Before reusing an existing selection, the runner verifies `selection_config.json`, all three weights, `r`, `k`, the selected CSV hash, and all manifest hashes. A stale uniform-weight selection therefore cannot be silently reused as a paper-main result.

Use `--dry-run` to inspect commands and `--limit 8` for a small inference smoke test. A selection
smoke test can use `build_rs_visemds_examples.py --limit-per-class 1` in a separate output folder.

## 7. Verification

```bash
python -m unittest discover -s RS-ViSemDS/tests -v
python -m py_compile RS-ViSemDS/build_rs_visemds_examples.py \
  RS-ViSemDS/run_rs_visemds_mllm.py RS-ViSemDS/run_rs_visemds_all.py
```

Before final experiments, visually inspect selected examples for several targets from every
class and archive the exact category-text file, RemoteCLIP checkpoint hash, manifests, selection
CSVs, and run configurations.

## 8. Timing interpretation

Support-image and category-prototype embedding construction is a one-time cached preprocessing
stage. Per-target results separately record selection, generation, and combined time. Do not
silently merge one-time cache construction into per-image inference time; state the convention
used in the paper table.

## 9. AutoDL and ablation scripts

The `run_*_autodl.sh` scripts locate the project root from their own file location, so they may
be called from any working directory. Override `PROJECT_ROOT`, `PYTHON_BIN`, model-path variables,
or `REMOTECLIP_CHECKPOINT` when the AutoDL layout differs from the defaults. The default
RemoteCLIP path is `checkpoints/RemoteCLIP-ViT-B-32.pt` under the project root.

The fallback prompt scripts intentionally reuse frozen selections rather than rebuilding them:

- Run `run_aid_three_classes_v3_autodl.sh` before the AID reference-only/fallback scripts.
- Run `run_nwpu_prompt_ablation_602020_autodl.sh` before the NWPU fallback scripts.

These scripts are auxiliary ablations. The paper's main experiment remains
`run_rs_visemds_all.py` with `(alpha, beta, gamma)=(0.6, 0.2, 0.2)` and
`prompt_mode=manuscript_v1`.
