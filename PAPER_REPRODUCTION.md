# Manuscript reproduction entry points

Only the commands in this file are eligible for the manuscript tables. They use
the fixed seed-42 evaluation manifests:

- `manifests/aid_eval100_seed42`: 1,000 test and 2,210 support images.
- `manifests/nwpu_eval100_seed42`: 800 test and 4,800 support images.

The older 24-image-per-class manifests and `run_full_data_baseline.py` are
historical experiments, not manuscript protocols.

## MLLM zero-shot, random few-shot, and visual kNN

`run_mllm_paper_runs.py` enforces ten runs, greedy decoding, 256 output tokens,
the Appendix-B prompt roles, no invalid-output regeneration, and arithmetic-mean
aggregation. Random demonstrations are independently resampled using seeds
42 through 51. Visual kNN uses global RemoteCLIP retrieval. Set `--model` to the
exact local checkpoint path and choose a stable output alias.

```bash
python run_mllm_paper_runs.py \
  --model /path/to/Qwen3-VL-8B-Instruct \
  --model-alias qwen3vl_8b \
  --backend transformers \
  --datasets aid nwpu_fg_urban \
  --methods zero random knn \
  --shots 3 \
  --runs 10 \
  --resume
```

Run the command separately for every open-weight checkpoint listed in the
paper. For GPT-4o, use `--backend api`, the exact `gpt-4o` model identifier, and
the official `OPENAI_API_KEY` / `https://api.openai.com/v1` endpoint.

## RS-ViSemDS

The formal method suite runs Gemma-3-12B, Qwen3-VL-8B, and InternVL3.5-14B on
both fixed datasets for ten runs and writes a ten-run arithmetic mean for every
model/dataset pair.

```bash
python RS-ViSemDS/run_rs_visemds_all.py --runs 10
```

The reported base prior is fixed to `(0.6, 0.2, 0.2)`. Temperatures are fitted
separately on a class-balanced support-only calibration subset. The paper does
not disclose the temperature objective or calibration-subset size; the code
records its explicit NLL objective, subset size, search bounds, and selected
support indices in every run configuration.

## Conventional few-shot visual baselines

The experiment prose states that class-balanced examples are retrieved with
RemoteCLIP and then used to fit the classification head of each frozen
ImageNet-pretrained backbone. This is the canonical implementation used here.

```bash
python run_all_per_class_fewshot.py \
  --datasets aid nwpu --shots 1 3 5 10 \
  --seeds 42 43 44 45 46 47 48 49 50 51 --resume
```

The same class-balanced RemoteCLIP selection is shared by the four frozen
ImageNet backbones. Each head is trained independently for all ten seeds, and
the launcher writes a separate four-metric arithmetic mean for every
dataset/shot/backbone combination.

## Full-data head-only references

Support-only development uses one seed-42 90/10 split. After configuration is
fixed, each head is reinitialized and trained on the complete support pool for
10 epochs using seeds 42 through 51. The fixed test set is never used for
selection.

```bash
python run_full_data_fixed_eval_all.py --seeds 42 43 44 45 46 47 48 49 50 51
```

## Verification

```bash
python -B -m unittest discover -s RS-ViSemDS/tests -v
python -B -m unittest discover -s Eval100_Reproducibility_Package/tests -v
python -B Eval100_Reproducibility_Package/verify_package.py
```

All four reported performance measures are macro/overall values computed from
the same fixed test predictions: accuracy, macro precision, macro recall, and
macro F1. Retrieval-method timing includes retrieval, demonstration selection,
prompt construction, and MLLM generation.
