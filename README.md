# RS-ViSemDS: Visual-Semantic Demonstration Selection for Remote Sensing Scene Classification with Open-Weight Multimodal Large Language Models

<p align="center">
  <a href="assets/fig02_datasets_models_baseline_protocols.pdf">📄 Figure 1: Datasets, Models, and Baseline Protocols</a>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="assets/fig03_rs_visemds_framework.pdf">🧩 Figure 2: RS-ViSemDS Framework</a>
</p>

This repository provides the experimental reproduction code for the paper **RS-ViSemDS**. It follows the final evaluation protocol used in the manuscript, with **100 fixed test images per class and random seed 42**.

This package contains only source code, configuration files, fixed data-split manifests, tests, execution scripts, and the two protocol figures linked above. It does **not** include the original images, model weights, RemoteCLIP caches, pre-generated demonstrations, predictions, metric files, confusion matrices, experimental plots, logs, or checkpoints. All retrieval files required for the experiments are generated locally by the code.

## 🧭 1. Experimental Scope

### MLLM Baselines

- GPT-4o
- Llama-3.2-11B-Vision-Instruct
- Gemma-3-12B
- Qwen2.5-VL-7B-Instruct
- Qwen3-VL-8B
- InternVL3.5-8B
- InternVL3.5-14B

The following settings are supported:

- Zero-shot
- Random few-shot
- Global RemoteCLIP visual-kNN few-shot

### Conventional Visual Baselines

- ResNet-18
- ResNet-50
- ViT-Tiny
- ViT-Small

The repository supports target-specific, class-wise kNN few-shot learning and full-data head-only transfer learning. For conventional few-shot learning, each model uses its own frozen ImageNet-pretrained backbone to extract retrieval features. RemoteCLIP is not used, and the training images are not randomly sampled from each class.

### RS-ViSemDS

The main experiments are conducted with the following three open-weight MLLMs:

- Gemma-3-12B
- Qwen3-VL-8B
- InternVL3.5-14B

## 🗂️ 2. Repository Structure

```text
configs/                         Dataset and category configurations
assets/                          Paper Figures 1 and 2 in PDF format
data_raw/                        Raw dataset directory; empty in the release package
examples/                        Runtime-generated Random/kNN demonstration files
manifests/                       Fixed seed-42 data splits used in the paper
strict_fewshot/                  Shared components for data, models, metrics, features, and MLLMs
RS-ViSemDS/                      RS-ViSemDS selection, prompting, inference, and test code
build_examples.py                MLLM Random/global-kNN demonstration generator
build_backbone_knn_examples.py   Per-class backbone-kNN generator for conventional visual models
prepare_eval100_protocol.py      Unified preparation entry point for MLLM demonstrations
run_*_aid_nwpu_all.py            Unified AID/NWPU entry points for individual MLLMs
run_all_per_class_fewshot.py     Unified entry point for the four conventional few-shot models
run_full_data_fixed_eval_all.py  Unified entry point for the four full-data models
verify_package.py                Integrity checker for data splits and the code-only release package
```

A detailed manuscript–code consistency audit is provided in `MANUSCRIPT_CODE_CONSISTENCY.md`. Random seeds and dataset sources are documented in `SEEDS_AND_DATASETS.md`, and the GitHub upload scope and excluded files are listed in `GITHUB_UPLOAD_CHECKLIST.md`.

## 🔒 3. Fixed Evaluation Protocol

| Dataset | Classes | Test images per class | Test total | Support pool |
|---|---:|---:|---:|---:|
| AID | 10 | 100 | 1000 | 2210 |
| NWPU-Urban | 8 | 100 | 800 | 4800 |

- The fixed random seed is `42`.
- AID manifest: `manifests/aid_eval100_seed42/`.
- NWPU-Urban manifest: `manifests/nwpu_eval100_seed42/`.
- The test set and support pool are strictly disjoint.
- Test images are never used as retrieval candidates, training or validation samples, or for hyperparameter tuning and model selection. They are used only as target queries for retrieval and final evaluation.
- The earlier protocol with 24 test images per class and its associated results are not included in this package.

> ⚠️ Do not regenerate the manifests with a new random split; otherwise, the resulting test sets will no longer match those used in the paper.

## 🧰 4. Data and Environment Preparation

Dataset download links are provided in `SEEDS_AND_DATASETS.md`. After downloading and extracting the datasets, place the class directories under:

```text
data_raw/AID_dataset/
data_raw/NWPU-RESISC45/
```

The class-directory names must match the `source_directory` entries in `configs/*.json` and the manifests.

Install the shared dependencies:

```bash
python -m pip install -r requirements.txt
```

For locally deployed MLLMs, install compatible versions of `transformers`, `accelerate`, and the corresponding visual dependencies according to the official model requirements. FlashAttention2 is optional. If it is unavailable, the models can fall back to standard attention, although inference speed and GPU memory usage may differ.

## 🔎 5. Generating MLLM Few-Shot Demonstrations

For MLLM Random and visual-kNN prompting, `k` denotes the **total number of demonstrations used for each target image**, not the number of demonstrations per class.

- Random: for each target image, a total of `k` images are sampled from the complete support pool using seed 42.
- Visual kNN: a frozen RemoteCLIP image encoder retrieves the global top-`k` images from the complete support pool.

After preparing the datasets and RemoteCLIP weights, run:

```bash
python prepare_eval100_protocol.py \
  --skip-manifests \
  --shots 1 3 5 10 \
  --strategies random knn \
  --remoteclip-checkpoint /path/to/RemoteCLIP-ViT-B-32.pt
```

The `--skip-manifests` option preserves the fixed data splits provided with the package. The generated CSV files and feature caches are runtime artifacts and are not included in the release package.

## 🤖 6. Running the MLLM Baselines

All locally deployed open-weight models use frozen parameters, `bfloat16`, automatic device mapping, greedy decoding, `do_sample=False`, and `max_new_tokens=256`. Each target image is evaluated with an independent context, and no filename, ground-truth label, or image metadata is provided as input.

Example for InternVL3.5-14B:

```bash
python run_internvl35_14b_aid_nwpu_all.py \
  --model /path/to/InternVL3.5-14B \
  --datasets aid nwpu_fg_urban \
  --shots 1 3 5 10
```

Entry points for the other open-weight models:

```text
run_llama32_11b_aid_nwpu_all.py
run_gemma3_12b_aid_nwpu_all.py
run_qwen25vl_7b_aid_nwpu_all.py
run_qwen3vl_8b_aid_nwpu_all.py
run_internvl35_8b_aid_nwpu_all.py
```

GPT-4o uses the official OpenAI API by default:

```bash
export OPENAI_API_KEY="your-key"
python run_gpt4o_aid_nwpu_all.py \
  --model gpt-4o \
  --datasets aid nwpu_fg_urban \
  --shots 3 5 10
```

Zero-shot, Random few-shot, and visual-kNN few-shot prompting do not use category descriptions or boundary-aware category rules. Outputs that cannot be parsed into one of the candidate labels are not retried and are counted as incorrect predictions.

The MLLM entry points support `--limit 5` for small-scale checks and `--dry-run` for previewing commands. Do not set `--limit` in formal experiments.

A unified AutoDL entry point is also provided. For example:

```bash
MODEL_ALIAS=internvl35_14b \
MLLM_MODEL_PATH=/root/autodl-tmp/models/InternVL3.5-14B \
PREPARE_EXAMPLES=1 \
REMOTECLIP_CHECKPOINT=/path/to/RemoteCLIP-ViT-B-32.pt \
bash ./run_open_mllm_eval100_autodl.sh
```

## 🖼️ 7. Running the Conventional Few-Shot Baselines

For each target image and each visual model, the corresponding frozen ImageNet-pretrained backbone is used to retrieve the `k_cls` most similar support images independently from every class. The resulting target-specific training set contains `k_cls × C` labeled images.

The training configuration is as follows: only the newly initialized classification head is optimized for 10 epochs using a batch size of 16, Adam with a learning rate of 0.001, cross-entropy loss, an input size of 224×224, and random seed 42.

On the first run, generate model-specific retrieval files and execute the experiments:

```bash
python run_all_per_class_fewshot.py \
  --datasets aid nwpu \
  --shots 1 3 5 10 \
  --models resnet18 resnet50 vit_tiny vit_small
```

To generate only the model-specific, class-wise kNN files:

```bash
python run_all_per_class_fewshot.py \
  --datasets aid nwpu \
  --models resnet18 resnet50 vit_tiny vit_small \
  --examples-only
```

Use `--skip-examples` only when retrieval files for the same manifest, model, and shot configuration have already been generated locally.

## ⚙️ 8. Running the Full-Data Baselines

The full-data experiments use seed 42. Each ImageNet-pretrained backbone remains frozen, and only a newly initialized classification head is trained:

1. Split the support pool into 90% internal training data and 10% validation data to determine the training configuration.
2. Reinitialize the classification head and train it on the complete support pool for a fixed 10 epochs.
3. Evaluate only the epoch-10 model once on the fixed test set.

```bash
python run_full_data_fixed_eval_all.py
```

The test set is never used for internal validation, learning-rate adjustment, or model selection. More detailed implementation parameters and any differences from the manuscript description are documented in `MANUSCRIPT_CODE_CONSISTENCY.md`.

## ✨ 9. Running RS-ViSemDS

The main experiments use the following fixed configuration:

- RemoteCLIP image and text encoders.
- Number of candidates per class: `r=3`.
- Total number of final demonstrations: `k=3`.
- Scoring weights: `(alpha, beta, gamma)=(0.6, 0.2, 0.2)`.
- Two-stage prompt mode used in the paper appendix: `manuscript_v1`.

```bash
python RS-ViSemDS/run_rs_visemds_all.py \
  --datasets aid nwpu_fg_urban \
  --models gemma3_12b qwen3vl_8b internvl35_14b \
  --model-path gemma3_12b=/path/to/gemma-3-12b-it \
  --model-path qwen3vl_8b=/path/to/Qwen3-VL-8B \
  --model-path internvl35_14b=/path/to/InternVL3.5-14B \
  --r 3 --k 3 \
  --alpha 0.6 --beta 0.2 --gamma 0.2 \
  --prompt-mode manuscript_v1 \
  --remoteclip-checkpoint /path/to/RemoteCLIP-ViT-B-32.pt
```

See `RS-ViSemDS/README.md` for details on the algorithm, scoring procedure, category text, prompt construction, runtime measurement, and single-model execution. Other prompt modes and weight configurations are intended only for auxiliary analysis or ablation studies and should not be mixed with the main experimental setting reported in the paper.

## ✅ 10. Verifying the Package

Before adding the raw datasets:

```bash
python verify_package.py
python -m unittest discover -s tests -v
python -m unittest discover -s RS-ViSemDS/tests -v
```

After adding the raw datasets:

```bash
python verify_package.py --allow-data
```

The verifier checks the number of fixed test images per class, support/test leakage, required entry points, file hashes, and whether the package accidentally contains experimental results, logs, caches, model weights, archives, or pre-generated demonstrations.
