# RS-ViSemDS

**Visual-Semantic Demonstration Selection for Remote Sensing Scene Classification with Open-Weight Multimodal Large Language Models**

This repository contains the data preparation, prompting, retrieval, evaluation, and visual-baseline code used for **RS-ViSemDS**, a training-free framework for remote-sensing scene classification with frozen multimodal large language models (MLLMs).

RS-ViSemDS treats task adaptation as a **query-adaptive demonstration-selection problem**. For each target image, it constructs a class-balanced candidate pool and ranks candidate demonstrations by jointly considering:

1. target-candidate visual relevance;
2. candidate-category typicality; and
3. target-category semantic affinity.

The selected demonstrations are inserted into a structured multimodal prompt, while **boundary-aware category rules** are used only during final MLLM inference to resolve ambiguous category pairs. Both the retrieval encoder and the MLLM remain frozen.

## Highlights

- Controlled comparison of zero-shot, random few-shot, and visual kNN prompting.
- Query-adaptive visual-semantic demonstration selection.
- Class-balanced candidate generation with global top-\(k\) reranking.
- RemoteCLIP-based image-image and image-text similarity.
- Separate category-description ensembles for retrieval and boundary-aware rules for inference.
- Evaluation with multiple open-weight MLLMs, GPT-4o, and ImageNet-pretrained ResNet/ViT baselines.
- No task-specific parameter updating for RS-ViSemDS.

## Main Results

The paper reports the following best RS-ViSemDS results:

| Dataset | Best backbone | Accuracy | Macro F1 |
|---|---|---:|---:|
| Selected AID subset | InternVL3.5-14B | 99.20% | 99.20% |
| NWPU-Urban | InternVL3.5-14B | 98.00% | 98.01% |

These results are obtained with frozen retrieval and MLLM backbones, a total prompt budget of three demonstrations, and the main scoring weights \((\alpha,\beta,\gamma)=(0.6,0.2,0.2)\).

---

## Experimental Protocol

### Fixed test sets

All methods use the same fixed, class-balanced test sets.

- **AID subset**
  - 10 selected classes
  - 3,210 images in total
  - 100 test images per class
  - 1,000 test images
  - 2,210 support images

- **NWPU-Urban**
  - 8 classes selected from NWPU-RESISC45
  - 700 images per class
  - 5,600 images in total
  - 100 test images per class
  - 800 test images
  - 4,800 support images

The test images are excluded from:

- demonstration retrieval;
- prompt construction;
- conventional-model training;
- validation;
- hyperparameter adjustment;
- model selection; and
- category-text or boundary-rule construction.

They are used only for final evaluation.

### Meaning of “shot”

The repository uses two different shot definitions.

#### MLLM prompting

For zero-shot, random few-shot, visual kNN, and RS-ViSemDS:

> `k-shot` means **k demonstrations in total for each target image**.

For example, a 3-shot prompt contains exactly three labeled support images, regardless of the number of classes.

#### Conventional visual few-shot baselines

For ResNet and ViT few-shot training:

> `k_cls-shot` means **k_cls labeled training images per class**.

Therefore:

- AID uses \(10 \times k_{\mathrm{cls}}\) training images.
- NWPU-Urban uses \(8 \times k_{\mathrm{cls}}\) training images.

The two protocols must not be conflated.

---

## Evaluated Methods

### 1. Zero-shot MLLM prompting

The model receives:

- a task instruction;
- the candidate label set;
- the target image; and
- the target query.

No demonstrations, category-description ensembles, or boundary-aware rules are used.

### 2. Random few-shot prompting

For each target image, \(k\) labeled demonstrations are sampled from the support pool without target-specific retrieval.

The formal experiments use total-shot budgets of:

```text
1, 3, 5, and 10 demonstrations
```

Random prompting is treated as a control condition. The paper shows that its gains are limited and unstable, and that more demonstrations may introduce irrelevant or distracting context.

### 3. Visual kNN few-shot prompting

RemoteCLIP is used as the frozen visual retrieval encoder.

For each target image:

1. compute its normalized image embedding;
2. compute normalized support-image embeddings;
3. rank all support images by cosine similarity; and
4. retrieve the global top-\(k\) images.

This baseline uses only target-support visual similarity. It does not use category descriptions or boundary-aware category rules.

### 4. RS-ViSemDS

RS-ViSemDS consists of three stages.

#### Stage 1: Shared visual-semantic representation

Let \(f_I\) and \(f_T\) denote the frozen RemoteCLIP image and text encoders.

For each support image and target image, normalized image embeddings are extracted. For each category, ten concise positive descriptions are encoded and aggregated into one normalized category-text prototype.

The category descriptions summarize:

- representative objects;
- spatial structures;
- scene composition; and
- land-use semantics.

They contain no test-image-specific information.

#### Stage 2: Class-balanced candidate generation

For every target image, retrieve the top-\(r\) visually similar support images from each category.

The main experiments use:

```text
r = 3 candidates per class
```

This produces:

- 30 candidates for the 10-class AID subset;
- 24 candidates for the 8-class NWPU-Urban subset.

Class balancing is applied only during candidate generation. The final top-\(k\) demonstration list is selected globally and is not required to contain one example from each class.

#### Stage 3: Visual-semantic reranking

For each candidate \(i\), compute:

\[
S_{\mathrm{img}}(i)=\operatorname{sim}(v^*,v_i)
\]

\[
S_{\mathrm{typ}}(i)=\operatorname{sim}(v_i,t_{y_i})
\]

\[
S_{\mathrm{sem}}(i)=\operatorname{sim}(v^*,t_{y_i})
\]

where:

- \(S_{\mathrm{img}}\) measures target-candidate visual relevance;
- \(S_{\mathrm{typ}}\) measures candidate-category typicality;
- \(S_{\mathrm{sem}}\) measures target-category semantic affinity.

Each component is min-max normalized over the candidate pool. The final score is:

\[
R_i=\alpha S_{\mathrm{img}}+\beta S_{\mathrm{typ}}+\gamma S_{\mathrm{sem}}
\]

The main configuration is:

```text
alpha = 0.6
beta  = 0.2
gamma = 0.2
```

The final prompt contains the globally highest-scoring:

```text
k = 3 demonstrations
```

#### Boundary-aware inference

Boundary-aware category rules are used only in the final MLLM prompt. They are not used to construct category prototypes and are not part of the zero-shot, random, or visual kNN baselines.

The prompt first asks the MLLM to compare the target with the selected demonstrations as complete scenes. Only when the leading two labels remain ambiguous does the model apply one conservative pairwise exclusion check.

The rules are designed to resolve recurring confusions such as:

- BareLand vs. Desert;
- Center vs. Church vs. Commercial;
- Dense vs. Medium vs. Sparse Residential;
- Commercial Area vs. Industrial Area; and
- Railway Station vs. other built-up categories.

---

## Evaluated Models

### Open-weight MLLMs

- Llama-3.2-11B-Vision-Instruct
- Gemma-3-12B
- Qwen2.5-VL-7B-Instruct
- Qwen3-VL-8B-Instruct
- InternVL3.5-8B
- InternVL3.5-14B

### Closed-source reference

- GPT-4o

### Conventional visual references

- ResNet-18
- ResNet-50
- ViT-Tiny
- ViT-Small

The ResNet and ViT backbones are initialized from ImageNet-pretrained weights and frozen. Only the newly initialized classification head is trained.

---

## Repository Layout

The exact layout may vary slightly across branches. The paper-aligned repository contains the following functional components:

```text
strict_fewshot_baselines/
  README.md
  requirements.txt

  configs/
    aid.json
    nwpu_fg_urban.json

  manifests/
    aid/
    nwpu_fg_urban/

  examples/
    aid_random/
    aid_knn/
    nwpu_random/
    nwpu_knn/

  category_texts/
    aid_descriptions.json
    aid_boundary_rules.json
    nwpu_urban_descriptions.json
    nwpu_urban_boundary_rules.json

  prompts/
    zero_shot.txt
    random_fewshot.txt
    knn_fewshot.txt
    rs_visemds.txt

  strict_fewshot/
    __init__.py
    data.py
    features.py
    metrics.py
    models.py
    utils.py

  prepare_manifest.py
  build_examples.py
  run_zero_shot_mllm.py
  run_fewshot_mllm.py
  run_knn_fewshot_mllm.py
  run_strict_baselines.py
  run_full_data_baseline.py

  # RS-ViSemDS entry point in the paper-aligned branch
  run_rs_visemds.py
```

Do not hard-code absolute dataset or checkpoint paths. All paths should be passed relative to the repository root or through command-line arguments.

---

## Environment

Install the dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first ResNet, ViT, or RemoteCLIP run may download pretrained weights. Run the command with network access or provide a local checkpoint explicitly.

Example:

```powershell
--remoteclip-checkpoint checkpoints/RemoteCLIP-ViT-B-32.pt
```

---

## Step 1: Prepare the Fixed Test and Support Manifests

Run from the repository root.

### AID

```bash
python prepare_manifest.py \
  --config configs/aid.json \
  --data-root data_raw/AID \
  --out-dir manifests/aid \
  --eval-per-class 100 \
  --seed 42
```

Expected split:

```text
test:    1,000 images
support: 2,210 images
```

### NWPU-Urban

```bash
python prepare_manifest.py \
  --config configs/nwpu_fg_urban.json \
  --data-root data_raw/NWPU-RESISC45 \
  --out-dir manifests/nwpu_fg_urban \
  --eval-per-class 100 \
  --seed 42
```

Expected split:

```text
test:      800 images
support: 4,800 images
```

The command writes:

```text
evaluation.csv
support.csv
class_order.json
summary.json
```

The same manifests must be reused by every formal experiment.

---

## Step 2: Build Random Few-Shot Demonstrations

### AID

```bash
python build_examples.py \
  --manifest-dir manifests/aid \
  --strategy random \
  --shots 1 3 5 10 \
  --out-dir examples/aid_random \
  --seed 42
```

### NWPU-Urban

```bash
python build_examples.py \
  --manifest-dir manifests/nwpu_fg_urban \
  --strategy random \
  --shots 1 3 5 10 \
  --out-dir examples/nwpu_random \
  --seed 42
```

For every target image, the random baseline samples \(k\) examples in total from the support pool.

When nested shot sets are enabled, the 1-, 3-, 5-, and 10-shot sets should be prefixes of one fixed ordered sample list so that:

```text
1-shot ⊂ 3-shot ⊂ 5-shot ⊂ 10-shot
```

---

## Step 3: Build Visual kNN Demonstrations

RemoteCLIP must be used for formal results.

### AID

```bash
python build_examples.py \
  --manifest-dir manifests/aid \
  --strategy knn \
  --knn-scope global \
  --shots 1 3 5 10 \
  --out-dir examples/aid_knn \
  --feature-backend remoteclip \
  --remoteclip-cache checkpoints
```

### NWPU-Urban

```bash
python build_examples.py \
  --manifest-dir manifests/nwpu_fg_urban \
  --strategy knn \
  --knn-scope global \
  --shots 1 3 5 10 \
  --out-dir examples/nwpu_knn \
  --feature-backend remoteclip \
  --remoteclip-cache checkpoints
```

`image_stats` may be used only for smoke tests. Results obtained with `image_stats` must not be reported as paper results.

---

## Step 4: Run Zero-Shot MLLM Evaluation

Example:

```bash
python run_zero_shot_mllm.py \
  --manifest-dir manifests/aid \
  --model llama-3.2-11b-vision-instruct \
  --api-base https://your-api-host/v1 \
  --api-key YOUR_API_KEY \
  --out-dir results/aid_zero_shot_llama11b
```

The zero-shot prompt must not include:

- labeled demonstrations;
- category-description ensembles; or
- boundary-aware category rules.

Typical outputs are:

```text
predictions.csv
summary.json
per_class_accuracy.csv
confusion_matrix.csv
run_config.json
```

---

## Step 5: Run Random Few-Shot MLLM Evaluation

Example for AID 5-shot:

```bash
python run_fewshot_mllm.py \
  --manifest-dir manifests/aid \
  --examples-csv examples/aid_random/examples_random_shot_5.csv \
  --model llama-3.2-11b-vision-instruct \
  --api-base https://your-api-host/v1 \
  --api-key YOUR_API_KEY \
  --out-dir results/aid_random_shot_5_llama11b
```

The prompt contains the selected labeled examples followed by the target image.

Use `--limit 5` for a short smoke test.

---

## Step 6: Run Visual kNN Few-Shot MLLM Evaluation

Example for AID 3-shot:

```bash
python run_knn_fewshot_mllm.py \
  --manifest-dir manifests/aid \
  --shot 3 \
  --model llama-3.2-11b-vision-instruct \
  --api-base https://your-api-host/v1 \
  --api-key YOUR_API_KEY \
  --feature-backend remoteclip \
  --remoteclip-cache checkpoints \
  --out-dir results/aid_knn_shot_3_llama11b
```

The formal visual kNN baseline retrieves the global top-\(k\) support images using RemoteCLIP image-image cosine similarity.

---

## Step 7: Run RS-ViSemDS

The following command illustrates the paper configuration. Use the matching entry point and flag names from the paper-aligned branch.

### AID

```bash
python run_rs_visemds.py \
  --manifest-dir manifests/aid \
  --model internvl3.5-14b \
  --category-descriptions category_texts/aid_descriptions.json \
  --boundary-rules category_texts/aid_boundary_rules.json \
  --candidate-per-class 3 \
  --shot 3 \
  --alpha 0.6 \
  --beta 0.2 \
  --gamma 0.2 \
  --feature-backend remoteclip \
  --remoteclip-cache checkpoints \
  --out-dir results/aid_rs_visemds_internvl35_14b
```

### NWPU-Urban

```bash
python run_rs_visemds.py \
  --manifest-dir manifests/nwpu_fg_urban \
  --model internvl3.5-14b \
  --category-descriptions category_texts/nwpu_urban_descriptions.json \
  --boundary-rules category_texts/nwpu_urban_boundary_rules.json \
  --candidate-per-class 3 \
  --shot 3 \
  --alpha 0.6 \
  --beta 0.2 \
  --gamma 0.2 \
  --feature-backend remoteclip \
  --remoteclip-cache checkpoints \
  --out-dir results/nwpu_rs_visemds_internvl35_14b
```

The formal implementation must:

1. retrieve the top three candidates from each class;
2. compute the three RS-ViSemDS score components;
3. normalize each score component within the current target-specific candidate pool;
4. compute the weighted score;
5. select the global top three candidates;
6. order demonstrations by descending visual-semantic score;
7. append boundary-aware rules only to the final RS-ViSemDS prompt; and
8. keep both RemoteCLIP and the MLLM frozen.

The target-category semantic-affinity score is used only as a prior for demonstration selection. It is not used as the final class-prediction score.

---

## Local Open-Weight MLLM Inference

The paper uses a single NVIDIA A800 80 GB GPU with Hugging Face Transformers.

Formal settings:

```text
torch_dtype        = bfloat16
device_map         = auto
trust_remote_code  = True
decoding           = greedy
do_sample          = False
max_new_tokens     = 256
temperature        = not used
top_k              = not used
top_p              = not used
```

All model parameters remain frozen during inference.

GPT-4o is evaluated through the official API as a closed-source reference.

---

## Conventional Few-Shot Visual Baselines

The conventional few-shot baselines use:

- ImageNet-pretrained ResNet-18, ResNet-50, ViT-Tiny, and ViT-Small;
- frozen pretrained backbones;
- newly initialized trainable classification heads;
- class-balanced random sampling from the support pool;
- \(k_{\mathrm{cls}}\) examples per class;
- 10 training epochs;
- batch size 16;
- Adam with learning rate 0.001;
- cross-entropy loss;
- input size \(224 \times 224\); and
- random seed 42.

The formal paper protocol does **not** define conventional few-shot training as one fresh model per test image. It also does not use the target image to retrieve the conventional model's training set.

A paper-aligned implementation should build one class-balanced training subset for each \(k_{\mathrm{cls}}\) setting and evaluate the resulting classifier on the complete fixed test set.

Example conceptual configuration:

```text
k_cls = 1, 3, 5, or 10 images per class
train mode = head only
backbone = frozen
epochs = 10
batch size = 16
seed = 42
```

---

## Full-Data Head-Only Baselines

Full-data baselines use the same fixed test sets as all MLLM methods.

The support pool is handled as follows:

1. split the support pool into 90% training and 10% validation using seed 42;
2. use this split only to determine the training configuration;
3. keep the fixed test set completely untouched;
4. after fixing the configuration, retrain the classification head on the complete support pool for 10 epochs; and
5. evaluate once on the fixed test set.

Therefore, the 10% validation subset is not the paper's final evaluation set.

Example:

```bash
python run_full_data_baseline.py \
  --manifest-dir manifests/aid \
  --models resnet18 resnet50 vit_tiny vit_small \
  --epochs 10 \
  --batch-size 16 \
  --lr 0.001 \
  --train-mode head \
  --seed 42 \
  --out-dir results/aid_full_data_head
```

The final implementation should record both the configuration-selection stage and the final retraining stage in `run_config.json`.

---

## Metrics

Report:

- accuracy;
- macro precision;
- macro recall;
- macro F1-score;
- per-class accuracy; and
- confusion matrices.

Invalid MLLM responses that cannot be matched to one predefined label are counted as incorrect predictions.

For the class-balanced test sets used here, macro recall is numerically equal to overall accuracy. Both are retained in the paper tables for metric completeness and comparability.

---

## Timing

For MLLM-based methods, average per-image time should include:

- RemoteCLIP feature retrieval, when applicable;
- candidate generation;
- visual-semantic scoring;
- demonstration selection;
- prompt construction; and
- MLLM generation.

For conventional visual models, per-image inference time is measured with batch size one and includes:

- image loading;
- preprocessing;
- device transfer; and
- forward computation.

Training or validation time is reported separately from per-image inference time.

---

## Ablation Studies

The paper includes three groups of ablations on NWPU-Urban with Qwen3-VL-8B under the 3-shot setting.

### Scoring components

```text
Visual only
Visual + Typicality
Visual + Semantic Affinity
Full RS-ViSemDS
```

### Category-text usage

```text
No category text
Prompt only
Selection only
Full RS-ViSemDS
```

### Weight sensitivity

```text
Uniform                    (1/3, 1/3, 1/3)
Typicality-dominant        (0.2, 0.6, 0.2)
Semantic-dominant          (0.2, 0.2, 0.6)
Visual-Typicality dominant (0.4, 0.4, 0.2)
Visual-Semantic dominant   (0.4, 0.2, 0.4)
Typicality-Semantic dom.   (0.2, 0.4, 0.4)
Visual-dominant, main      (0.6, 0.2, 0.2)
```

All ablation variants must reuse the same:

- fixed test set;
- support pool;
- candidate budget;
- prompt budget;
- category texts;
- inference settings; and
- output parser,

except for the component explicitly being ablated.

---

## Reproducibility

Use seed 42 consistently unless multiple-seed robustness is being studied.

The repository should record:

- input-manifest hashes;
- example-manifest hashes;
- category-text file hashes;
- prompt-template hashes;
- model identifiers;
- pretrained checkpoint identifiers;
- package versions;
- CUDA version;
- GPU name;
- decoding settings;
- scoring weights;
- candidate budget;
- prompt budget; and
- random seed.

Do not mix results produced under different:

- test manifests;
- support pools;
- RemoteCLIP checkpoints;
- prompt templates;
- category-text files;
- scoring weights; or
- model revisions.

---

## Data and Code Availability

The paper uses the public AID and NWPU-RESISC45 datasets.

Repository:

```text
https://github.com/OpenReasonLab/RS-ViSemDS
```

Dataset licensing and redistribution conditions remain subject to the original dataset providers.

---
