# RS-ViSemDS

Official implementation of **RS-ViSemDS: Query-Adaptive Visual-Semantic Demonstration Selection for Remote Sensing Scene Classification With Open-Weight Multimodal Large Language Models**.

RS-ViSemDS addresses the limitations of visual nearest-neighbor prompting by jointly modeling target-candidate visual relevance, candidate-category typicality, and target-category semantic affinity. It adaptively balances these signals according to query-specific evidence concentration and disagreement, while boundary-aware category rules guide frozen MLLM inference without task-specific parameter updates.

[Experimental protocols](assets/fig02_datasets_models_baseline_protocols.pdf) · [RS-ViSemDS framework](assets/fig03_rs_visemds_framework.pdf)

## Evaluation protocol

| Dataset | Selected classes | Fixed test set | Support pool |
|---|---:|---:|---:|
| AID | 10 | 1,000 (100/class) | 2,210 |
| NWPU-Urban | 8 | 800 (100/class) | 4,800 |

The fixed split uses seed 42. Accuracy, macro Precision, macro Recall, and macro F1 are reported as arithmetic means over ten runs.

## Method

- Short category-description ensembles and boundary-aware category rules encode complementary category knowledge.
- Class-balanced visual retrieval with the top 3 candidates per class.
- Support-only temperature calibration and adaptive visual-semantic fusion with base prior `(0.6, 0.2, 0.2)`.
- Pure global Top-3 selection followed by greedy frozen-MLLM inference.

The main RS-ViSemDS experiments use Gemma-3-12B, Qwen3-VL-8B, and InternVL3.5-14B.

## Quick start

```bash
python -m pip install -r requirements.txt
python run_mllm_paper_runs.py --help
python RS-ViSemDS/run_rs_visemds_all.py --runs 10
python run_all_per_class_fewshot.py --help
python run_full_data_fixed_eval_all.py --help
```

Formal commands and model-path examples are provided in [PAPER_REPRODUCTION.md](PAPER_REPRODUCTION.md).

## Data and models

Place AID and NWPU-RESISC45 under `data_raw/`, or pass dataset paths through the command-line arguments. Datasets, model weights, caches, API keys, and experiment outputs are not included. GPT-4o reads `OPENAI_API_KEY` from the environment.

## Verification

```bash
python -B verify_package.py
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s RS-ViSemDS/tests -v
```

## License

Released under the [MIT License](LICENSE).
