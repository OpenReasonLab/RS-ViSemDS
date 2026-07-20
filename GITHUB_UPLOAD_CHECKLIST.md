# GitHub Upload Checklist

## Recommended upload scope

Upload the complete contents of `Eval100_Reproducibility_Package/` as one GitHub repository.
Do not upload selected scripts separately: the launchers depend on the shared modules, configs,
fixed manifests, tests, and documentation in this package.

The repository should include:

- `.gitignore` and `.gitattributes`.
- The existing repository `LICENSE` (MIT, copyright OpenReasonLab).
- `README.md`, `PACKAGE_CONTENTS.md`, `PACKAGE_FILE_LIST.txt`,
  `MANUSCRIPT_CODE_CONSISTENCY.md`, `SEEDS_AND_DATASETS.md`, and `requirements.txt`.
- `assets/fig02_datasets_models_baseline_protocols.pdf` and
  `assets/fig03_rs_visemds_framework.pdf`.
- All files under `configs/`, `manifests/`, `strict_fewshot/`, `tests/`, and `RS-ViSemDS/`.
- All root-level `.py` and `.sh` source files.
- `data_raw/AID_dataset/.gitkeep` and `data_raw/NWPU-RESISC45/.gitkeep` so that the expected
  empty dataset directories appear after cloning.

`PACKAGE_FILE_LIST.txt` is required rather than optional because `verify_package.py` validates
every published payload file against its recorded size and SHA-256 digest.

## Never upload

- Dataset images, dataset archives, or extracted dataset payloads.
- Model weights, RemoteCLIP weights, Hugging Face caches, or feature caches.
- Generated Random, kNN, or RS-ViSemDS selection CSV files.
- Predictions, metrics, confusion matrices, checkpoints, logs, result tables, or result figures.
- `results*`, `logs*`, `examples/`, `RS-ViSemDS/examples/`, `checkpoints/`, `models/`,
  `RemoteCLIP/`, `archive/`, `__pycache__/`, or `.pytest_cache/`.
- `.env` files, API keys, SSH credentials, passwords, or machine-specific environment files.
- The old 24-images-per-class protocol and its outputs.

## Before pushing

Run from the repository root:

```bash
python verify_package.py
python -m unittest discover -s tests -v
python -m unittest discover -s RS-ViSemDS/tests -v
```

The repository's existing MIT `LICENSE` is included and must be retained. The dependency names
in `requirements.txt` are not version-pinned; for stronger environment reproducibility, publish
a lock file or an export of the exact successful AutoDL environment after verifying it on the
target GPU stack.
