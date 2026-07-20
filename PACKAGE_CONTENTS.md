# Package Contents

## Included

- Seven eval100 MLLM launchers and the shared zero-shot, Random few-shot, and global RemoteCLIP-kNN evaluators.
- Four conventional visual few-shot baselines with model-specific ImageNet-backbone kNN retrieval.
- Four full-data frozen-backbone, head-only transfer-learning baselines.
- The complete RS-ViSemDS implementation, manuscript prompt, tests, and launchers.
- The manuscript's Figure 2 and Figure 3 protocol/framework PDFs under `assets/`.
- AID and NWPU-Urban configuration files and exact seed-42 eval100 manifests.
- Raw-data directory placeholders tracked with `.gitkeep`; no dataset images are included.
- Package tests, consistency notes, and a SHA-256 payload inventory.
- The repository's existing MIT license.

## Intentionally Excluded

- Raw images and dataset archives.
- MLLM, ImageNet-backbone, and RemoteCLIP weights or caches.
- Generated Random/kNN/RS-ViSemDS example selections.
- Predictions, metrics, confusion matrices, checkpoints, logs, result plots, and result tables.
- PDFs other than the two explicitly packaged manuscript protocol/framework figures.
- API keys, passwords, SSH credentials, and machine-specific environment files.
- Old 24-images-per-class code and obsolete experiment outputs.

`PACKAGE_FILE_LIST.txt` inventories every included payload file except itself. The package
validator rejects generated examples and experiment-result artifacts.
