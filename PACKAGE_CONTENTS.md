# Package Contents

## Included

- Seven current eval100 MLLM launchers.
- Shared zero-shot, Random few-shot, and global RemoteCLIP-kNN MLLM evaluators.
- Four traditional vision model few-shot and fixed-evaluation full-data runners.
- `strict_fewshot` shared data, model, feature, metric, and local-MLLM utilities.
- AID and NWPU-Urban configs.
- Exact seed-42 eval100 manifests.
- Precomputed per-class RemoteCLIP kNN CSV/JSON files used by traditional few-shot runs.
- Complete RS-ViSemDS implementation, tests, and shell launchers.
- Two empty raw-dataset directories.

## Intentionally excluded

- All raw images and downloaded dataset archives.
- All MLLM, ImageNet backbone, and RemoteCLIP weight files.
- Checkpoints, feature caches, generated RS-ViSemDS selections, results, logs, plots, and tables.
- API keys, passwords, SSH credentials, and machine-specific environment files.
- Old 24-images-per-class manifests and launchers.
- Old HRS/Skill Prompt experiments that are not the final RS-ViSemDS implementation.
- Duplicate model-folder copies of common scripts.

`PACKAGE_FILE_LIST.txt` is generated from the final package and gives the relative path, byte size, and SHA-256 hash of every payload file (the list file itself is excluded).
