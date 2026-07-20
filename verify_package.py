from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from eval100_protocol import DATASET_CONFIG, validate_eval100_manifest


ROOT = Path(__file__).resolve().parent
DATA_DIRS = (
    ROOT / "data_raw" / "AID_dataset",
    ROOT / "data_raw" / "NWPU-RESISC45",
)
GENERATED_DIRS = (
    ROOT / "examples",
    ROOT / "RS-ViSemDS" / "examples",
)
BANNED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".pt", ".pth", ".bin", ".safetensors", ".zip", ".tar", ".gz",
    ".log", ".xlsx", ".xls", ".pdf",
}
BANNED_DIRECTORY_NAMES = {
    "__pycache__", ".pytest_cache", "logs", "archive", "checkpoints",
    "results", "results_eval100_seed42", "ablations",
}
ALLOWED_PDFS = {
    Path("assets/fig02_datasets_models_baseline_protocols.pdf"),
    Path("assets/fig03_rs_visemds_framework.pdf"),
}
REQUIRED_LAUNCHERS = (
    "run_gpt4o_aid_nwpu_all.py",
    "run_llama32_11b_aid_nwpu_all.py",
    "run_gemma3_12b_aid_nwpu_all.py",
    "run_qwen25vl_7b_aid_nwpu_all.py",
    "run_qwen3vl_8b_aid_nwpu_all.py",
    "run_internvl35_8b_aid_nwpu_all.py",
    "run_internvl35_14b_aid_nwpu_all.py",
    "run_open_mllm_eval100_autodl.sh",
    "build_backbone_knn_examples.py",
    "run_all_per_class_fewshot.py",
    "run_full_data_fixed_eval_all.py",
    "RS-ViSemDS/run_rs_visemds_all.py",
)
REQUIRED_PUBLIC_FILES = (
    ".gitignore",
    ".gitattributes",
    "README.md",
    "LICENSE",
    "GITHUB_UPLOAD_CHECKLIST.md",
    "PACKAGE_CONTENTS.md",
    "MANUSCRIPT_CODE_CONSISTENCY.md",
    "SEEDS_AND_DATASETS.md",
    "requirements.txt",
    "data_raw/AID_dataset/.gitkeep",
    "data_raw/NWPU-RESISC45/.gitkeep",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the clean eval100 code package.")
    parser.add_argument(
        "--allow-data",
        action="store_true",
        help="Allow files under data_raw after the datasets have been installed.",
    )
    return parser.parse_args()


def validate_payload_hashes() -> None:
    list_path = ROOT / "PACKAGE_FILE_LIST.txt"
    with list_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    listed = {row["relative_path"] for row in rows}
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path != list_path
    }
    if listed != actual:
        raise ValueError(
            f"Payload list mismatch: {len(actual - listed)} unlisted, "
            f"{len(listed - actual)} missing."
        )
    for row in rows:
        path = ROOT / row["relative_path"]
        if path.stat().st_size != int(row["bytes"]):
            raise ValueError(f"Payload size mismatch: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Payload SHA-256 mismatch: {path}")


def main() -> None:
    args = parse_args()
    for relative_path in REQUIRED_PUBLIC_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing required public file: {path}")
    for relative_path in REQUIRED_LAUNCHERS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing required launcher: {path}")
    for relative_path in ALLOWED_PDFS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing manuscript figure: {path}")

    for config in DATASET_CONFIG.values():
        validate_eval100_manifest(ROOT / config["manifest"])

    for data_dir in DATA_DIRS:
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {data_dir}")
        payload_files = [
            path for path in data_dir.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        ]
        if not args.allow_data and payload_files:
            raise ValueError(f"Dataset directory is not empty: {data_dir}")

    for generated_dir in GENERATED_DIRS:
        if generated_dir.is_dir() and any(path.is_file() for path in generated_dir.rglob("*")):
            raise ValueError(f"Generated experiment artifacts are packaged in: {generated_dir}")

    banned_dirs = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_dir() and path.name.lower() in BANNED_DIRECTORY_NAMES
    ]
    if banned_dirs:
        raise ValueError(f"Output/cache directories are packaged: {banned_dirs[:10]}")

    banned_files = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in BANNED_SUFFIXES
        and "data_raw" not in path.parts
        and path.relative_to(ROOT) not in ALLOWED_PDFS
    ]
    if banned_files:
        raise ValueError(f"Unexpected data/result/weight files: {banned_files[:10]}")

    validate_payload_hashes()
    print("Package validation passed.")
    print("- eval100 manifests: valid and leakage-free")
    print(f"- dataset directories: {'allowed' if args.allow_data else 'empty'}")
    print("- generated examples, results, logs, caches, weights, and archives: absent")
    print("- allowed manuscript Figure 2/3 PDFs: present")
    print("- GitHub metadata, documentation, and dataset placeholders: present")
    print("- required MLLM, traditional, full-data, and RS-ViSemDS code: present")
    print("- payload SHA-256 hashes: valid")


if __name__ == "__main__":
    main()
