from __future__ import annotations

import argparse
from pathlib import Path

from eval100_protocol import DATASET_CONFIG, validate_eval100_manifest


ROOT = Path(__file__).resolve().parent
DATA_DIRS = (
    ROOT / "data_raw" / "AID_dataset",
    ROOT / "data_raw" / "NWPU-RESISC45",
)
BANNED_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
    ".pt", ".pth", ".bin", ".safetensors", ".zip", ".tar", ".gz",
}
REQUIRED_LAUNCHERS = (
    "run_gpt4o_aid_nwpu_all.py",
    "run_llama32_11b_aid_nwpu_all.py",
    "run_gemma3_12b_aid_nwpu_all.py",
    "run_qwen25vl_7b_aid_nwpu_all.py",
    "run_qwen3vl_8b_aid_nwpu_all.py",
    "run_internvl35_8b_aid_nwpu_all.py",
    "run_internvl35_14b_aid_nwpu_all.py",
    "run_all_per_class_fewshot.py",
    "run_full_data_fixed_eval_all.py",
    "RS-ViSemDS/run_rs_visemds_all.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the clean eval100 code package.")
    parser.add_argument(
        "--allow-data",
        action="store_true",
        help="Allow files under data_raw after the datasets have been installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for relative_path in REQUIRED_LAUNCHERS:
        path = ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing required launcher: {path}")

    for config in DATASET_CONFIG.values():
        validate_eval100_manifest(ROOT / config["manifest"])

    for data_dir in DATA_DIRS:
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Missing dataset directory: {data_dir}")
        if not args.allow_data and any(data_dir.iterdir()):
            raise ValueError(f"Dataset directory is not empty: {data_dir}")

    banned = [
        path.relative_to(ROOT)
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in BANNED_SUFFIXES
        and "data_raw" not in path.parts
    ]
    if banned:
        raise ValueError(f"Unexpected data/weight/archive files: {banned[:10]}")

    print("Package validation passed.")
    print("- eval100 manifests: valid")
    print(f"- dataset directories: {'allowed' if args.allow_data else 'empty'}")
    print("- required MLLM, traditional, and RS-ViSemDS launchers: present")
    print("- packaged image/weight/archive files: none")


if __name__ == "__main__":
    main()
