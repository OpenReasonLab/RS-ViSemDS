from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval100_protocol import validate_eval100_manifest
from strict_fewshot.utils import repo_path, sha256_file, write_csv, write_json

from rs_visemds.embedding_backend import load_or_build_embeddings
from rs_visemds.selector import select_demonstrations


SELECTED_FIELDS = [
    "strategy", "shot", "target_id", "target_label", "target_path",
    "example_label", "example_path", "rank", "score", "sampling_seed",
    "support_index", "S_img", "S_typ", "S_sem", "S_img_norm",
    "S_typ_norm", "S_sem_norm", "selection_seconds",
]
CANDIDATE_FIELDS = [
    "target_id", "target_label", "target_path", "candidate_rank", "selected",
    "support_index", "example_label", "example_path", "R", "S_img", "S_typ",
    "S_sem", "S_img_norm", "S_typ_norm", "S_sem_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RS-ViSemDS demonstrations for a fixed eval100 manifest."
    )
    parser.add_argument("--dataset", required=True, choices=["aid", "nwpu_fg_urban"])
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cache-dir", default="RS-ViSemDS/cache")
    parser.add_argument("--remoteclip-cache", default="checkpoints")
    parser.add_argument("--remoteclip-checkpoint", default="")
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--feature-num-workers", type=int, default=0)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--limit-per-class", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = (args.alpha, args.beta, args.gamma)
    if any(value < 0 for value in weights) or abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("alpha, beta, and gamma must be non-negative and sum to 1")
    manifest_dir = repo_path(args.manifest_dir, PROJECT_ROOT)
    out_dir = repo_path(args.out_dir, PROJECT_ROOT)
    cache_dir = repo_path(args.cache_dir, PROJECT_ROOT)
    checkpoint = None
    if args.remoteclip_checkpoint:
        checkpoint_arg = Path(args.remoteclip_checkpoint).expanduser()
        checkpoint = (
            checkpoint_arg
            if checkpoint_arg.is_absolute()
            else repo_path(checkpoint_arg, PROJECT_ROOT)
        )
    remoteclip_cache = repo_path(args.remoteclip_cache, PROJECT_ROOT)
    validate_eval100_manifest(manifest_dir)
    bundle = load_or_build_embeddings(
        dataset=args.dataset,
        manifest_dir=manifest_dir,
        project_root=PROJECT_ROOT,
        cache_dir=cache_dir,
        remoteclip_cache=remoteclip_cache,
        remoteclip_checkpoint=checkpoint,
        batch_size=args.feature_batch_size,
        num_workers=args.feature_num_workers,
        force=args.force_cache,
    )
    _assert_no_leakage(bundle.target_rows, bundle.support_rows)
    targets = _limit_per_class(
        bundle.target_rows, args.limit_per_class if args.limit_per_class > 0 else None
    )
    target_index = {row["target_id"]: index for index, row in enumerate(bundle.target_rows)}
    support_labels = [row["label"] for row in bundle.support_rows]
    selected_rows: list[dict] = []
    candidate_rows: list[dict] = []

    for number, target in enumerate(targets, start=1):
        start = time.perf_counter()
        selected, candidates = select_demonstrations(
            target_embedding=bundle.target_embeddings[target_index[target["target_id"]]],
            support_embeddings=bundle.support_embeddings,
            support_labels=support_labels,
            category_prototypes=bundle.category_prototypes,
            class_order=bundle.class_order,
            r=args.r,
            k=args.k,
            weights=(args.alpha, args.beta, args.gamma),
            eps=args.eps,
        )
        elapsed = time.perf_counter() - start
        selected_indices = {row.support_index for row in selected}
        for rank, row in enumerate(selected, start=1):
            example = bundle.support_rows[row.support_index]
            selected_rows.append({
                "strategy": "rs_visemds",
                "shot": args.k,
                "target_id": target["target_id"],
                "target_label": target["label"],
                "target_path": target["path"],
                "example_label": example["label"],
                "example_path": example["path"],
                "rank": rank,
                "score": _fmt(row.score),
                "sampling_seed": "",
                "support_index": row.support_index,
                "S_img": _fmt(row.s_img),
                "S_typ": _fmt(row.s_typ),
                "S_sem": _fmt(row.s_sem),
                "S_img_norm": _fmt(row.s_img_norm),
                "S_typ_norm": _fmt(row.s_typ_norm),
                "S_sem_norm": _fmt(row.s_sem_norm),
                "selection_seconds": f"{elapsed:.8f}",
            })
        for rank, row in enumerate(candidates, start=1):
            example = bundle.support_rows[row.support_index]
            candidate_rows.append({
                "target_id": target["target_id"],
                "target_label": target["label"],
                "target_path": target["path"],
                "candidate_rank": rank,
                "selected": int(row.support_index in selected_indices),
                "support_index": row.support_index,
                "example_label": example["label"],
                "example_path": example["path"],
                "R": _fmt(row.score),
                "S_img": _fmt(row.s_img),
                "S_typ": _fmt(row.s_typ),
                "S_sem": _fmt(row.s_sem),
                "S_img_norm": _fmt(row.s_img_norm),
                "S_typ_norm": _fmt(row.s_typ_norm),
                "S_sem_norm": _fmt(row.s_sem_norm),
            })
        if number % 100 == 0 or number == len(targets):
            print(f"Selected demonstrations for {number}/{len(targets)} targets")

    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = out_dir / f"examples_rs_visemds_shot_{args.k}.csv"
    write_csv(selected_path, selected_rows, SELECTED_FIELDS)
    write_csv(out_dir / "candidate_scores.csv", candidate_rows, CANDIDATE_FIELDS)
    write_json(out_dir / "selection_config.json", {
        "method": "RS-ViSemDS",
        "strategy": "rs_visemds",
        "dataset": args.dataset,
        "manifest_dir": str(manifest_dir),
        "r_per_class": args.r,
        "k_total_demonstrations": args.k,
        "candidate_pool_size": args.r * len(bundle.class_order),
        "weights": {"alpha": args.alpha, "beta": args.beta, "gamma": args.gamma},
        "eps": args.eps,
        "score_normalization": "component-wise min-max over the r*C candidate pool",
        "selection_order": "descending R; ties by descending S_img then support index",
        "num_targets": len(targets),
        "num_support": len(bundle.support_rows),
        "class_order": bundle.class_order,
        "evaluation_sha256": sha256_file(manifest_dir / "evaluation.csv"),
        "support_sha256": sha256_file(manifest_dir / "support.csv"),
        "class_order_sha256": sha256_file(manifest_dir / "class_order.json"),
        "selected_examples_sha256": sha256_file(selected_path),
        "embedding_metadata": bundle.metadata,
        "limit_per_class": args.limit_per_class or None,
    })
    print(f"Wrote RS-ViSemDS selections to {out_dir}")


def _assert_no_leakage(evaluation: list[dict], support: list[dict]) -> None:
    overlap = {row["path"] for row in evaluation} & {row["path"] for row in support}
    if overlap:
        raise ValueError(f"Evaluation/support leakage: {sorted(overlap)[:5]}")


def _limit_per_class(rows: list[dict], limit: int | None) -> list[dict]:
    if limit is None:
        return rows
    counts = defaultdict(int)
    output = []
    for row in rows:
        if counts[row["label"]] < limit:
            output.append(row)
            counts[row["label"]] += 1
    return output


def _fmt(value: float) -> str:
    return f"{float(value):.8f}"


if __name__ == "__main__":
    main()
