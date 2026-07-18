from __future__ import annotations

import argparse
from pathlib import Path

from strict_fewshot.utils import (
    read_csv,
    read_json,
    repo_path,
    seeded_sample,
    sha256_file,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--strategy", choices=["random", "knn", "hrs"], required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-backend", choices=["remoteclip", "image_stats"], default="remoteclip")
    parser.add_argument(
        "--knn-scope",
        choices=["global", "per_class"],
        default="global",
        help=(
            "global selects k examples from the whole support pool; per_class selects "
            "k nearest examples independently from every class (N-way k-shot)."
        ),
    )
    parser.add_argument("--remoteclip-cache", default="checkpoints")
    parser.add_argument("--remoteclip-checkpoint", default=None)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--feature-num-workers", type=int, default=0)
    parser.add_argument("--hrs-csv", default=None)
    return parser.parse_args()


def load_manifest(manifest_dir: Path):
    evaluation = read_csv(manifest_dir / "evaluation.csv")
    support = read_csv(manifest_dir / "support.csv")
    summary = read_json(manifest_dir / "summary.json")
    classes = read_json(manifest_dir / "class_order.json")["classes"]
    data_root = repo_path(summary["data_root"], Path.cwd())
    return evaluation, support, classes, data_root


def write_examples(out_dir: Path, strategy: str, shot: int, rows: list[dict]) -> None:
    fieldnames = [
        "strategy", "shot", "target_id", "target_label", "target_path",
        "example_label", "example_path", "rank", "score", "sampling_seed",
    ]
    write_csv(out_dir / f"examples_{strategy}_shot_{shot}.csv", rows, fieldnames)


def build_random(evaluation, support, classes, shots, out_dir, seed):
    unique_shots = sorted(set(shots))
    max_shot = max(unique_shots)
    if len(support) < max_shot:
        raise ValueError(f"Support pool has {len(support)} images, need shot={max_shot}")

    rows_by_shot = {shot: [] for shot in unique_shots}
    for target_index, target in enumerate(evaluation):
        target_seed = seed + target_index * 1009
        ordered = seeded_sample(support, max_shot, target_seed)

        for shot in unique_shots:
            chosen = ordered[:shot]

            for rank, ex in enumerate(chosen, start=1):
                rows_by_shot[shot].append({
                    "strategy": "random",
                    "shot": shot,
                    "target_id": target["target_id"],
                    "target_label": target["label"],
                    "target_path": target["path"],
                    "example_label": ex["label"],
                    "example_path": ex["path"],
                    "rank": rank,
                    "score": "",
                    "sampling_seed": target_seed,
                })

    for shot in unique_shots:
        write_examples(out_dir, "random", shot, rows_by_shot[shot])

    write_json(out_dir / "random_sampling_config.json", {
        "strategy": "random",
        "seed": seed,
        "shots": unique_shots,
        "shot_definition": "total_labeled_images_per_query",
        "sampling_pool": "entire_support_pool",
        "class_balancing": False,
        "class_coverage_required": False,
        "replacement": False,
        "query_specific": True,
        "nested_prefixes": True,
        "num_evaluation_images": len(evaluation),
        "num_support_images": len(support),
        "target_seed_rule": "seed + evaluation_index * 1009",
    })


def build_knn(
    evaluation, support, classes, shots, out_dir, data_root, backend,
    cache_dir, checkpoint, feature_batch_size, feature_num_workers, knn_scope,
):
    import torch

    from strict_fewshot.features import extract_image_stats, extract_remoteclip

    all_paths = [data_root / r["path"] for r in evaluation] + [data_root / r["path"] for r in support]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backend == "image_stats":
        feats = extract_image_stats(all_paths)
        resolved_checkpoint = None
    else:
        if checkpoint:
            checkpoint_path = Path(checkpoint).expanduser()
            ckpt = checkpoint_path.resolve() if checkpoint_path.is_absolute() else repo_path(checkpoint_path, Path.cwd())
        else:
            ckpt = None
        feats, resolved_checkpoint = extract_remoteclip(
            all_paths,
            repo_path(cache_dir, Path.cwd()),
            ckpt,
            device,
            batch_size=feature_batch_size,
            num_workers=feature_num_workers,
        )

    eval_feats = feats[:len(evaluation)]
    support_feats = feats[len(evaluation):]
    unique_shots = sorted(set(shots))
    max_shot = max(unique_shots)
    support_indices_by_class = {
        class_name: [index for index, row in enumerate(support) if row["label"] == class_name]
        for class_name in classes
    }
    if knn_scope == "per_class":
        insufficient = {
            class_name: len(indices)
            for class_name, indices in support_indices_by_class.items()
            if len(indices) < max_shot
        }
        if insufficient:
            raise ValueError(
                f"Classes without enough support images for shot={max_shot}: {insufficient}"
            )
    elif len(support) < max_shot:
        raise ValueError(f"Support pool has {len(support)} images, need shot={max_shot}")

    rows_by_shot = {shot: [] for shot in unique_shots}
    similarities = eval_feats @ support_feats.T
    for target_idx, target in enumerate(evaluation):
        if knn_scope == "per_class":
            ranked_by_class = []
            for class_name in classes:
                class_indices = support_indices_by_class[class_name]
                class_scores = similarities[target_idx, class_indices]
                values, local_indices = torch.topk(class_scores, k=max_shot)
                ranked_by_class.extend(
                    (class_name, float(value), class_indices[local_index])
                    for value, local_index in zip(values.tolist(), local_indices.tolist())
                )
        else:
            values, indices = torch.topk(similarities[target_idx], k=max_shot)
            global_ranking = list(zip(values.tolist(), indices.tolist()))

        for shot in unique_shots:
            if knn_scope == "per_class":
                selected = []
                for class_name in classes:
                    selected.extend(
                        (value, support_idx)
                        for ranked_class, value, support_idx in ranked_by_class
                        if ranked_class == class_name
                    )
                selected = [
                    item
                    for class_index in range(len(classes))
                    for item in selected[class_index * max_shot:class_index * max_shot + shot]
                ]
            else:
                selected = global_ranking[:shot]

            for rank, (value, support_idx) in enumerate(selected, start=1):
                ex = support[support_idx]
                rows_by_shot[shot].append({
                    "strategy": "knn",
                    "shot": shot,
                    "target_id": target["target_id"],
                    "target_label": target["label"],
                    "target_path": target["path"],
                    "example_label": ex["label"],
                    "example_path": ex["path"],
                    "rank": rank,
                    "score": f"{value:.8f}",
                    "sampling_seed": "",
                })

    for shot in unique_shots:
        write_examples(out_dir, "knn", shot, rows_by_shot[shot])

    write_json(out_dir / "retrieval_config.json", {
        "strategy": "knn",
        "feature_backend": backend,
        "encoder": "RemoteCLIP-ViT-B-32" if backend == "remoteclip" else "image_stats_debug_only",
        "similarity": "cosine_similarity",
        "l2_normalized_embeddings": True,
        "scope": "per_class_support_pool" if knn_scope == "per_class" else "global_support_pool",
        "shot_definition": "examples_per_class" if knn_scope == "per_class" else "total_examples",
        "examples_per_target": {
            str(shot): shot * len(classes) if knn_scope == "per_class" else shot
            for shot in unique_shots
        },
        "shots": unique_shots,
        "nested_top_k": True,
        "num_evaluation_images": len(evaluation),
        "num_support_images": len(support),
        "feature_batch_size": feature_batch_size,
        "feature_num_workers": feature_num_workers,
        "checkpoint_file": resolved_checkpoint.name if resolved_checkpoint else None,
        "checkpoint_sha256": sha256_file(resolved_checkpoint) if resolved_checkpoint else None,
    })


def build_hrs(evaluation, support, classes, shots, out_dir, hrs_csv):
    if hrs_csv is None:
        raise ValueError("--hrs-csv is required for --strategy hrs")
    hrs_rows = read_csv(hrs_csv)
    eval_by_id = {r["target_id"]: r for r in evaluation}
    support_paths = {(r["label"], r["path"]) for r in support}

    for shot in shots:
        rows = []
        grouped = {}
        for row in hrs_rows:
            grouped.setdefault(row["target_id"], []).append(row)
        for target in evaluation:
            candidates = sorted(
                grouped.get(target["target_id"], []),
                key=lambda r: int(r.get("rank") or 999999),
            )[:shot]
            if len(candidates) != shot:
                raise ValueError(f"HRS target={target['target_id']} has {len(candidates)} examples, need {shot}")
            seen_paths = set()
            for rank, ex in enumerate(candidates, start=1):
                key = (ex["example_label"], ex["example_path"])
                if key not in support_paths:
                    raise ValueError(f"HRS example is not in support pool: {key}")
                if ex["example_path"] in seen_paths:
                    raise ValueError(f"Duplicate HRS example for target={target['target_id']}: {ex['example_path']}")
                seen_paths.add(ex["example_path"])
                rows.append({
                    "strategy": "hrs",
                    "shot": shot,
                    "target_id": target["target_id"],
                    "target_label": target["label"],
                    "target_path": target["path"],
                    "example_label": ex["example_label"],
                    "example_path": ex["example_path"],
                    "rank": rank,
                    "score": ex.get("score", ""),
                    "sampling_seed": ex.get("sampling_seed", ""),
                })
        write_examples(out_dir, "hrs", shot, rows)


def main():
    args = parse_args()
    manifest_dir = repo_path(args.manifest_dir, Path.cwd())
    out_dir = repo_path(args.out_dir, Path.cwd())
    evaluation, support, classes, data_root = load_manifest(manifest_dir)

    if args.strategy == "random":
        build_random(evaluation, support, classes, args.shots, out_dir, args.seed)
    elif args.strategy == "knn":
        build_knn(
            evaluation, support, classes, args.shots, out_dir, data_root,
            args.feature_backend, args.remoteclip_cache, args.remoteclip_checkpoint,
            args.feature_batch_size, args.feature_num_workers, args.knn_scope,
        )
    else:
        hrs_csv = repo_path(args.hrs_csv, Path.cwd()) if args.hrs_csv else None
        build_hrs(evaluation, support, classes, args.shots, out_dir, hrs_csv)
    print(f"Wrote {args.strategy} examples to {out_dir}")


if __name__ == "__main__":
    main()
