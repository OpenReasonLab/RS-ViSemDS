from __future__ import annotations


def summarize_predictions(rows: list[dict], class_order: list[str]) -> tuple[dict, list[dict], dict[str, list[list[int]]]]:
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    summary = {}
    per_class_rows = []
    matrices = {}

    for model, model_rows in by_model.items():
        correct = sum(int(r["correct"]) for r in model_rows)
        total = len(model_rows)
        avg_train_seconds = _mean_float(model_rows, "train_seconds")
        avg_predict_seconds = _mean_float(model_rows, "predict_seconds")
        avg_total_seconds = _mean_float(model_rows, "total_seconds")
        per_class_acc = {}
        f1_values = []
        matrix = [[0 for _ in class_order] for _ in class_order]
        class_to_idx = {c: i for i, c in enumerate(class_order)}

        for r in model_rows:
            true_i = class_to_idx[r["true_label"]]
            pred_i = class_to_idx[r["pred_label"]]
            matrix[true_i][pred_i] += 1

        for cls in class_order:
            idx = class_to_idx[cls]
            tp = matrix[idx][idx]
            support = sum(matrix[idx])
            pred_count = sum(matrix[r][idx] for r in range(len(class_order)))
            recall = tp / support if support else 0.0
            precision = tp / pred_count if pred_count else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            per_class_acc[cls] = recall
            f1_values.append(f1)
            per_class_rows.append({
                "model": model,
                "class": cls,
                "support": support,
                "accuracy": recall,
                "precision": precision,
                "f1": f1,
            })

        summary[model] = {
            "num_targets": total,
            "overall_accuracy": correct / total if total else 0.0,
            "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
            "avg_train_seconds_per_target": avg_train_seconds,
            "avg_predict_seconds_per_target": avg_predict_seconds,
            "avg_total_seconds_per_target": avg_total_seconds,
            "per_class_accuracy": per_class_acc,
        }
        matrices[model] = matrix

    return summary, per_class_rows, matrices


def _mean_float(rows: list[dict], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        values.append(float(value))
    return sum(values) / len(values) if values else 0.0
