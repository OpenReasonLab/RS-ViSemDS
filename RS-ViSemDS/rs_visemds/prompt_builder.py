from __future__ import annotations

import hashlib
import json
from pathlib import Path

from strict_fewshot.local_mllm import image_part, load_rgb_image, text_part

from .category_texts import boundary_rules, canonical_dataset_name


SYSTEM_PROMPT = (
    "You are a remote-sensing scene classification assistant. Analyze only visible "
    "content in the supplied overhead images. Select exactly one label from the "
    "candidate list."
)
PROMPT_MODE = "paper_v1"


def boundary_rule_block(dataset: str, class_order: list[str]) -> str:
    rules = boundary_rules(dataset, class_order)
    return "\n".join(f"- {label}: {rules[label]}" for label in class_order)


def task_text(dataset: str, class_order: list[str], example_count: int) -> str:
    labels = ", ".join(class_order)
    return "\n\n".join([
        "Task Instruction: The selected labeled images below are the only "
        "positive classification evidence. Compare the target with the "
        "reference examples as complete scenes.",
        f"Candidate Label Set: {labels}. Allowed answer strings must be copied "
        "exactly, including capitalization and underscores.",
        "Boundary-aware Category Rules:\n"
        "Stage A: Compare the target with the demonstrations as complete scenes "
        "and identify the provisional label P and runner-up R. If P is a clear "
        "match, return P. Stage B: Only if P and R remain ambiguous, apply "
        "exactly one P-versus-R exclusion check. Keep P unless clear scene-level "
        "counterevidence contradicts it and the target remains consistent with "
        "R. The check cannot positively support a label or introduce a third "
        "label.\n" + boundary_rule_block(dataset, class_order),
        f"Visual-Semantic Demonstrations: {example_count} score-ordered labeled "
        "image(s) follow.",
    ])


def output_instruction(class_order: list[str]) -> str:
    return (
        "Query: Classify the target image into exactly one candidate class. "
        "Return exactly one compact JSON object and no other text: "
        '{"answer":"<one candidate class>",'
        '"thoughts":"<brief observable visual evidence>",'
        '"score":<number from 0 to 1>}. '
        f"Allowed answers: {', '.join(class_order)}"
    )


def category_rules_sha256(dataset: str, class_order: list[str]) -> str:
    payload = {
        "dataset": canonical_dataset_name(dataset),
        "class_order": list(class_order),
        "prompt_mode": PROMPT_MODE,
        "category_rule_type": "complete_boundary_aware_category_rules",
        "category_rules": boundary_rule_block(dataset, class_order),
    }
    return _sha256_payload(payload)


def prompt_template_sha256(
    dataset: str,
    class_order: list[str],
    example_count: int = 3,
) -> str:
    """Hash the exact static prompt structure, excluding image-specific values."""
    if example_count != 3:
        raise ValueError("Formal Adaptive RS-ViSemDS uses exactly three examples")
    payload = {
        "system_prompt": SYSTEM_PROMPT,
        "task_text": task_text(dataset, class_order, example_count),
        "example_blocks": [
            (
                f"Selected example {index}/{example_count}. "
                "Ground-truth label: {example_label}\n<image>"
            )
            for index in range(1, example_count + 1)
        ],
        "target_block": "Target Input: the next image is the unlabeled target image.\n<image>",
        "output_instruction": output_instruction(class_order),
        "prompt_mode": PROMPT_MODE,
    }
    return _sha256_payload(payload)


def _sha256_payload(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_local_messages_and_images(
    data_root: Path,
    target_path: str,
    dataset: str,
    class_order: list[str],
    examples: list[dict],
) -> tuple[list[dict], list]:
    content = [text_part(task_text(dataset, class_order, len(examples)))]
    images = []
    for index, example in enumerate(examples, start=1):
        content.append(text_part(
            f"Selected example {index}/{len(examples)}. "
            f"Ground-truth label: {example['example_label']}"
        ))
        content.append(image_part())
        images.append(load_rgb_image(data_root / example["example_path"]))
    content.append(text_part("Target Input: the next image is the unlabeled target image."))
    content.append(image_part())
    images.append(load_rgb_image(data_root / target_path))
    content.append(text_part(output_instruction(class_order)))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ], images
