from __future__ import annotations

from pathlib import Path

from strict_fewshot.local_mllm import image_part, load_rgb_image, text_part

from .category_texts import boundary_rules, canonical_dataset_name


SYSTEM_PROMPT = (
    "You are a remote-sensing scene classification assistant. Analyze only visible "
    "content in the supplied overhead images. Do not use filenames, metadata, or "
    "unstated context. Select exactly one label from the candidate list."
)

PROMPT_MODES = (
    "manuscript_v1",
    "legacy",
    "reference_guided_v1",
    "reference_only_v1",
    "reference_fallback_v2",
    "reference_fallback_v3",
)

MANUSCRIPT_INSTRUCTION = (
    "The selected labeled images below are the only positive classification evidence. "
    "Compare the target with the reference examples as complete scenes. Stage A: "
    "Compare the target with the demonstrations as complete scenes and identify the "
    "provisional label P and runner-up R. If P is a clear match, return P. Stage B: "
    "Only if P and R remain ambiguous, apply exactly one P-versus-R exclusion check. "
    "Keep P unless clear scene-level counterevidence contradicts it and the target "
    "remains consistent with R. The check cannot positively support a label or "
    "introduce a third label."
)

REFERENCE_GUIDED_INSTRUCTION = (
    "The following labeled images are reference examples selected for their visual "
    "and semantic relevance. Use them to identify label-associated visual patterns, "
    "including object type, building footprint, spatial layout, density, roads, "
    "parking areas, and surrounding context. First compare the target image with "
    "the visual patterns shown in the reference examples. Then use the category "
    "descriptions as secondary guidance to resolve remaining ambiguity. Do not "
    "mechanically copy the majority example label, and do not let a single generic "
    "cue—such as a large roof, dense buildings, parking areas, or an urban setting—"
    "override the overall visual evidence."
)

REFERENCE_ONLY_INSTRUCTION = (
    "The following labeled images are reference examples selected for their visual "
    "and semantic relevance. Use them to identify label-associated visual patterns, "
    "including object type, building footprint, spatial layout, density, roads, "
    "parking areas, and surrounding context. Compare the target image with the "
    "overall visual patterns shown in these examples and select the best matching "
    "candidate label. Do not mechanically copy the majority example label, and do "
    "not let a single generic cue—such as a large roof, dense buildings, parking "
    "areas, or an urban setting—override the complete visual evidence."
)

REFERENCE_FALLBACK_V2_INSTRUCTION = (
    "The labeled images below are the primary reference evidence. Infer "
    "label-associated visual patterns from the reference examples, including object "
    "type, building footprint, spatial layout, density, roads, parking areas, and "
    "surrounding context. First classify the target image according to its best "
    "overall visual match to those examples. If the example comparison produces one "
    "clearly best-matching label, select that label directly and do not let the "
    "category descriptions override it. Consult the category descriptions only as "
    "tie-breakers if two or more candidate labels remain similarly plausible after "
    "comparing the target with the reference examples. In that case, use only the "
    "descriptions of the competing labels to resolve their boundary. Treat the "
    "descriptions as secondary constraints, not as standalone category templates. "
    "Do not mechanically copy the majority example label, and do not classify from "
    "a single generic cue such as a large roof, visible streets, trees, parking "
    "areas, dense buildings, or an urban setting. Resolve any ambiguity internally. "
    "Always select exactly one label from the candidate set. Do not output "
    "uncertainty, multiple labels, unknown, or a request for more information."
)

REFERENCE_FALLBACK_V3_INSTRUCTION = (
    "The labeled images below are the only positive classification evidence. Compare "
    "the target with the reference examples as complete scenes, using object and roof "
    "types, footprint distribution, scene-wide density, spacing, road geometry, open "
    "space, and surrounding context. Stage A -- demonstration-only decision: "
    "internally rank the candidate labels from the reference examples alone. Freeze "
    "the best label as the provisional choice P and the second-best label as the "
    "runner-up R. If P is a clear overall match, output P and ignore all textual "
    "checks below. Stage B -- exclusion-only tie-break: enter this stage only when P "
    "and R remain genuinely indistinguishable after the full-scene example "
    "comparison. Consult exactly the single P-versus-R exclusion check below. These "
    "checks have zero positive weight: a feature mentioned in a check must never "
    "increase a label's score or make that label a candidate. Keep P by default. "
    "Change from P to R only when all three conditions hold: (1) clearly visible, "
    "scene-level counterevidence explicitly contradicts P; (2) R was already the "
    "runner-up from Stage A; and (3) the target remains a coherent overall match to "
    "R's reference examples. If counterevidence is absent, local, weak, mixed, or "
    "inferred from semantic purpose, keep P. Never use a check to introduce a third "
    "label. If no textual check is defined for P and R, keep P. Do not infer civic, "
    "commercial, residential, or religious function merely from an unusual building, "
    "a circular roof, a dome, a courtyard, roads, parking, trees, visible streets, or "
    "urban density. A single generic cue cannot trigger a label change. Repeated "
    "example-label support counts only when those examples genuinely match the target "
    "as complete scenes. Resolve the decision internally and return exactly one label "
    "from the candidate set; never output uncertainty, multiple labels, unknown, or a "
    "request for more information."
)

AID_REFERENCE_DESCRIPTIONS = {
    "Center": (
        "A central urban or civic complex organized around plazas, major "
        "intersections, landmark groups, and mixed large-scale structures. A "
        "single unusual building or general urban density alone is not sufficient; "
        "compare carefully with the Center reference examples."
    ),
    "Church": (
        "A worship-related building or religious complex. Evidence may include an "
        "elongated or cross-like footprint, nave-like roof, tower, spire, dome, "
        "courtyard, or a dominant worship building. If these cues are subtle, rely "
        "on similarity to the Church reference examples rather than defaulting to "
        "Center because of the surrounding urban context."
    ),
    "Commercial": (
        "Retail, office, service, shopping, or mixed-use complexes, often with "
        "large business blocks, access roads, parking, and paved service areas. "
        "Distinguish this from Center by commercial organization rather than "
        "general urban density, and from DenseResidential by the absence of "
        "dominant repeated housing patterns."
    ),
}

FALLBACK_DESCRIPTIONS_V2 = {
    "aid": {
        "Center": (
            "Use Center as a tie-breaker only when a visually distinctive civic, "
            "cultural, or landmark complex forms the clear spatial focus, with "
            "surrounding plazas, courtyards, roads, or structures organized around "
            "it. Dense urban development, a major intersection, parking areas, or a "
            "large roof alone is not sufficient. Prefer Commercial when multiple "
            "business or mixed-use blocks dominate without a clear landmark focal "
            "complex."
        ),
        "Church": (
            "Use Church as a tie-breaker when the dominant building matches the "
            "Church reference examples and shows worship-specific structural "
            "evidence, such as an elongated nave, cruciform plan, attached tower or "
            "spire, a dome structurally associated with the main worship building, "
            "or a coherent religious compound. A generic dome, courtyard, unusual "
            "roof, parking area, or dense urban setting alone is insufficient. Do "
            "not reject Church merely because it is surrounded by commercial or "
            "residential buildings."
        ),
        "Commercial": (
            "Use Commercial as a tie-breaker when business-oriented urban fabric "
            "dominates, such as mixed building scales, large or deep building "
            "footprints, multi-building blocks, major-road frontage, paved access or "
            "service areas, or parking. Commercial scenes may be compact and may not "
            "contain a large parking lot. Prefer DenseResidential only when repeated "
            "small housing roofs and residential lots dominate. Prefer Center only "
            "when a distinct civic or landmark complex clearly organizes the scene."
        ),
    },
    "nwpu_fg_urban": {
        "dense_residential": (
            "Use dense_residential as a tie-breaker when most of the scene is "
            "dominated by many closely spaced residential buildings or housing "
            "blocks, with narrow lots and short inter-building gaps. Visible streets, "
            "trees, or small yards do not by themselves make the scene "
            "medium_residential. Prefer medium_residential only when wider and more "
            "consistent spacing, larger yards or vegetation corridors, and lower "
            "overall building coverage are evident across most of the scene."
        ),
    },
}

PAIRWISE_EXCLUSION_CHECKS_V3 = {
    "aid": {
        "Center versus Commercial": (
            "Center is contradicted when the scene is a distributed fabric of several "
            "comparable building blocks and access or service spaces, with no single "
            "compound governing the scene through a clear axial, radial, plaza, or "
            "courtyard composition. An unusual, circular, or large roof, parking, "
            "roads, or a central position alone cannot preserve Center. Commercial is "
            "contradicted when the alleged commercial evidence consists only of roads, "
            "parking, or urban density and the entire scene is instead organized as "
            "one coherent focal architectural compound. Do not assign civic or "
            "business function from roof appearance alone."
        ),
        "Center versus Church": (
            "Church fails a strict structural gate unless the target and the Church "
            "references share a coherent worship-building configuration, such as an "
            "elongated nave with a terminal or crossing element, a cruciform main "
            "footprint, an attached tower or spire, or a dome structurally integrated "
            "with the main hall. A dome, circular roof, unusual landmark, courtyard, "
            "or prominent building alone contradicts neither Center nor proves Church. "
            "Center is contradicted only when that worship-specific configuration is "
            "clearly visible at the dominant focal building."
        ),
        "Church versus Commercial": (
            "Church is contradicted when no worship-specific configuration is visible "
            "and the decision depends only on a large roof, parking, roads, dense "
            "blocks, or a courtyard. Commercial is contradicted when a structurally "
            "coherent worship building is the unmistakable scene focus; surrounding "
            "shops, housing, roads, or parking do not cancel that structure."
        ),
        "Commercial versus DenseResidential": (
            "DenseResidential is contradicted when a few large, deep, or irregular "
            "non-house-like footprints and their access or service spaces dominate the "
            "scene. Commercial is contradicted when the dominant texture is many "
            "repeated small dwelling-scale roofs and residential lots, even if roads "
            "are narrow and no large parking lot is visible. Dense urban texture alone "
            "supports neither label."
        ),
    },
    "nwpu_fg_urban": {
        "dense_residential versus medium_residential": (
            "dense_residential is contradicted only when wider spacing, larger yards, "
            "or vegetation corridors are consistent across most housing blocks and "
            "materially lower the scene-wide roof coverage. Visible streets, trees, "
            "pools, water, or a few open gaps are local context and cannot by "
            "themselves contradict dense_residential. medium_residential is "
            "contradicted when close roof-to-roof packing dominates most blocks despite "
            "visible streets or vegetation."
        ),
        "dense_residential versus mobile_home_park": (
            "mobile_home_park is contradicted unless most units are narrow elongated "
            "rectangles with near-uniform size and orientation arranged in long "
            "repeated parallel rows. A regular street grid, repeated ordinary houses, "
            "small rectangular roofs, or close spacing alone is insufficient. "
            "dense_residential is contradicted only when that trailer-like row "
            "morphology dominates the scene."
        ),
    },
}


def fallback_description_block_v2(dataset: str, class_order: list[str]) -> str:
    rules = boundary_rules(dataset, class_order)
    dataset_key = canonical_dataset_name(dataset)
    rules.update({
        label: description
        for label, description in FALLBACK_DESCRIPTIONS_V2[dataset_key].items()
        if label in rules
    })
    return "\n".join(f"- {label}: {rules[label]}" for label in class_order)


def fallback_description_block_v3(dataset: str, class_order: list[str]) -> str:
    dataset_key = canonical_dataset_name(dataset)
    checks = PAIRWISE_EXCLUSION_CHECKS_V3[dataset_key]
    return "\n".join(f"- {pair}: {rule}" for pair, rule in checks.items())


def boundary_rule_block(dataset: str, class_order: list[str]) -> str:
    rules = boundary_rules(dataset, class_order)
    return "\n".join(f"- {label}: {rules[label]}" for label in class_order)


def task_text(
    dataset: str,
    class_order: list[str],
    example_count: int,
    prompt_mode: str = "legacy",
) -> str:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode: {prompt_mode}")
    labels = ", ".join(class_order)
    if prompt_mode == "manuscript_v1":
        return "\n\n".join([
            "Task Instruction: " + MANUSCRIPT_INSTRUCTION,
            f"Candidate Label Set: {labels}",
            "Allowed answer strings must be copied exactly, including capitalization and underscores.",
            "Boundary-aware Category Rules:\n" + boundary_rule_block(dataset, class_order),
            f"Visual-Semantic Demonstrations: {example_count} score-ordered labeled image(s) follow.",
        ])
    if prompt_mode == "reference_fallback_v3":
        return "\n\n".join([
            "Task Instruction: " + REFERENCE_FALLBACK_V3_INSTRUCTION,
            f"Candidate Label Set: {labels}",
            "Allowed answer strings must be copied exactly, including capitalization and underscores.",
            "Pairwise Exclusion Checks "
            "(use exactly one P-versus-R check; never use as positive evidence):\n"
            + fallback_description_block_v3(dataset, class_order),
            f"Reference Demonstrations: {example_count} score-ordered labeled image(s) follow.",
        ])
    if prompt_mode == "reference_fallback_v2":
        return "\n\n".join([
            "Task Instruction: " + REFERENCE_FALLBACK_V2_INSTRUCTION,
            f"Candidate Label Set: {labels}",
            "Allowed answer strings must be copied exactly, including capitalization and underscores.",
            "Fallback Category Descriptions "
            "(consult only after comparing the reference examples):\n"
            + fallback_description_block_v2(dataset, class_order),
            f"Reference Demonstrations: {example_count} score-ordered labeled image(s) follow.",
        ])
    if prompt_mode == "reference_only_v1":
        return "\n\n".join([
            "Task Instruction: " + REFERENCE_ONLY_INSTRUCTION,
            f"Candidate Label Set: {labels}",
            "Allowed answer strings must be copied exactly, including capitalization and underscores.",
            f"Reference Demonstrations: {example_count} score-ordered labeled image(s) follow.",
        ])
    if prompt_mode == "reference_guided_v1":
        rules = boundary_rules(dataset, class_order)
        if dataset == "aid":
            rules.update({
                label: description
                for label, description in AID_REFERENCE_DESCRIPTIONS.items()
                if label in rules
            })
        description_block = "\n".join(
            f"- {label}: {rules[label]}" for label in class_order
        )
        return "\n\n".join([
            "Task Instruction: " + REFERENCE_GUIDED_INSTRUCTION,
            f"Candidate Label Set: {labels}",
            "Allowed answer strings must be copied exactly, including capitalization and underscores.",
            "Category Descriptions (secondary guidance):\n" + description_block,
            f"Reference Demonstrations: {example_count} score-ordered labeled image(s) follow.",
        ])
    return "\n\n".join([
        "Task Instruction: Use the boundary-aware category rules and selected "
        "visual-semantic demonstrations below to classify the target image.",
        f"Candidate Label Set: {labels}",
        "Allowed answer strings must be copied exactly, including capitalization and underscores.",
        "Boundary-aware Category Rules:\n" + boundary_rule_block(dataset, class_order),
        f"Visual-Semantic Demonstrations: {example_count} score-ordered labeled image(s) follow.",
    ])


def output_instruction(
    class_order: list[str],
    prompt_mode: str = "legacy",
) -> str:
    if prompt_mode == "reference_fallback_v3":
        return (
            "Classify the target into exactly one candidate class. Return exactly one "
            "compact JSON object and no other text: "
            '{"thoughts":"<brief full-scene comparison to the reference examples; '
            "if and only if Stage B changed the provisional choice, state the decisive "
            'visible counterevidence>","answer":"<one candidate class>",'
            '"score":<number from 0 to 1>}. Do not mention inferred land-use purpose '
            "as evidence. Do not mention labels other than the final answer unless "
            "Stage B was actually required. "
            f"Allowed answers: {', '.join(class_order)}"
        )
    return (
        "Classify the target image into exactly one candidate class. Return exactly one "
        "compact JSON object and no other text. Use this schema: "
        '{"thoughts":"<brief observable visual evidence>",'
        '"answer":"<one candidate class>","score":<number from 0 to 1>}. '
        f"Allowed answers: {', '.join(class_order)}"
    )


def build_local_messages_and_images(
    data_root: Path,
    target_path: str,
    dataset: str,
    class_order: list[str],
    examples: list[dict],
    retry_instruction: str = "",
    prompt_mode: str = "legacy",
) -> tuple[list[dict], list]:
    content = [text_part(SYSTEM_PROMPT + "\n\n" + task_text(
        dataset, class_order, len(examples), prompt_mode=prompt_mode
    ))]
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
    content.append(text_part(output_instruction(
        class_order, prompt_mode=prompt_mode
    )))
    if retry_instruction:
        content.append(text_part(retry_instruction))
    return [{"role": "user", "content": content}], images
