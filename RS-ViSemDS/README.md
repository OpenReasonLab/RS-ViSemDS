# RS-ViSemDS paper implementation

This directory implements the final method described in the accompanying paper.
The retrieval encoder and MLLM remain frozen; the test label is introduced only
after retrieval, prompt construction, and inference.

## Implemented protocol

- RemoteCLIP image/text embeddings are L2 normalized.
- Each class prototype is the renormalized mean of ten normalized descriptions.
- The candidate pool contains the top `r=3` visual neighbors from every class.
- Candidate image, typicality, and semantic scores are min-max normalized inside
  the current `r*C` pool.
- Visual class evidence is a stable temperature-scaled log-mean-exp over the
  complete class support set. The top-r pool is not used for this evidence.
- Semantic class evidence is target/prototype cosine similarity divided by its
  separately calibrated temperature.
- Both evidence distributions use softmax. Concentration, normalized JSD, and
  the log-odds update implement equations (9)-(13).
- The typicality weight remains fixed at `beta0=0.2`; the main base prior is
  `(0.6, 0.2, 0.2)`.
- The final demonstrations are the pure global Top-3. There is no visual anchor
  and no final class-balance constraint.
- Formal prompts use all boundary-aware category rules, a real system role, and
  the Appendix-B ordering.
- Greedy decoding uses `bfloat16`, `device_map=auto`, `do_sample=False`, and
  `max_new_tokens=256`.
- A response that cannot be uniquely matched to one candidate label is retained
  and counted as incorrect; it is not regenerated on resume.

## Support-only temperature calibration

`run_rs_visemds.py` deterministically samples a class-balanced calibration subset
from the support pool (seed 42 by default) and fits `T_vis` and `T_sem` separately
by multiclass negative log likelihood. Every held-out support query is excluded
from its own visual reference set. The chosen indices, temperatures, objective
values, search bounds, and elapsed time are stored in `run_config.json`.

The paper specifies support-only stratified calibration but does not state its
optimization objective or subset size. This repository therefore makes those
choices explicit and configurable instead of silently fixing undocumented values.

The main prior `(0.6, 0.2, 0.2)` is the fixed value reported by the paper and is
never selected or revised with test images. The manuscript does not disclose an
objective for re-estimating this prior, so the reproduction code does not invent
one.

## Formal runs

For the paper's required ten repeated runs, use the suite entry point from the
repository root:

```bash
python RS-ViSemDS/run_rs_visemds_all.py --runs 10
```

Replace `--model-id` with the exact Gemma-3-12B or InternVL3.5-14B checkpoint
used by the experiment. The public package contains only the current adaptive
selector and the locked `paper_v1` inference path.

## Verification

```bash
python -m unittest discover -s RS-ViSemDS/tests -t RS-ViSemDS -v
```

The tests explicitly cover equations (7)-(15), support-query exclusion,
temperature calibration, pure global Top-k selection, prompt structure, label
isolation, and invalid-output accounting.
