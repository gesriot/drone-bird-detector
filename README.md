# Drone Bird Detector

Training pipeline and experiment records for a two-class aerial object detector:

- `bird`
- `airplane` (used for airplanes, helicopters, UAVs, and drones)

This repository intentionally does not include source datasets or labels. The
training corpus contains third-party datasets with mixed redistribution terms,
so the public repository keeps only the reproducible pipeline, anonymized data
provenance, metrics, and release-ready model artifacts.

## Repository Layout

```text
configs/                 Example local configuration files
experiments/             Sanitized metrics from the training chain
model/                   Released final model (.pt/.onnx/.torchscript) + license
release-assets/weights/  Local staging area for GitHub Release assets
scripts/                 Portable preparation, training, evaluation utilities
DATA_PROVENANCE.md       Public data provenance summary
MODEL_CARD.md            Model notes, metrics, intended use, limitations
NOTICE.md                License scope and third-party terms notice
```

## What Is Not Included

The following are intentionally excluded from Git:

- source images and labels
- derived train/val image trees
- absolute local path lists
- raw run directories
- virtual environments
- intermediate stage checkpoints in normal Git history

The released final model (`stage08`) is published in-repo under `model/` in
three formats (`.pt`, `.onnx`, `.torchscript`) with its AGPL-3.0 license. Its
checkpoint metadata was sanitized with `scripts/sanitize_checkpoint.py` before
export. Intermediate per-stage weights stay out of Git; stage them in
`release-assets/weights/` (ignored by Git) and attach to GitHub Releases instead.

## Reproduce Locally

1. Obtain the source datasets under their original terms.
2. Create `configs/datasets.local.yaml` from `configs/datasets.example.yaml`.
3. Create `configs/stages.local.yaml` from `configs/stages.example.yaml`.
4. Prepare each dataset so it uses the unified class scheme:

```text
0 = bird
1 = airplane
```

5. Run the sequential training chain:

```bash
python scripts/train_sequential.py --config configs/stages.local.yaml
```

6. Evaluate selected checkpoints:

```bash
python scripts/evaluate_checkpoints.py --config configs/eval.example.yaml
```

## Release Notes

Before publishing weights, inspect and sanitize checkpoint metadata:

```bash
python scripts/inspect_checkpoint.py release-assets/weights/stage07_best.pt
```

If you publish weights, include `MODEL_CARD.md`, `DATA_PROVENANCE.md`, and
`NOTICE.md` in the release notes. The repository license applies to code and
documentation only; datasets and model weights may be subject to separate
third-party terms.
