# Model Card

## Summary

This detector is fine-tuned for small aerial objects with two classes:

```text
0 = bird
1 = airplane
```

The `airplane` class includes drones and other aircraft-like aerial objects.

## Training Recipe

Training is sequential. Each stage starts from the previous stage's `best.pt`
checkpoint. Early drone-heavy stages teach long-range drone appearance; later
balanced bird/drone stages reduce false positives and catastrophic forgetting.

| Stage | Dataset code | Max epochs | Patience | Notes |
| --- | --- | ---: | ---: | --- |
| `stage01` | `dataset01` | 30 | 7 | Drone-only warm-up |
| `stage02` | `dataset02` | 30 | 7 | Multi-sensor visible/IR aerial data |
| `stage03` | `dataset03` | 100 | 20 | Small bird/drone set |
| `stage04` | `dataset04` | 100 | 20 | Small bird/drone set |
| `stage05` | `dataset05` | 60 | 12 | Medium bird/drone set |
| `stage06` | `dataset06` | 50 | 10 | Large balanced bird/drone set |
| `stage07` | `dataset07_merged` | 20 | 4 | Merged train set with new public data |
| `stage08` | `dataset07_merged` | 30 | 8 | Warm-restart continuation from `stage07` |

## Stage Metrics

Metrics below are validation metrics for the stage checkpoint unless noted.

| Stage | Source | Minutes | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: |
| `stage01` | `done.json` validation of `best.pt` | 463.9 | 0.5759 | 0.2565 |
| `stage02` | `done.json` validation of `best.pt` | 1049.8 | 0.6144 | 0.2641 |
| `stage03` | `done.json` validation of `best.pt` | 285.3 | 0.8644 | 0.3318 |
| `stage04` | `done.json` validation of `best.pt` | 589.9 | 0.8114 | 0.4455 |
| `stage05` | `done.json` validation of `best.pt` | 792.8 | 0.9373 | 0.4973 |
| `stage06` | `done.json` validation of `best.pt` | 1658.3 | 0.9660 | 0.6589 |
| `stage07` | `done.json` validation of `best.pt` | 4107.6 | 0.7152 | 0.4907 |
| `stage08` | warm-restart in progress | - | - | - |

Detailed per-epoch CSV files are in `experiments/stages/*/results.csv`.

## Intended Use

- Research and prototyping for aerial object detection.
- Offline analysis of bird-vs-drone false positives.
- Fine-tuning a larger compatible detector with the same anonymized data chain.

## Limitations

- Training data is not redistributed.
- Some source datasets contain near-duplicate video frames.
- Some stages are domain-specific and should not be interpreted as independent
  general-purpose benchmarks.
- Metrics are only comparable when the same private data ledger and split rules
  are used.

## Release Artifacts

Local release-ready weights are staged as:

```text
release-assets/weights/stage01_best.pt
release-assets/weights/stage02_best.pt
release-assets/weights/stage03_best.pt
release-assets/weights/stage04_best.pt
release-assets/weights/stage05_best.pt
release-assets/weights/stage06_best.pt
release-assets/weights/stage07_best.pt
```

The `release-assets/weights` directory is ignored by Git. Attach these files to
GitHub Releases instead of committing them to repository history.

`stage08_best.pt` should be added after the warm-restart continuation finishes.
