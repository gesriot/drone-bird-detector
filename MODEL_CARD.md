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
| `stage08` | `dataset07_merged` | 30 | 8 | Warm-restart continuation from `stage07`; released final checkpoint |
| `stage09` | `dataset07_merged` | 30 | 8 | Further warm-restart from `stage08`; early-stopped, no improvement |

The released final model is `stage08_best.pt`. The stage 09 warm-restart was an
additional continuation that early-stopped without improving on stage 08, so it
is recorded for completeness but not released as the final checkpoint.

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
| `stage08` | `done.json` validation of `best.pt` | 7175.9 | 0.7649 | 0.5295 |
| `stage09` | `done.json` validation of `best.pt` | 1907.6 | 0.7564 | 0.5081 |

Detailed per-epoch CSV files are in `experiments/stages/*/results.csv`.

## Cross-Domain Evaluation

Final checkpoints were validated on each domain's held-out validation set
(mAP50-95). Dataset codes match `DATA_PROVENANCE.md`; the raw table is in
`experiments/cross_domain_mAP50_95.csv`.

| Eval domain | `stage06` | `stage07` | `stage08` (final) | `stage09` |
| --- | ---: | ---: | ---: | ---: |
| `dataset01` | 0.027 | 0.218 | **0.263** | 0.265 |
| `dataset02` | 0.018 | 0.198 | **0.238** | 0.197 |
| `dataset03` | 0.019 | 0.253 | **0.295** | 0.291 |
| `dataset04` | 0.279 | 0.505 | **0.536** | 0.524 |
| `dataset05` | 0.032 | 0.432 | **0.452** | 0.447 |
| `dataset06` | 0.660 | 0.648 | **0.672** | 0.659 |
| `dataset07_merged` (all domains) | 0.204 | 0.491 | **0.529** | 0.508 |

`stage08` is the best checkpoint on 5 of 7 domains and has the highest overall
cross-domain mAP50-95 (0.529), so it is released as the final model. The
`stage09` warm-restart did not improve the aggregate and regressed on several
domains.

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
release-assets/weights/stage08_best.pt   # released final model
release-assets/weights/stage09_best.pt
```

The `release-assets/weights` directory is ignored by Git. Attach these files to
GitHub Releases instead of committing them to repository history.

Before publishing any checkpoint, sanitize embedded training metadata (local
paths and run names) with `scripts/sanitize_checkpoint.py`, then verify with
`scripts/inspect_checkpoint.py`.
