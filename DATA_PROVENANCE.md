# Data Provenance

This public repository does not redistribute source datasets, images, labels, or
derived image crops. Dataset names and URLs are intentionally abstracted in this
public file. A private source ledger should be maintained outside Git as
`SOURCE_LEDGER.private.md` and should map each dataset code below to the exact
source URL, version, download date, license text, checksums, and citation.

## Unified Label Scheme

All training data is normalized to:

```text
0 = bird
1 = airplane
```

The `airplane` class intentionally includes airplanes, helicopters, UAVs, and
drones so the detector remains compatible with the base taxonomy used by the
upstream model family.

## Public Dataset Codes

| Code | Public description | Training role | Redistribution posture |
| --- | --- | --- | --- |
| `dataset01` | Restricted-access long-range drone imagery | Drone-only warm-up | Do not redistribute |
| `dataset02` | Public multi-sensor aerial video dataset | Mixed aerial objects, visible/IR | Prefer source link; do not mirror by default |
| `dataset03` | Small public drone/bird object detection set | Bird/drone balancing | Do not redistribute without verified source license |
| `dataset04` | Small public drone/bird object detection set | Bird/drone balancing | Do not redistribute without verified source license |
| `dataset05` | Medium public drone/bird object detection set | Bird/drone balancing | Do not redistribute without verified source license |
| `dataset06` | Large public drone/bird object detection set | Final balanced public domain stage | Do not redistribute without verified source license |
| `dataset07a` | Public drone/bird YOLO-format dataset | Merged-stage data expansion | Redistribute only if complying with database/content terms |
| `dataset07b` | Public YOLO-format bird/drone dataset | Train-only expansion due to pre-augmentation | Redistribute only with required attribution |

## Split and Leakage Controls

- Video-like drone-only data is split by complete sequence/session rather than
  random frame to reduce near-duplicate leakage.
- One drone-only dataset is downsampled during training because adjacent frames
  are highly redundant.
- The merged stage uses a deterministic block split for sequential frames.
- Pre-augmented data is used only in train where appropriate.
- New training candidates are checked against existing validation/test images
  with a perceptual hash duplicate scan.

## Public Release Policy

Allowed in this repository:

- preparation and training scripts
- anonymized dataset codes and counts
- metrics and model cards
- release assets for trained weights

Not allowed in this repository:

- source dataset archives
- images, labels, videos, or extracted frames
- path lists containing private absolute paths
- third-party dataset README/license files unless redistribution is permitted

For any public model-weight release, keep a private source ledger and comply
with the original dataset licenses and attribution requirements.
