# Weight Release Assets

This directory is a local staging area for model weights that should be attached
to GitHub Releases, not committed to normal Git history.

Expected local files:

```text
stage01_best.pt
stage02_best.pt
stage03_best.pt
stage04_best.pt
stage05_best.pt
stage06_best.pt
stage07_best.pt
stage08_best.pt  # after warm-restart completion
```

Run this before publishing a checkpoint:

```bash
python scripts/inspect_checkpoint.py release-assets/weights/stage07_best.pt
```

Checksums for the local staged files are recorded in `SHA256SUMS`.
