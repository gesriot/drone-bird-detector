"""Strip local/private training metadata from a checkpoint before publishing.

Ultralytics checkpoints embed the local training environment: absolute dataset
and weight paths, the run name, and a `git` block with the local repository root.
This rewrites those fields to non-identifying values without touching the model
weights, so a published checkpoint reveals only the public class scheme.

    python scripts/sanitize_checkpoint.py final.pt --output release.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

# Absolute-path prefixes that indicate a local, private filesystem location.
PRIVATE_PREFIXES = ("/Users/", "/home/", "/root/", "C:\\", "\\\\")


def scrub_path(value: object) -> object:
    """Replace an absolute local path with just its file name."""
    if isinstance(value, str) and any(prefix in value for prefix in PRIVATE_PREFIXES):
        return Path(value.replace("\\", "/")).name
    return value


def sanitize_args(args: object) -> None:
    """Scrub a training-args mapping in place (dict or namespace)."""
    mapping = args if isinstance(args, dict) else getattr(args, "__dict__", None)
    if not isinstance(mapping, dict):
        return
    # The run name and save directory encode internal stage naming; neutralize
    # them rather than leaking the basename of an internal run directory.
    if "name" in mapping:
        mapping["name"] = "train"
    if "save_dir" in mapping:
        mapping["save_dir"] = "runs/train"
    # Reduce every remaining absolute path (data, model, project, ...) to a bare
    # file name so no local directory layout leaks.
    for key, value in list(mapping.items()):
        if key in ("name", "save_dir"):
            continue
        mapping[key] = scrub_path(value)


def sanitize(checkpoint: dict) -> dict:
    # The git block carries the local repository root path; drop it entirely.
    checkpoint.pop("git", None)
    # The same training args live in three places: the top-level dict and the
    # `.args` of the embedded model and EMA modules.
    sanitize_args(checkpoint.get("train_args"))
    for key in ("model", "ema"):
        obj = checkpoint.get(key)
        if obj is not None and hasattr(obj, "args"):
            sanitize_args(obj.args)
    return checkpoint


def remaining_private_strings(checkpoint: dict) -> list[str]:
    found: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, str):
            if any(prefix in value for prefix in PRIVATE_PREFIXES):
                found.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)

    for key in ("git", "train_args", "train_metrics", "train_results", "date", "docs"):
        walk(checkpoint.get(key))
    for key in ("model", "ema"):
        obj = checkpoint.get(key)
        if obj is not None and hasattr(obj, "args"):
            args = obj.args
            walk(args if isinstance(args, dict) else getattr(args, "__dict__", {}))
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True, help="path for the sanitized copy")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"unexpected checkpoint format: {type(checkpoint)}")

    checkpoint = sanitize(checkpoint)
    leftover = remaining_private_strings(checkpoint)
    if leftover:
        raise SystemExit("private strings still present after sanitization:\n" + "\n".join(leftover))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"sanitized -> {output}")


if __name__ == "__main__":
    main()
