"""Inspect a PyTorch checkpoint for metadata before publishing it."""

from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--keys", action="store_true", help="Only print top-level keys")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        print(type(checkpoint))
        return

    print("top-level keys:")
    pprint(sorted(checkpoint.keys()))
    if args.keys:
        return

    interesting = [
        "args",
        "train_args",
        "date",
        "version",
        "license",
        "docs",
        "epoch",
        "best_fitness",
    ]
    for key in interesting:
        if key in checkpoint:
            print(f"\n[{key}]")
            pprint(checkpoint[key])


if __name__ == "__main__":
    main()
