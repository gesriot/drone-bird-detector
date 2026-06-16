"""Build a YOLO data.yaml backed by train.txt and val.txt path lists.

This helper is intentionally generic. Use it for merged/private datasets where
the images remain in their original locations and should not be copied into the
repository.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(path: Path) -> list[str]:
    if path.is_file():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(str(p) for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", nargs="+", required=True, help="Image dirs or existing .txt lists")
    parser.add_argument("--val", nargs="+", required=True, help="Image dirs or existing .txt lists")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train = [p for source in args.train for p in iter_images(Path(source))]
    val = [p for source in args.val for p in iter_images(Path(source))]

    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(val)

    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "train.txt",
                "val": "val.txt",
                "nc": 2,
                "names": {0: "bird", 1: "airplane"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"train: {len(train)}, val: {len(val)}")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
