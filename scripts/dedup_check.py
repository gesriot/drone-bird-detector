"""Perceptual duplicate check between candidate train images and eval images."""

from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS]


def dhash(path: Path) -> tuple[str, int | None]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return str(path), None
    image = cv2.resize(image, (9, 8))
    bits = 0
    for row in range(8):
        for col in range(8):
            bits = (bits << 1) | (1 if image[row, col] > image[row, col + 1] else 0)
    return str(path), bits


def hash_all(paths: list[Path], workers: int) -> list[tuple[str, int | None]]:
    with Pool(workers) as pool:
        return pool.map(dhash, paths, chunksize=64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", nargs="+", required=True, help="Eval image dirs or .txt lists")
    parser.add_argument("--candidate-train", nargs="+", required=True, help="Candidate train dirs or .txt lists")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    eval_paths = [p for source in args.eval for p in iter_images(Path(source))]
    train_paths = [p for source in args.candidate_train for p in iter_images(Path(source))]

    eval_hashes: dict[int, list[str]] = {}
    for path, hash_value in hash_all(eval_paths, args.workers):
        if hash_value is not None:
            eval_hashes.setdefault(hash_value, []).append(path)

    hits = []
    for path, hash_value in hash_all(train_paths, args.workers):
        if hash_value is not None and hash_value in eval_hashes:
            hits.append(f"{path}\t{eval_hashes[hash_value][0]}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(hits) + ("\n" if hits else ""), encoding="utf-8")
    print(f"eval images: {len(eval_paths)}")
    print(f"candidate train images: {len(train_paths)}")
    print(f"collisions: {len(hits)} -> {output}")


if __name__ == "__main__":
    main()
