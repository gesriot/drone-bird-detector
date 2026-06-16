"""Evaluate selected checkpoints across anonymized eval sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.example.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    root = resolve(cfg_path.parent, cfg.get("root", ".")).resolve()
    runs_dir = resolve(root, cfg.get("runs_dir", "runs"))
    device = cfg.get("device")

    results: dict[str, dict[str, dict[str, float]]] = {}
    for model_name, model_path_value in cfg["models"].items():
        model_path = resolve(root, model_path_value)
        if not model_path.exists():
            print(f"skip {model_name}: {model_path} missing")
            continue
        model = YOLO(str(model_path))
        results[model_name] = {}
        for dataset_name, spec in cfg["eval_sets"].items():
            data_yaml = resolve(root, spec["data"])
            split = spec.get("split", "val")
            metrics = model.val(
                data=str(data_yaml),
                split=split,
                device=device,
                project=str(runs_dir / "eval"),
                name=f"{model_name}_{dataset_name}",
                exist_ok=True,
                plots=False,
                verbose=False,
            )
            results[model_name][dataset_name] = {
                "mAP50": round(metrics.box.map50, 4),
                "mAP50_95": round(metrics.box.map, 4),
            }
            print(
                f"{model_name:16s} {dataset_name:18s} "
                f"mAP50={metrics.box.map50:.4f} "
                f"mAP50-95={metrics.box.map:.4f}"
            )

    out_dir = runs_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    eval_names = list(cfg["eval_sets"])
    lines = [
        "| model | " + " | ".join(eval_names) + " |",
        "|" + "---|" * (len(eval_names) + 1),
    ]
    for model_name, per_dataset in results.items():
        row = [
            f"{per_dataset[name]['mAP50_95']:.3f}" if name in per_dataset else "-"
            for name in eval_names
        ]
        lines.append(f"| {model_name} | " + " | ".join(row) + " |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
