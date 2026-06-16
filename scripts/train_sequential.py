"""Run a resumable sequential detector fine-tuning chain.

The configuration file contains anonymized stage and dataset codes. Each stage
starts from the previous stage's best checkpoint. Completed stages are skipped
when both weights/best.pt and done.json are present.
"""

from __future__ import annotations

import argparse
import json
import time
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
    parser.add_argument("--config", default="configs/stages.local.yaml")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    root = resolve(cfg_path.parent, cfg.get("root", ".")).resolve()
    runs_dir = resolve(root, cfg.get("runs_dir", "runs"))
    common = dict(cfg.get("common", {}))
    common["project"] = str(runs_dir)

    weights = str(resolve(root, cfg["initial_weights"]))
    for stage in cfg["stages"]:
        stage_id = stage["id"]
        data_yaml = resolve(root, stage["data"])
        stage_dir = runs_dir / stage_id
        best = stage_dir / "weights" / "best.pt"
        marker = stage_dir / "done.json"

        if marker.exists() and best.exists():
            print(f"=== {stage_id}: already done, skipping ===")
            weights = str(best)
            continue

        print(f"=== {stage_id}: training from {weights} ===", flush=True)
        t0 = time.time()
        model = YOLO(weights)
        model.train(
            data=str(data_yaml),
            epochs=int(stage["epochs"]),
            patience=int(stage["patience"]),
            name=stage_id,
            **common,
        )

        if not best.exists():
            raise FileNotFoundError(f"{best} missing after training")

        metrics = YOLO(str(best)).val(
            data=str(data_yaml),
            device=common.get("device"),
            project=str(runs_dir),
            name=f"{stage_id}_val",
            exist_ok=True,
        )
        marker.write_text(
            json.dumps(
                {
                    "minutes": round((time.time() - t0) / 60, 1),
                    "mAP50": round(metrics.box.map50, 4),
                    "mAP50_95": round(metrics.box.map, 4),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"=== {stage_id} done: mAP50={metrics.box.map50:.4f} "
            f"mAP50-95={metrics.box.map:.4f} ===",
            flush=True,
        )
        weights = str(best)

    print("ALL STAGES COMPLETE. Final weights:", weights, flush=True)


if __name__ == "__main__":
    main()
