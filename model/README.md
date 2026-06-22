# Released Model

Two-class aerial object detector, fine-tuned from the Ultralytics YOLO26s base
model. This is the `stage08` checkpoint (the released final model; see
`MODEL_CARD.md`).

```text
0 = bird
1 = airplane   # includes airplanes, helicopters, UAVs and drones
```

Validation metrics (stage 08, 640x640): mAP50 0.7649, mAP50-95 0.5295.

## Formats

| File | Format | Use |
| --- | --- | --- |
| `yolo26s_drone_bird.pt` | Ultralytics PyTorch | `ultralytics` Python API, further fine-tuning |
| `yolo26s_drone_bird.onnx` | ONNX (opset 20) | ONNX Runtime, OpenCV DNN, cross-framework |
| `yolo26s_drone_bird.torchscript` | TorchScript | LibTorch / `tch` (e.g. the Rust pipeline) |

All three are exported at `imgsz=640` and share the same weights. `SHA256SUMS`
records their checksums.

## Privacy

Training metadata in the checkpoint (local paths, run names, git root) was
stripped with `scripts/sanitize_checkpoint.py` before export. Only the public
class scheme and standard Ultralytics metadata remain.

## License

The weights derive from the Ultralytics YOLO model family and are distributed
under **AGPL-3.0** (`LICENSE-Ultralytics-AGPL-3.0.txt`), not the repository's MIT
license. See `../NOTICE.md` and `../DATA_PROVENANCE.md`. Training datasets are
not redistributed.
