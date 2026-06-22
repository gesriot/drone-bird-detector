# Stage 09

Additional warm-restart continuation from `stage08_best.pt` (30 epochs max,
patience 8). Early-stopped at epoch 9.

Status: complete. This continuation did **not** improve on stage 08
(mAP50 0.7564 vs 0.7649, mAP50-95 0.5081 vs 0.5295), so the released final
model remains `stage08_best.pt`.

Artifacts:

```text
results.csv
done.json
release-assets/weights/stage09_best.pt
```
