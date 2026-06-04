# Training Version Archive

This folder stores the non-model artifacts for the three dataset/training versions used in Pavement Crack Detection V5.

The original version folders contained YOLO training runs with:

- `args.yaml` training configuration files
- `results.csv` metric logs
- confusion matrices and result plots
- validation batch images
- version screenshots
- `best.pt` and `last.pt` model checkpoints

Only the non-model artifacts are stored in GitHub. The `.pt` checkpoints are stored on Hugging Face:

```text
https://huggingface.co/hamid989877/pavement-crack-detection-v5-model
```

## Checkpoint Paths On Hugging Face

```text
versions/version-1/run-1/detect/train/weights/best.pt
versions/version-1/run-1/detect/train/weights/last.pt
versions/version-1/run-2/detect/train/weights/best.pt
versions/version-1/run-2/detect/train/weights/last.pt
versions/version-1/run-3/detect/train/weights/best.pt
versions/version-1/run-3/detect/train/weights/last.pt
versions/version-1/run-4/detect/train/weights/best.pt
versions/version-1/run-4/detect/train/weights/last.pt
versions/version-1/run-5/detect/train/weights/best.pt
versions/version-1/run-5/detect/train/weights/last.pt
versions/version-1/run-6/detect/train/weights/best.pt
versions/version-1/run-6/detect/train/weights/last.pt
versions/version-2/run-1/detect/train/weights/best.pt
versions/version-2/run-1/detect/train/weights/last.pt
versions/version-3/run-1/detect/train/weights/best.pt
versions/version-3/run-1/detect/train/weights/last.pt
```

Download all versioned checkpoints:

```powershell
hf download hamid989877/pavement-crack-detection-v5-model --include "versions/**/*.pt" --local-dir .
```

## Final Epoch Metrics

| Version | Run | Base model | Epochs | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---:|---:|---:|---:|---:|
| `version-1` | `run-1` | `yolo11m.pt` | 50 | 0.8940 | 0.5254 | 0.8331 | 0.8283 |
| `version-1` | `run-2` | `yolo26m.pt` | 50 | 0.8813 | 0.5123 | 0.8201 | 0.8337 |
| `version-1` | `run-3` | `yolo26x.pt` | 50 | 0.8804 | 0.5151 | 0.8401 | 0.8104 |
| `version-1` | `run-4` | `yolo11x.pt` | 50 | 0.8821 | 0.5157 | 0.8215 | 0.8311 |
| `version-1` | `run-5` | `yolo11n.pt` | 50 | 0.8574 | 0.4923 | 0.7851 | 0.8069 |
| `version-1` | `run-6` | `yolov8m.pt` | 50 | 0.8844 | 0.5164 | 0.8294 | 0.8111 |
| `version-2` | `run-1` | `yolo26m.pt` | 50 | 0.8453 | 0.4952 | 0.8461 | 0.8014 |
| `version-3` | `run-1` | `yolo26x.pt` | 50 | 0.8695 | 0.5075 | 0.8317 | 0.8286 |
