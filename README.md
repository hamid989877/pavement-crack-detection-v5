# Pavement Crack Detection V5

Pavement Crack Detection V5 is a YOLO-based computer vision project for detecting road-surface damage from images and videos. The project combines training artifacts from several dataset/model versions with a web interface that lets a user upload an image or video, run inference, and review annotated results.

The repository is intentionally split across two platforms:

- GitHub stores the website, backend code, documentation, and non-model training artifacts.
- Hugging Face stores the YOLO checkpoint files, because `.pt` model weights are large.

## Links

- GitHub repository: [hamid989877/pavement-crack-detection-v5](https://github.com/hamid989877/pavement-crack-detection-v5)
- Static GitHub Pages preview: [pavement-crack-detection-v5](https://hamid989877.github.io/pavement-crack-detection-v5/)
- Hugging Face model repository: [pavement-crack-detection-v5-model](https://huggingface.co/hamid989877/pavement-crack-detection-v5-model)

## What The Project Does

The application is designed around a trained Ultralytics YOLO model for road damage detection. It supports:

- A home page explaining the project and showing a pavement-crack themed interface.
- Image detection with uploaded JPG, PNG, WebP, or BMP files.
- Video detection with uploaded MP4, MOV, AVI, MKV, or WebM files.
- Live video processing preview while inference is running.
- A saved processed video with bounding boxes.
- A detection log for video runs, including timestamp, frame number, crack type, confidence, and bounding box coordinates.

The model classes used by the app are:

| Class ID | Label |
|---:|---|
| 0 | Alligator Crack |
| 1 | Longitudinal Crack |
| 2 | Pothole |
| 3 | Transverse Crack |

## How We Built It

1. Dataset and training versions were organized as `version-1`, `version-2`, and `version-3`.
2. Each version contains one or more YOLO training runs with configuration files, result CSVs, validation images, plots, and confusion matrices.
3. Large model checkpoints (`best.pt` and `last.pt`) were moved to Hugging Face under versioned paths.
4. Non-model training artifacts were added to GitHub under `versions/` so the project history can be reviewed without storing huge checkpoints in Git.
5. A FastAPI backend was created to load `model/best.pt` once and reuse it for image and video inference.
6. Static HTML, CSS, and JavaScript pages were added for the home page, image upload workflow, and video upload workflow.
7. Video inference was upgraded from a long blocking request to a background job with progress polling, live MJPEG preview, saved output video, and a detection-detail table.
8. The UI was styled around an asphalt/crack color palette using dark gray, amber/yellow highlights, white text, and red error states.

## Repository Layout

```text
backend/
  main.py                  FastAPI app and YOLO inference endpoints

static/
  index.html               Home/project page
  image.html               Image detection page
  video.html               Video detection page
  assets/                  CSS, JavaScript, and visual assets

versions/
  version-1/               Non-model artifacts from version 1 training runs
  version-2/               Non-model artifacts from version 2 training runs
  version-3/               Non-model artifacts from version 3 training runs

model/
  .gitkeep                 Placeholder only; download best.pt from Hugging Face

outputs/
  .gitkeep                 Generated detection videos are written here locally
```

## Dataset And Training Versions

The `versions/` folder contains the training outputs that are safe to keep in GitHub: result CSVs, YAML arguments, plots, validation examples, and version screenshots. Model checkpoints are stored separately on Hugging Face.

| Version | Runs | Non-model artifacts in GitHub | Checkpoints in Hugging Face |
|---|---:|---|---|
| `version-1` | 6 | Training plots, CSVs, YAML configs, validation images | `versions/version-1/.../weights/best.pt` and `last.pt` |
| `version-2` | 1 | Training plots, CSVs, YAML configs, validation images | `versions/version-2/.../weights/best.pt` and `last.pt` |
| `version-3` | 1 | Training plots, CSVs, YAML configs, validation images | `versions/version-3/.../weights/best.pt` and `last.pt` |

Training metric summary from the final logged epoch of each run:

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

## Model Storage

The active app model and versioned checkpoints are stored in Hugging Face:

```text
https://huggingface.co/hamid989877/pavement-crack-detection-v5-model
```

The root `best.pt` is the model expected by the local app. Versioned checkpoints are available under:

```text
versions/version-1/run-*/detect/train/weights/
versions/version-2/run-1/detect/train/weights/
versions/version-3/run-1/detect/train/weights/
```

Download the active app model:

```powershell
hf download hamid989877/pavement-crack-detection-v5-model best.pt --local-dir model
```

Download all versioned checkpoints:

```powershell
hf download hamid989877/pavement-crack-detection-v5-model --include "versions/**/*.pt" --local-dir .
```

## Run Locally

1. Create and activate a virtual environment.

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```powershell
   pip install -r requirements.txt
   pip install -U huggingface_hub
   ```

3. Download the active model from Hugging Face.

   ```powershell
   hf download hamid989877/pavement-crack-detection-v5-model best.pt --local-dir model
   ```

4. Start the backend and website.

   ```powershell
   uvicorn backend.main:app --reload
   ```

5. Open the site.

   ```text
   http://127.0.0.1:8000
   ```

## API Overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Confirms the backend is running and the model file exists |
| `/api/detect/image` | POST | Runs YOLO detection on one uploaded image |
| `/api/detect/video` | POST | Starts a background video detection job |
| `/api/detect/video/{job_id}` | GET | Returns video job progress, output URL, and detection log |
| `/api/detect/video/{job_id}/stream` | GET | Streams live annotated frames while a video job is running |

## Notes

- GitHub Pages is a static preview, so upload detection does not run there by itself.
- Real detection needs the FastAPI backend running locally or on a backend host.
- Generated videos are saved in `outputs/` during local development.
- Large model formats are ignored by `.gitignore` and should stay in Hugging Face.
