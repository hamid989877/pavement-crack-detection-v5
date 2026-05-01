---
title: Pavement Crack Detection V5
sdk: docker
app_port: 7860
pinned: false
---

# Pavement Crack Detection V5

A small website for presenting a trained YOLO project and running detections with a custom `best.pt` model.

## Live Website

[Open the full detection app on Hugging Face Spaces](https://hamid989877-pavement-crack-detection-v5.hf.space)

## Pages

- Project page with project details and an animated detection preview.
- Image detection page for uploading an image and viewing annotated results.
- Video detection page for uploading a video and receiving a processed video.

## Setup

1. Put your YOLO model at:

   ```text
   model/best.pt
   ```

   You can also keep the model somewhere else and set `YOLO_MODEL_PATH`.

2. Create and activate a Python environment.

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

4. Start the website.

   ```powershell
   uvicorn backend.main:app --reload
   ```

5. Open:

   ```text
   http://127.0.0.1:8000
   ```

## Notes

- Generated videos are written to `outputs/`.
- Model files are ignored by normal git because `.pt` files can be large. Use Git LFS for Hugging Face Spaces.
- The API loads the model once and reuses it for image and video requests.
- GitHub Pages deploys the static UI preview from `static/`. The upload controls need the Python backend to run real YOLO detection.

## Hugging Face Spaces

This project is ready for a Docker Space. The Space must include the model file at:

```text
model/best.pt
```

The app runs on port `7860`, which is the default port used by Docker Spaces.
