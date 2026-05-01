# Pavement Crack Detection V5

A small website for presenting a trained YOLO project and running detections with a custom `best.pt` model.

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
- Model files are ignored by git because `.pt` files can be large. Use Git LFS or a release asset if you want to store the model in GitHub.
- The API loads the model once and reuses it for image and video requests.
- GitHub Pages deploys the static UI preview from `static/`. The upload controls need the Python backend to run real YOLO detection.
