# Pavement Crack Detection V5

A website and FastAPI backend for running pavement crack detection with a trained YOLO model.

## Project Locations

- Website and backend code: [GitHub repository](https://github.com/hamid989877/pavement-crack-detection-v5)
- Static website preview: [GitHub Pages](https://hamid989877.github.io/pavement-crack-detection-v5/)
- YOLO model only: [Hugging Face model repository](https://huggingface.co/hamid989877/pavement-crack-detection-v5-model)

## Pages

- Home page with project details.
- Image detection page for uploading an image and viewing annotated results.
- Video detection page for uploading a video, viewing live processing, and receiving a detection log.

## Setup

1. Create and activate a Python environment.

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Download the model from Hugging Face.

   ```powershell
   pip install -U huggingface_hub
   hf download hamid989877/pavement-crack-detection-v5-model best.pt --local-dir model
   ```

   The model should be available at:

   ```text
   model/best.pt
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

- GitHub stores the website and backend source.
- Hugging Face stores the YOLO model file.
- Generated videos are written to `outputs/`.
- The API loads the model once and reuses it for image and video requests.
- GitHub Pages is a static UI preview. Real YOLO detection needs the Python backend running locally or on a backend host.
