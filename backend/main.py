from __future__ import annotations

import base64
import os
import shutil
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    from ultralytics import YOLO
except ModuleNotFoundError:  # pragma: no cover - handled at runtime for clearer setup errors
    YOLO = None


ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "static"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH", ROOT_DIR / "model" / "best.pt"))

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
VIDEO_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-msvideo",
    "video/x-matroska",
    "video/webm",
}

app = FastAPI(
    title="YOLO Vision Lab",
    description="Website and API for image and video detection with a custom YOLO model.",
    version="1.0.0",
)


app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


@lru_cache(maxsize=1)
def load_model() -> Any:
    if YOLO is None:
        raise HTTPException(
            status_code=503,
            detail="Ultralytics is not installed. Run: pip install -r requirements.txt",
        )

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model file not found at {MODEL_PATH}. Put best.pt in the model folder or set YOLO_MODEL_PATH.",
        )

    return YOLO(str(MODEL_PATH))


def page_response(page: str) -> FileResponse:
    path = STATIC_DIR / page
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(path)


def detections_from_result(result: Any) -> list[dict[str, Any]]:
    names = result.names or {}
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return []

    detections: list[dict[str, Any]] = []
    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()

    for index, coords in enumerate(xyxy):
        class_id = int(classes[index])
        detections.append(
            {
                "class_id": class_id,
                "label": str(names.get(class_id, class_id)),
                "confidence": round(float(confidences[index]), 4),
                "box": {
                    "x1": round(float(coords[0]), 2),
                    "y1": round(float(coords[1]), 2),
                    "x2": round(float(coords[2]), 2),
                    "y2": round(float(coords[3]), 2),
                },
            }
        )

    return detections


def encoded_jpeg(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image.")
    payload = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


@app.get("/")
def home() -> FileResponse:
    return page_response("index.html")


@app.get("/index.html")
def home_file() -> FileResponse:
    return page_response("index.html")


@app.get("/image")
def image_page() -> FileResponse:
    return page_response("image.html")


@app.get("/image.html")
def image_file() -> FileResponse:
    return page_response("image.html")


@app.get("/video")
def video_page() -> FileResponse:
    return page_response("video.html")


@app.get("/video.html")
def video_file() -> FileResponse:
    return page_response("video.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_found": MODEL_PATH.exists(),
    }


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    imgsz: int = Query(640, ge=320, le=1280),
) -> dict[str, Any]:
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(status_code=400, detail="Upload a JPG, PNG, WebP, or BMP image.")

    payload = await file.read()
    image_array = np.frombuffer(payload, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="The uploaded image could not be read.")

    model = load_model()
    result = model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)[0]
    annotated = result.plot()
    detections = detections_from_result(result)
    height, width = image.shape[:2]

    return {
        "filename": file.filename,
        "width": width,
        "height": height,
        "detections": detections,
        "count": len(detections),
        "annotated_image": encoded_jpeg(annotated),
    }


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    imgsz: int = Query(640, ge=320, le=1280),
    max_frames: int = Query(0, ge=0, le=20000),
) -> dict[str, Any]:
    if file.content_type not in VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Upload an MP4, MOV, AVI, MKV, or WebM video.")

    model = load_model()
    OUTPUT_DIR.mkdir(exist_ok=True)

    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    output_path = OUTPUT_DIR / f"detection-{job_id}.mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as upload:
        shutil.copyfileobj(file.file, upload)
        input_path = Path(upload.name)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded video could not be read.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    if width <= 0 or height <= 0:
        cap.release()
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded video has invalid dimensions.")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_count = 0
    total_detections = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_count >= max_frames:
                break

            result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
            total_detections += len(detections_from_result(result))
            writer.write(result.plot())
            frame_count += 1
    finally:
        cap.release()
        writer.release()
        input_path.unlink(missing_ok=True)

    if frame_count == 0:
        output_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="No video frames were processed.")

    return {
        "filename": file.filename,
        "frames": frame_count,
        "fps": round(float(fps), 2),
        "width": width,
        "height": height,
        "total_detections": total_detections,
        "video_url": f"/outputs/{output_path.name}",
    }
