from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
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

VIDEO_JOBS: dict[str, dict[str, Any]] = {}
VIDEO_JOBS_LOCK = threading.Lock()


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


def jpeg_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return b""
    return buffer.tobytes()


def draw_detections_on_frame(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        box = detection["box"]
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])
        label = f"{detection['label']} {detection['confidence']:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (55, 58, 64), 2)
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1, label_size[1] + baseline + 6)
        cv2.rectangle(
            annotated,
            (x1, label_y - label_size[1] - baseline - 6),
            (x1 + label_size[0] + 8, label_y + baseline - 2),
            (55, 58, 64),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 4, label_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def update_video_job(job_id: str, **values: Any) -> None:
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if not job:
            return
        job.update(values)
        job["updated_at"] = time.time()


def video_job_payload(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    total_frames = int(job.get("total_frames") or 0)
    frames = int(job.get("frames") or 0)
    progress = round(frames / total_frames, 4) if total_frames else 0
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "filename": job.get("filename"),
        "frames": frames,
        "analyzed_frames": int(job.get("analyzed_frames") or 0),
        "total_frames": total_frames,
        "progress": min(progress, 1),
        "fps": job.get("fps"),
        "width": job.get("width"),
        "height": job.get("height"),
        "total_detections": int(job.get("total_detections") or 0),
        "video_url": job.get("video_url"),
        "stream_url": f"/api/detect/video/{job_id}/stream",
        "error": job.get("error"),
    }


def require_video_job(job_id: str) -> dict[str, Any]:
    with VIDEO_JOBS_LOCK:
        job = VIDEO_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Video detection job not found.")
        return dict(job)


def transcode_for_browser(raw_path: Path, output_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raw_path.replace(output_path)
        return output_path

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(raw_path),
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=300)
        raw_path.unlink(missing_ok=True)
    except (subprocess.SubprocessError, OSError):
        raw_path.replace(output_path)

    return output_path


def process_video_job(
    job_id: str,
    input_path: Path,
    raw_output_path: Path,
    output_path: Path,
    conf: float,
    imgsz: int,
    max_frames: int,
    frame_stride: int,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    writer: cv2.VideoWriter | None = None
    frame_count = 0
    analyzed_frames = 0
    total_detections = 0
    last_detections: list[dict[str, Any]] | None = None

    try:
        if not cap.isOpened():
            raise ValueError("The uploaded video could not be read.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        if width <= 0 or height <= 0:
            raise ValueError("The uploaded video has invalid dimensions.")

        writer = cv2.VideoWriter(
            str(raw_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise ValueError("Could not start the output video recorder.")

        model = load_model()
        update_video_job(job_id, status="processing")

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_count >= max_frames:
                break

            should_analyze = frame_count % frame_stride == 0
            if should_analyze or last_detections is None:
                result = model.predict(source=frame, conf=conf, imgsz=imgsz, verbose=False)[0]
                detections = detections_from_result(result)
                total_detections += len(detections)
                analyzed_frames += 1
                annotated = result.plot()
                last_detections = detections
            else:
                annotated = draw_detections_on_frame(frame, last_detections)

            writer.write(annotated)
            frame_count += 1

            latest_frame = jpeg_bytes(annotated)
            update_video_job(
                job_id,
                frames=frame_count,
                analyzed_frames=analyzed_frames,
                total_detections=total_detections,
                latest_frame=latest_frame,
            )

        if frame_count == 0:
            raise ValueError("No video frames were processed.")

        if writer:
            writer.release()
            writer = None

        transcode_for_browser(raw_output_path, output_path)
        update_video_job(
            job_id,
            status="completed",
            frames=frame_count,
            analyzed_frames=analyzed_frames,
            total_detections=total_detections,
            video_url=f"/outputs/{output_path.name}",
        )
    except Exception as exc:  # noqa: BLE001 - user-facing job errors are captured in status
        raw_output_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        update_video_job(job_id, status="error", error=str(exc))
    finally:
        cap.release()
        if writer:
            writer.release()
        input_path.unlink(missing_ok=True)


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
    frame_stride: int = Query(2, ge=1, le=10),
) -> dict[str, Any]:
    if file.content_type not in VIDEO_TYPES:
        raise HTTPException(status_code=400, detail="Upload an MP4, MOV, AVI, MKV, or WebM video.")

    load_model()
    OUTPUT_DIR.mkdir(exist_ok=True)

    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    raw_output_path = OUTPUT_DIR / f"detection-{job_id}-raw.mp4"
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded video has invalid dimensions.")

    cap.release()
    expected_frames = min(total_frames, max_frames) if max_frames and total_frames else total_frames
    with VIDEO_JOBS_LOCK:
        VIDEO_JOBS[job_id] = {
            "status": "queued",
            "filename": file.filename,
            "frames": 0,
            "analyzed_frames": 0,
            "total_frames": expected_frames,
            "fps": round(float(fps), 2),
            "width": width,
            "height": height,
            "total_detections": 0,
            "video_url": None,
            "latest_frame": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    worker = threading.Thread(
        target=process_video_job,
        args=(job_id, input_path, raw_output_path, output_path, conf, imgsz, max_frames, frame_stride),
        daemon=True,
    )
    worker.start()

    return {
        "job_id": job_id,
        "filename": file.filename,
        "status": "queued",
        "frames": 0,
        "analyzed_frames": 0,
        "total_frames": expected_frames,
        "fps": round(float(fps), 2),
        "width": width,
        "height": height,
        "total_detections": 0,
        "stream_url": f"/api/detect/video/{job_id}/stream",
        "status_url": f"/api/detect/video/{job_id}",
        "video_url": None,
    }


@app.get("/api/detect/video/{job_id}")
def video_job_status(job_id: str) -> dict[str, Any]:
    job = require_video_job(job_id)
    return video_job_payload(job_id, job)


@app.get("/api/detect/video/{job_id}/stream")
def video_job_stream(job_id: str) -> StreamingResponse:
    require_video_job(job_id)

    def stream_frames() -> Any:
        last_frame: bytes | None = None
        while True:
            with VIDEO_JOBS_LOCK:
                job = VIDEO_JOBS.get(job_id)
                if not job:
                    break
                frame = job.get("latest_frame")
                status = job.get("status")

            if frame and frame != last_frame:
                last_frame = frame
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-cache\r\n\r\n" + frame + b"\r\n"
                )

            if status in {"completed", "error"}:
                break

            time.sleep(0.18)

    return StreamingResponse(
        stream_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
