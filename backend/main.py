from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .video_processor import process_video
from .detector import inference_runtime

ROOT = Path(__file__).resolve().parents[1]
JOB_ROOT = ROOT / "runs" / "dashboard"
DEMO_VIDEO = ROOT / "assets" / "final.mp4"
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pothole_dashboard")

app = FastAPI(title="Pothole Detection Dashboard", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET", "POST"], allow_headers=["*"])
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


@app.get("/")
def root() -> dict:
    return {"service": "Pothole Detection API", "dashboard": "http://127.0.0.1:5173",
            "health": "/api/health", "docs": "/docs"}


def _run_job(job_id: str, source: Path, display_name: str) -> None:
    output = JOB_ROOT / job_id / "processed.mp4"
    try:
        def update(percent: int, message: str) -> None:
            with jobs_lock:
                jobs[job_id].update(progress=percent, message=message)
        with jobs_lock:
            jobs[job_id].update(status="processing", message="Loading YOLO model", progress=0)
        result = process_video(source, output, update)
        result["video"] = display_name
        (output.parent / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        with jobs_lock:
            jobs[job_id].update(status="completed", progress=100, message="Processing complete", result=result)
    except Exception as exc:
        logger.exception("Video job %s failed", job_id)
        with jobs_lock:
            jobs[job_id].update(status="failed", message=str(exc), error=str(exc))


def _create_job(source: Path, display_name: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with jobs_lock:
        jobs[job_id] = {"job_id": job_id, "status": "queued", "progress": 0, "message": "Queued for processing"}
    threading.Thread(target=_run_job, args=(job_id, source, display_name), daemon=True).start()
    return job_id


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "model_available": (ROOT / "models" / "best.pt").is_file(),
            "demo_available": DEMO_VIDEO.is_file(), **inference_runtime()}


@app.post("/api/process")
async def upload_video(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "").name
    if not name.casefold().endswith(".mp4"):
        raise HTTPException(415, "Unsupported file. Please upload an MP4 video.")
    job_id = uuid.uuid4().hex[:12]
    folder = JOB_ROOT / job_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "input.mp4"
    size = 0
    try:
        with target.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Video exceeds the 500 MB upload limit.")
                destination.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        shutil.rmtree(folder, ignore_errors=True)
        raise
    if size == 0:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, "The uploaded video is empty.")
    with jobs_lock:
        jobs[job_id] = {"job_id": job_id, "status": "queued", "progress": 0, "message": "Queued for processing"}
    threading.Thread(target=_run_job, args=(job_id, target, name), daemon=True).start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/process/demo")
def process_demo() -> dict:
    if not DEMO_VIDEO.is_file():
        raise HTTPException(404, "Demo video assets/final.mp4 was not found.")
    return {"job_id": _create_job(DEMO_VIDEO, DEMO_VIDEO.name), "status": "queued"}


@app.get("/api/status/{job_id}")
def get_status(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Processing job not found.")
        return {key: value for key, value in job.items() if key not in {"result", "error"}} | ({"error": job["error"]} if "error" in job else {})


@app.get("/api/results/{job_id}")
def get_results(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Processing job not found.")
        if job["status"] == "failed":
            raise HTTPException(500, job.get("error", "Video processing failed."))
        if job["status"] != "completed":
            raise HTTPException(409, "Video processing is still in progress.")
        return job["result"]


@app.get("/api/video/{job_id}")
def get_video(job_id: str) -> FileResponse:
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Processing job not found.")
        if job["status"] != "completed":
            raise HTTPException(409, "Processed video is not ready yet.")
    path = JOB_ROOT / job_id / "processed.mp4"
    if not path.is_file():
        raise HTTPException(404, "Processed video file was not found.")
    return FileResponse(path, media_type="video/mp4", filename="processed.mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
