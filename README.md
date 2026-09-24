# Roadwatch — pothole video analysis

Roadwatch analyzes **recorded road videos** with the project's trained YOLOv8 pothole model. It detects lane markings, groups repeated detections of the same pothole, and shows which detections fall in the camera vehicle's lane. The app starts with a minimal upload/results page; a separate detailed dashboard shows the event log, lane diagnostics, and alert timeline.

> **This is an offline video-analysis prototype, not a live in-car warning system.** The app processes the full uploaded clip before showing its annotated video and results. Playback warnings are synchronized to that processed clip; they are not issued while a car is driving.

## What it shows

- Annotated MP4 with pothole boxes and the estimated current-lane region.
- Counts for potholes in the current lane and outside it, plus a tracked-event log in the detailed view.
- An amber **Pothole ahead** message during playback when a tracked current-lane pothole was repeatedly detected farther up the frame.
- Local GPU inference when a compatible CUDA-enabled PyTorch installation is available; CPU is the fallback.
- A visual risk estimate and a DAY/NIGHT scene indicator in the detailed view.

The amber warning uses image position, **not measured distance**. Speed, pothole depth, and stopping time are not measured by this project.

## Requirements

- Python 3.11, Node.js/npm, FFmpeg and FFprobe on `PATH`.
- The trained pothole weights at **`models/best.pt`**, included in this repository. Generic YOLO weights will not detect this project's pothole class.
- A CUDA-enabled PyTorch installation for NVIDIA GPU inference, or a CPU PyTorch installation.

Windows installation instructions for both GPU and CPU are in [SETUP_WINDOWS.md](SETUP_WINDOWS.md). The GPU instructions are tailored to the RTX 50-series/CUDA 12.8 environment used during development. On other systems, install a PyTorch build appropriate to your hardware, then install the Python packages required by the backend.

## Start the app

From the repository root, after following the platform setup instructions:

```powershell
.\.venv\Scripts\python.exe -m backend.main
```

In a second terminal:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

Open **http://127.0.0.1:5173**. On macOS/Linux, activate your Python environment and use `python -m backend.main`, `npm ci`, and `npm run dev` instead. The API runs at **http://127.0.0.1:8000**; `/api/health` reports whether the model and bundled demo video are present and which inference device was selected.

## Use it

1. Choose an MP4 video or select **Try the demo** to process `assets/final.mp4`. You can also drop an MP4 onto the page. The upload limit is 500 MB.
2. Wait for processing to finish. The result page then shows the annotated video and simple counts. A warning appears over the video at qualifying playback times.
3. Open **Detailed dashboard** for lane confidence, visual risk estimates, the tracked-event log, and a list of warning times. Its **Presentation view** button returns to the minimal page. Switching views keeps the current playback position.

Processed inputs, annotated videos, and JSON results are written under `runs/dashboard/<job-id>/`. This directory is ignored by Git. The bundled demo clip is tracked in `assets/final.mp4`, and the trained model is tracked in `models/best.pt`. Local test videos are not included.

## How it works

```text
Uploaded MP4
   ↓ FFmpeg frame decoding
YOLOv8 pothole detection + OpenCV lane estimation
   ↓ lane association and centroid tracking
Annotated MP4 + JSON results + playback warning intervals
   ↓
React results page / detailed dashboard
```

The backend processes frames in order. YOLO detects candidate potholes at a 0.25 confidence threshold; OpenCV estimates lane boundaries from visible markings. The bottom-center of each pothole box is compared with the current-lane region. A lightweight tracker groups detections across adjacent frames into events. When lane confidence is insufficient, detections are marked uncertain instead of being assigned to the current lane.

An early warning interval requires at least two observations of a tracked pothole in the current lane, detection confidence of at least 0.35, and a box bottom above 72% of image height. This is only a visual “farther up the frame” rule. It does not estimate meters to the pothole. The detailed dashboard shows a visual risk score from box area, confidence, and image position; it does not measure physical severity or depth.

YOLO runs on the selected CUDA device when available. FFmpeg decoding/encoding and OpenCV lane estimation still run on CPU. Set `YOLO_DEVICE=cpu` to force CPU or `YOLO_DEVICE=cuda:0` to require the first GPU. The default is `auto`. See [SETUP_WINDOWS.md](SETUP_WINDOWS.md) for PowerShell examples.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Model/demo availability and inference device |
| `POST /api/process` | Upload an MP4 and start a job |
| `POST /api/process/demo` | Start a job with the bundled demo clip |
| `GET /api/status/{job_id}` | Processing state and progress |
| `GET /api/results/{job_id}` | Counts, events, lane data, and `forward_alerts` |
| `GET /api/video/{job_id}` | Stream the completed annotated MP4 |

## Checks

With the Python environment installed:

```powershell
.\.venv\Scripts\python.exe -m unittest test_forward_alerts test_lane_tracking test_inference_device -v
.\.venv\Scripts\python.exe test_lane_pipeline.py
cd frontend
npm.cmd run build
```

The lane-pipeline smoke check loads `models/best.pt` and needs the bundled demo video. The unit tests cover warning intervals, lane stability, and inference-device selection.

## Limitations

- This is **batch processing**. It does not read a live vehicle camera, measure end-to-end warning latency, or provide a driving-safety guarantee.
- The pothole model and lane thresholds were tested on limited footage. Faded markings, curves, shadows, rain, night scenes, camera motion, and different road surfaces can reduce accuracy. Counts may include false detections.
- Lane association is based on a box contact point and estimated lane geometry. Tracking can split or merge nearby potholes.
- “Pothole ahead” is based on image height and cannot determine actual distance or time to contact. The risk score is visual only. There is no calibrated speed, physical pothole depth, or road-surface height estimate.
- Job state is held in memory and is lost when the backend restarts. Files under `runs/dashboard/` remain on disk.

The original inference notebook and project report remain in the repository for reference. The dashboard uses the existing trained weights without retraining them.
