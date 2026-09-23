# Roadwatch — YOLOv8 Pothole Detection Dashboard

A local, CPU-based road video analysis dashboard built around the project's existing trained YOLOv8 weights. It detects lane markings dynamically from each frame, associates potholes with the camera's ego lane, tracks events, shows red hazard boxes and a video-derived DAY/NIGHT indicator, and presents lane confidence and visual risk estimates in a browser. Vehicle speed remains N/A because the video has no physical speed source or calibrated scale.

## Existing project assets

- `models/best.pt` is the existing trained Pothole model and is loaded directly. It is not retrained, replaced, or modified.
- `assets/final.mp4` is the built-in 10-second demo input.
- `Pothole_Detection.py` and `Pothole_Detection.ipynb` remain as the original inference script and historical notebook. The notebook's Colab, Drive, and Roboflow steps are not used by the dashboard.
- The original direct YOLO inference call remains in `Pothole_Detection.py`; the web processor uses the same weights, class, and confidence threshold, with frame-wise inference to add lane-aware annotations.

## Architecture

```text
React + Vite browser → FastAPI → FFmpeg frame decode/encode → OpenCV overlays
                                                     ↘ existing YOLOv8 best.pt
Processed MP4 + JSON statistics ← lane filtering + event tracking + visual risk score
```

The app processes videos locally. It does not use Google Drive, Roboflow, a downloaded model, CUDA, or retraining. FFmpeg/FFprobe are used for video I/O because they are reliable on the current macOS setup; OpenCV draws and evaluates the lane geometry.

## Requirements

- macOS with the existing project `.venv` (Python 3.11, PyTorch 2.2.2, existing Ultralytics 8.0.200)
- Node.js and npm
- FFmpeg and FFprobe available on `PATH` (`brew install ffmpeg` if missing)

The new Python requirements are only the FastAPI server and multipart upload parser. Install into the existing `.venv`; do not recreate the environment or install/upgrade PyTorch, NumPy, or Ultralytics.

## Install and start

From the project root:

```bash
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m backend.main
```

The backend listens at `http://127.0.0.1:8000`. In a second terminal, from the project root:

```bash
cd frontend
npm install
npm run dev
```

Open **http://127.0.0.1:5173** in your browser. Keep both terminals running.

To show lane candidates, camera path, edges, and the approximate road mask in the processed feed, start the backend with `DEBUG_LANES=true python -m backend.main` (after activating `.venv`). Run the smoke checks with `.venv/bin/python test_lane_pipeline.py`.

## Run the demo or upload a video

1. Open the dashboard and click **Run demo** to process `assets/final.mp4`.
2. Or click **Upload video** (or drop a file onto the page) and choose an MP4. The upload limit is 500 MB.
3. Processing status and progress appear above the player. When complete, the processed video is displayed in the page with native play, pause, seek, volume, and fullscreen controls. Detection statistics and the unique event log update with the result.

Processed inputs, annotated videos, and a `results.json` copy are stored under ignored `runs/dashboard/<job-id>/` folders. Results report ego-lane, adjacent-road, outside-road, uncertain-lane, and lane-confidence values. API endpoints are `POST /api/process`, `POST /api/process/demo`, `GET /api/status/{job_id}`, `GET /api/results/{job_id}`, and `GET /api/video/{job_id}`. `/api/health` reports backend/model/demo availability.

## Current-lane filtering

Inspection note: the original working video path loads `models/best.pt` in `backend/detector.py`, calls YOLO per decoded frame in `backend/video_processor.py`, writes an annotated H.264 MP4 with FFmpeg, then returns its JSON results through FastAPI for the React player and statistics cards. The old documentation referred to a fixed trapezoid; the lane polygon is now built from frame-derived OpenCV Canny/Hough line candidates fitted to curved boundaries.

The detector selects the nearest plausible visible marking on either side of the configured camera path, fits and temporally smooths a curve, and forms a polygon between those observed boundaries. It has no fixed-polygon fallback: when the boundary confidence decays below threshold, the video labels the lane uncertain and potholes are excluded from ego-lane severity totals. Lane confidence is a line-support/vertical-coverage heuristic, not a calibrated probability. `camera_x` is the normalized camera optical-path position, not a selected left/right lane; tune it along with `LANE_CONFIG` for a different camera position, crop, lighting, or road. `DEBUG_LANES=true` adds candidate Hough lines, the camera path, edge map, and approximate low-chroma road mask.

Each YOLO box is associated using its bottom-center point. Points inside the detected ego-lane polygon count as relevant; points outside it are further classified against a low-chroma road-surface mask as adjacent roadway or outside-road. The road mask is a lightweight color heuristic and should be calibrated for different pavement/lighting; uncertain lane frames are kept out of primary risk statistics rather than guessed.

Pothole boxes are rendered in red, while lane overlays retain their separate lane color. A 160×90 grayscale thumbnail is sampled once per second; the median luminance classifies the processed clip as DAY or NIGHT using a threshold in `backend/vehicle_status.py`. This is a scene-lighting indicator, not weather detection. The supplied video has no usable GPS/CAN speed or calibrated scene scale, so the API returns `speed_kmh: null` and the UI displays `N/A`; it does not invent a real-world speed.

The HTML video element uses native controls with autoplay disabled. A compact control row adds playback time, restart, and playback-rate choices. Physical pothole depth is explicitly reported as not available.

## Tracking and severity

The lightweight centroid tracker in `backend/tracker.py` associates nearby pothole bottom-centers between frames and allows short detection gaps. Repeated appearances of one object become one event; the detection table records its first frame and the annotated video shows its tracked ID. Very long gaps, rapid camera changes, or similar potholes close together can still split or merge tracks.

`backend/severity.py` produces a transparent **visual risk estimate**, not a depth measurement. Its score is `0.50 × normalized box area + 0.25 × confidence + 0.25 × proximity`, where box area is normalized to 8% of the image and proximity increases as the box reaches the lower part of the frame. Defaults classify scores below 0.38 as LOW, below 0.67 as MEDIUM, and the rest as HIGH. Constants are configurable there. These values should be calibrated against the intended demonstration and are not validated road-safety thresholds.

## Limitations

- Lane geometry uses OpenCV edge/Hough candidates rather than a learned lane-segmentation model. Faded markings, intersections, shadows, unusual pavement colors, sharp curves, camera motion, and lane changes can reduce confidence; the camera path calibration and thresholds may need tuning.
- The road-surface mask is a low-chroma/value heuristic, not semantic road segmentation. Adjacent-versus-outside-road labels can be uncertain on gravel, concrete, wet, or strongly shadowed roads.
- Current-lane classification uses the bounding-box bottom-center; uncertain or partly occluded road contacts may be assigned incorrectly.
- Severity is a visual risk estimate only. It does **not** measure physical pothole depth in centimeters. Depth measurement needs additional calibrated stereo/depth sensors or another validated measurement method.
- Centroid tracking is intentionally lightweight; challenging occlusion and crowded detections can affect unique counts.
- In-memory job state resets when the backend restarts. This is intended for local project demonstrations.

## Original inference

The original command remains available from the project root:

```bash
source .venv/bin/activate
python Pothole_Detection.py
```

It continues to use `models/best.pt` on `assets/final.mp4` and save the YOLO output under `runs/detect/predict/`.
