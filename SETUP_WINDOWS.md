# Windows setup

## NVIDIA GPU (RTX 50-series)

For this RTX 5050 laptop, use Python 3.11 with CUDA-enabled PyTorch. Stop the
backend before replacing packages in an existing environment:

```powershell
# Run the venv command only when creating a new environment.
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

PyTorch 2.7 added Blackwell support with CUDA 12.8 wheels; see the
[release notes](https://pytorch.org/blog/pytorch-2-7/) and
[official version pairings](https://pytorch.org/get-started/previous-versions/).
The GPU requirements also update Ultralytics for newer PyTorch checkpoint loading.
The original trained `models/best.pt` is retained.

`YOLO_DEVICE` defaults to `auto` (CUDA when available, otherwise CPU). To force
CPU for comparison, set `$env:YOLO_DEVICE = 'cpu'` before starting the backend.
To require a GPU, set `$env:YOLO_DEVICE = 'cuda:0'`; unavailable devices produce
an error. Remove the override with `Remove-Item Env:YOLO_DEVICE`.
The API health/results and dashboard report the selected inference device.
YOLO inference uses the GPU; lane detection and FFmpeg processing still use CPU.

Follow the system-tool and startup instructions below, but do not install the
CPU requirements into the GPU environment.

## Original CPU setup

Use Python 3.11 for the inference versions documented by this project.
Node.js/npm and FFmpeg (including FFprobe) are also required.

Install the system tools if absent, then open a new PowerShell terminal so PATH
changes are available:

```powershell
winget install --id Python.Python.3.11 --exact --source winget --scope user
winget install --id Gyan.FFmpeg --exact --source winget --scope user
```

From the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r backend/requirements-windows.txt
.\.venv\Scripts\python.exe -m pip check
ffmpeg -version
ffprobe -version
```

The CPU PyTorch pair follows the project's documented PyTorch version and the
[official compatibility instructions](https://docs.pytorch.org/get-started/previous-versions/).
NumPy and OpenCV are pinned to avoid installing NumPy 2 into that older inference
stack. This file sets up a fresh environment; it does not upgrade an existing
working macOS environment.

The trained pothole weights are included at `models/best.pt`. Keep this file in
place when setting up a fresh clone. Generic YOLO weights are not a replacement
for this project's pothole model.

Start the API:

```powershell
.\.venv\Scripts\python.exe -m backend.main
```

In a second terminal:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev
```

Open http://127.0.0.1:5173. API health is at http://127.0.0.1:8000/api/health.
Health reports model/demo file presence; it does not validate model loading.

Checks (the lane pipeline smoke check requires the trained model):

```powershell
.\.venv\Scripts\python.exe test_lane_pipeline.py
cd frontend
npm.cmd run build
```

The current lane detector uses OpenCV; no additional lane model is required.
Evaluating a learned lane detector is a separate step with its own weights and
runtime requirements.
