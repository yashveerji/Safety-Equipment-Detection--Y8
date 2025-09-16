# Safety Detection (PPE) – YOLOv8 Real‑Time System

An optimized, configurable personal protective equipment (PPE) detection pipeline built on **Ultralytics YOLOv8**, featuring threaded video capture, adaptive frame processing, compliance coloring, and export-ready architecture for future extensions.

---
## 1. What This Project Does
Detects safety‑related objects (hardhats, masks, safety vests, etc.) in video streams or webcam feeds and visually classifies them as compliant (green), non‑compliant (red), or neutral (blue). Designed for real-time monitoring, demo playback, or batch video analysis.

---
## 2. Core Features
| Category | Feature |
|----------|---------|
| Performance | Threaded frame grabbing, frame skipping (`--frame-stride`), dynamic resizing (`--max-dim`), optional half precision (`--half`), warm-up stage |
| Usability | Rich CLI, self-test mode, headless mode, optional video saving |
| Visualization | Color-coded bounding boxes + confidence + optional FPS overlay |
| Robustness | Graceful end-of-stream handling, safe device selection, configurable buffer size |
| Extensibility | Modular functions (`extract_detections`, `annotate_detections`, `determine_color`, `FrameGrabber`) |

---
## 3. Quick Start (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Sanity check (loads model + dummy inference)
python Safety-Detection.py --self-test

# Run on sample video
python Safety-Detection.py --model ppe.pt --source .\videos\huuman.mp4

# Live webcam
python Safety-Detection.py --source 0
```

---
## 4. CLI Usage Cheatsheet
```
--model <path>            Path to model weights (default: ppe.pt)
--source <path|index>     Video file or webcam index (default: ./videos/huuman.mp4)
--conf <float>            Confidence threshold (default: 0.5)
--device <cpu|cuda|...>   Inference device (auto if omitted)
--save <out.mp4>          Save annotated video
--headless                Disable GUI window
--self-test               Run a dummy inference and exit
--max-frames <N>          Stop after N frames (testing)
--buffer-size <N>         Capture buffer queue size (default 3)
--frame-stride <N>        Process every Nth frame (>=1)
--max-dim <pixels>        Resize longer image side before inference
--half                    Use half precision (CUDA only)
--imgsz <size>            Override YOLO inference size (square)
--set-num-threads <N>     Set torch/OpenCV thread count
--warmup <N>              Warm-up inferences before timing (default 1)
--no-fps-overlay          Disable FPS text drawing
```

### Example Scenarios
| Goal | Command |
|------|---------|
| Save output (no window) | `python Safety-Detection.py --source video.mp4 --save out/annot.mp4 --headless` |
| Faster on CPU | `python Safety-Detection.py --frame-stride 2 --max-dim 720 --conf 0.4` |
| High FPS GPU | `python Safety-Detection.py --half --max-dim 640 --frame-stride 1` |
| Stress test 100 frames | `python Safety-Detection.py --max-frames 100 --no-fps-overlay` |

---
## 5. Model & Classes
Default class order (must match training):
```
['Hardhat','Mask','NO-Hardhat','NO-Mask','NO-Safety Vest','Person','Safety Cone','Safety Vest','machinery','vehicle']
```
Color logic:
* Green: compliant (`Hardhat`, `Safety Vest`, `Mask`)
* Red: non-compliant (`NO-Hardhat`, `NO-Safety Vest`, `NO-Mask`)
* Blue: neutral/other

To change: edit `CLASS_NAMES` and sets `COMPLIANT` / `NON_COMPLIANT` in `Safety-Detection.py`.

---
## 6. Architecture Overview

Component | Purpose
----------|--------
`FrameGrabber` | Background thread reading frames into a bounded queue; drops oldest to stay real-time.
`run_detection()` | Orchestrates capture, inference, annotation, display, writing, and stats.
`extract_detections()` | Parses raw YOLO results into structured detection dicts.
`annotate_detections()` | Draws bounding boxes + labels + colors.
`determine_color()` | Encapsulates compliance color policy.
`self_test()` | Lightweight pipeline check (model loads + one dummy inference).

Data Flow:
```
Capture Thread --> Queue --> Main Loop --> (Stride Skip?) --> (Resize) --> YOLO --> Annotate --> Display/Save
```

---
## 7. Performance Tuning Guide
Problem | Suggested Flags | Notes
--------|------------------|------
Slow decode / lag | `--buffer-size 5` | Larger buffer helps if decoding jittery.
High latency | Lower `--buffer-size`, add `--frame-stride 2` | Drops frames to stay “live”.
Low FPS on CPU | `--max-dim 640 --frame-stride 3 --conf 0.4` | Smaller images + skipping.
GPU under-utilized | `--half --max-dim 640 --warmup 3` | Warm-up & FP16 on CUDA.
Stutters writing video | Use SSD, avoid resizing after annotate | If resizing occurs for writer, pre-set capture resolution.

General Tips:
* Use `--frame-stride` instead of reading every frame when real-time view > completeness.
* `--max-dim` preserves aspect ratio and reduces inference cost.
* Combine `--half` + `--imgsz 640` for typical performance balance on GPU.

### 7.1 Advanced Latency Controls (New)

Flag | Purpose | When to Use
-----|---------|------------
`--async-infer` | Runs model inference in a background thread so the display loop stays responsive | Webcam feeds feeling delayed
`--target-fps <N>` | Desired display FPS used together with dynamic skipping | You want consistent visual rate
`--dynamic-skip` | Automatically increases frame skipping when falling behind target FPS | You can't guess a good stride upfront

Recommended combos:
* Lowest latency (GPU): `--async-infer --target-fps 30 --dynamic-skip --max-dim 640 --half`
* CPU real-time attempt: `--async-infer --target-fps 20 --dynamic-skip --frame-stride 1 --max-dim 640`
* Aggressive drop mode: `--async-infer --target-fps 25 --dynamic-skip --frame-stride 2 --max-dim 512`

How it works:
* The capture thread always pushes the most recent frame.
* With `--async-infer`, inference runs in parallel; the main loop displays the latest annotated frame (or raw if inference not ready).
* With `--dynamic-skip`, if actual display FPS < 85% of target, it increases an internal skip modifier; if >105%, it relaxes it.
* This keeps perceived latency low by preferring frame freshness over completeness.

Indicators:
* Overlay text shows `FPS:` plus `tgt:` and `ds:` (dynamic skip level) when enabled.
* Higher `ds` value means more frequent skipping due to load.

If latency persists:
- Historical trend sparklines for FPS / detections / non-compliance
- Detection filtering (substring & non-compliant only toggle)
- Fullscreen live stream mode & frame capture
- Interactive zone drawing directly on the video (polygon creation)
- Export detection history as CSV or NDJSON
- Optional audible beep on new alerts
1. Verify camera output resolution (drop to 720p or 480p).
2. Ensure no other heavy GPU processes are running.
3. Try `--imgsz 480` along with `--max-dim 640`.
4. Disable video writer if not needed (I/O can stall pipeline on slow disks).
5. Set `--set-num-threads 1` on oversubscribed CPUs to reduce context switching.

### 7.2 Detection Export & Tracking

- Fullscreen toggle (fills display)
- Draw Zone mode: click to add polygon points, name & save.
New Flags:
| Flag | Description |

Filtering:

- Text filter matches substring of class name (case-insensitive)
- Non-Comp checkbox shows only classes starting with `NO-`
|------|-------------|
| `--export-json path.ndjson` | Writes one JSON object per processed frame (NDJSON) containing detection metadata. |
| `--track` | Enables lightweight IOU-based tracker assigning `track_id` per object. |
Exports:

- CSV: flat rows of detection history (rolling ~500 detections)
- NDJSON: one JSON object per line for streaming / ingestion pipelines

Sparklines are lightweight inline SVG (no heavy chart libraries) for performance.

| `--stats-interval N` | Controls how often (in processed frames) stats are printed in synchronous mode. |


Interactive zone drawing (from Live Stream panel) persists zones immediately with default alert classes (all non-compliant) and adjustable name.
NDJSON Frame Schema Example:
```json
{"ts": 1726500000.123, "detections": [
	{"bbox": [120, 45, 260, 330], "cls": 0, "cls_name": "Hardhat", "conf": 0.87, "track_id": 3, "compliance": "compliant"},
	{"bbox": [400, 80, 545, 360], "cls": 2, "cls_name": "NO-Hardhat", "conf": 0.79, "track_id": 4, "compliance": "non-compliant"}
]}
- UI theming presets & accessibility improvements
```

Usage Examples:
```powershell
# Export detections with tracking (webcam)
python Safety-Detection.py --source 0 --track --export-json logs\detections.ndjson --async-infer

# Synchronous run with custom stats interval
python Safety-Detection.py --source .\videos\huuman.mp4 --stats-interval 25 --frame-stride 2
```

Notes:
* Tracking is IOU-only; IDs may jump on occlusion.
* For robust tracking (re-ID, motion), integrate a dedicated tracker later (e.g., BYTETrack).
* File is flushed per frame; for very large exports consider compressing afterward.

### 7.3 Zones, Alerts, CSV & Beep

New Advanced Flags:
| Flag | Purpose |
|------|---------|
| `--zones zones.json` | Load polygon zones to monitor for non-compliant classes |
| `--draw-zones` | Overlay zone polygons on output frames |
| `--alert-log alerts.ndjson` | Append alert events (zone + class + confidence) as NDJSON |
| `--export-csv stats.csv` | Per-frame summary (counts + alerts) CSV |
| `--beep-on-alert` | Play system beep when any alert triggers |

Zone File Schema (`zones.json`):
```json
[
	{
		"name": "LoadingDock",
		"polygon": [[100,100],[500,100],[500,400],[100,400]],
		"alert_on": ["NO-Hardhat", "NO-Safety Vest"],
		"min_conf": 0.4
	},
	{
		"name": "VehicleLane",
		"polygon": [[600,50],[1100,50],[1100,550],[600,550]],
		"alert_on": ["Person"],
		"min_conf": 0.3
	}
]
```

Alert Event NDJSON Example:
```json
{"ts": 1726505555.321, "zone": "LoadingDock", "cls_name": "NO-Hardhat", "conf": 0.82, "track_id": 7, "center": [320.5, 244.0]}
```

CSV Columns:
```
frame, ts, total, compliant, non_compliant, alerts
```

Usage Examples:
```powershell
# Monitor zones with audible alerts and logging
python Safety-Detection.py --source 0 --zones zones.json --draw-zones --alert-log logs\alerts.ndjson --export-csv logs\summary.csv --beep-on-alert --track --async-infer

# Offline video analysis with zone export only
python Safety-Detection.py --source .\videos\huuman.mp4 --zones zones.json --export-csv logs\video_stats.csv --export-json logs\dets.ndjson --frame-stride 2
```

Tips:
* Place larger zones earlier in the JSON if many overlaps (micro-optimization).
* Use moderate `--frame-stride` with `--async-infer` for low-latency alerting.
* Consider separate process to tail `alerts.ndjson` and trigger external notifications.

---
## 8. Extending the Project
Task | How
-----|----
Export detections | Wrap `results` parsing and save JSON per frame.
Add tracking | Integrate a tracker (e.g., BYTETrack, StrongSORT) after `extract_detections` and before `annotate_detections`.
Stream over network | Replace display with WebSocket/WebRTC push.
Alerting | Add rule engine: if red class persists N frames -> log/notify.
Batch image folder | Iterate files; modify `open_source` or create a new path.

---
## 9. Troubleshooting
Issue | Cause | Fix
------|-------|----
Black window | Bad path or codec | Verify file path & try MP4/H.264.
No CUDA | Driver / toolkit mismatch | Reinstall PyTorch per https://pytorch.org.
Half precision ignored | CPU only | Remove `--half` or switch to GPU.
Writer file empty | Early quit or unsupported codec | Let run a few seconds; try `.mp4` with `mp4v`.
Low accuracy | Threshold too high | Lower `--conf` to 0.35.

Log Hints:
* `[STATS]` lines show cumulative throughput.
* Warm-up eliminates initial cold start latency.

---
## 10. Retraining (High Level)
1. Collect annotated dataset with target safety classes.
2. Create YOLO dataset YAML (train/val paths + class names).
3. Train via Ultralytics: `yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640`.
4. Replace `ppe.pt` with your new weights.
5. Update `CLASS_NAMES` if class order differs.

---
## 11. Security & Deployment Notes
* Avoid exposing raw video feeds over insecure networks.
* For edge devices (Jetson, etc.), lower `--max-dim` and use a small model variant.
* Containerization: pin requirements; ensure GPU base image includes matching CUDA runtime.

---

## 12. Web API & Frontend (New)

A lightweight FastAPI server (`web_app.py`) plus a static HTML frontend (`frontend/index.html`) let you:
* Upload an image and get annotated preview + JSON detections
* Upload a video and download an annotated MP4
* View a live MJPEG webcam stream with on-the-fly detections

### 12.1 Install Extra Dependencies
Already appended to `requirements.txt` (FastAPI, Uvicorn, etc.). If you installed earlier, run:
```powershell
pip install -r requirements.txt
```

### 12.2 Run the Server
```powershell
uvicorn web_app:app --reload --port 8000
```
Then open: `http://127.0.0.1:8000/`

Environment Variable:
* `PPE_MODEL` – override model path (defaults to `ppe.pt`). Example:
```powershell
$env:PPE_MODEL = 'models\\custom_ppe.pt'; uvicorn web_app:app --port 8000
```

### 12.3 API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | HTML frontend (inlined fallback) |
| GET | `/api/health` | Health + model info |
| POST | `/api/predict/image` | Multipart `file` image → JSON {detections, latency_ms, image_b64} |
| POST | `/api/predict/video` | Multipart `file` video → annotated MP4 binary |
| GET | `/stream` | MJPEG live webcam stream |

Sample `curl` Image Request:
```bash
curl -F "file=@test.jpg" http://127.0.0.1:8000/api/predict/image > resp.json
```

### 12.4 Notes & Limitations
* Video endpoint writes temp files (`upload_input.tmp`, `processed_output.mp4`) – safe for single user testing. For concurrent/production: generate unique temp names and clean asynchronously.
* MJPEG stream uses camera index 0; adjust code for multi-camera.
* For HTTPS or auth, wrap behind a reverse proxy (Nginx / Traefik) and add auth middleware.
* Large videos processed fully before response (no progressive streaming yet).
* CPU-only environments: consider reducing model size or adding `max_dim` in `Detector`.

### 12.5 Embedding the API
Programmatic usage (Python):
```python
import requests
with open('image.jpg','rb') as f:
	r = requests.post('http://localhost:8000/api/predict/image', files={'file': f})
print(r.json())
```

---

---
## 13. Acknowledgments
* Ultralytics for YOLOv8
* Open-source ecosystem: PyTorch, OpenCV, cvzone

---
## 14. FAQ
Q: How do I speed up buffering?  
A: Use threaded grabber (already enabled), add `--frame-stride 2`, limit resolution with `--max-dim 720`, and enable `--half` on GPU.

Q: Is accuracy worse with `--frame-stride`?  
A: You may miss transient events between processed frames; choose stride based on acceptable trade-offs.

Q: Why FPS varies?  
A: Different scene complexities change inference duration; averaging over more frames gives stable stats.

---
## 15. Minimal Example (Programmatic Use)
```python
from ultralytics import YOLO
import cv2
from Safety-Detection import extract_detections, annotate_detections  # adjust if packaging

model = YOLO('ppe.pt')
cap = cv2.VideoCapture('videos/huuman.mp4')
while True:
	ok, frame = cap.read()
	if not ok:
		break
	results = model(frame, stream=True, verbose=False)
	dets = extract_detections(results, 0.5)
	annotate_detections(frame, dets)
	cv2.imshow('demo', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
```

---
## 16. Next Steps (Optional Enhancements)
* Add JSON export of detections (`--export-json`).
* Integrate tracking + identity persistence.
* Provide Dockerfile + GPU build instructions.
* Add automated test for CLI argument parsing.

---
Feel free to request any of these enhancements and they can be added quickly.

---
## 17. React Frontend (Modern Dashboard)

A modern React + TypeScript + Tailwind SPA (`frontend-react/`) provides a richer console for live monitoring, configuration, and alert review.

### 17.1 Features
Panel | Capability
------|-----------
Live Stream | Start/stop MJPEG stream, frame capture download, live frame/latency indicators
Controls | Adjust camera, confidence threshold, toggle tracking / zone drawing / beep, upload zone JSON
Stats | FPS, total frames, detection counts, non-compliance, computed latency (ms)
Detections | Table of most recent annotated detections (class, conf, bbox, track id)
Alerts | Last triggered zone/class alerts (with confidence and track id if present)
Zones | List existing zones, point count, delete individual zones, view JSON schema
Toasts | Global success/error/info notifications
Dark Mode | Automatic dark theme (Tailwind) with modern styling

Latency is derived from backend-provided timestamps: `latency_ms = (server_time - last_frame_ts) * 1000`. A large latency indicates buffering or camera stall.

### 17.2 Run the React UI
Prereq: Backend running (FastAPI) on port 8000.

```powershell
cd Safety-Detection-YOLOv8\frontend-react
npm install
npm run dev
```
Then open the shown local address (typically http://127.0.0.1:5173). The dev server proxies API calls to `http://localhost:8000`.

### 17.3 Production Build
```powershell
cd Safety-Detection-YOLOv8\frontend-react
npm run build
```
Output will be in `frontend-react/dist/`. You can serve it behind any static file server or integrate with FastAPI via `StaticFiles` mount.

### 17.4 Zone JSON (Upload Format)
When using the Controls panel's Zone JSON upload, supply an array of objects (minimal form also accepted):
```json
[
	{
		"name": "Zone A",
		"points": [[100,120],[180,130],[140,200]],
		"alert_on": ["NO-Hardhat"],
		"min_conf": 0.4
	}
]
```
Backend stores them as `{ name, polygon, alert_on, min_conf }` (note: property `points` is accepted during upload and treated as `polygon`).

### 17.5 App State & Polling
The frontend uses a global context provider (`AppContext`) to: 
* Load config, cameras, zones on mount
* Poll `/api/live_stats` every ~750ms only while streaming
* Compute latency client-side (see formula above)
* Expose actions: start/stop stream, apply config, add/delete zones, refresh lists, capture frame

### 17.6 Extending the Dashboard
Enhancement | Direction
------------|----------
Audio alerts | Play sound on new alert diff in `alerts_last`
Historical graphs | Add a ring buffer of FPS/detections and display sparkline
Export detections | Button calling new endpoint to fetch last N detections NDJSON
Auth | Insert login page + token header on all fetches
WebSocket | Replace polling with push stats + binary frame streaming

### 17.7 Troubleshooting (Frontend)
Issue | Suggestion
------|-----------
Blank stream area | Ensure backend running; check browser network for `/stream` 200 status
Latency grows continuously | Camera delivering frames slowly; reduce resolution or restart capture device
Cameras list empty | No accessible indexes 0–5; adjust probe logic in `web_app.py`
Zone upload error | Validate JSON array root and each object has `name` & `points`

---
## 18. Changelog (Recent Additions)
Version | Highlights
--------|----------
0.2.x | Added FastAPI backend endpoints, basic HTML UI
0.3.x | Added tracking, zones, alerts, async improvements
0.4.x | Introduced React + Tailwind dashboard with latency metric & structured panels

---
## 19. Quick Run Summary
Backend:
```powershell
uvicorn web_app:app --reload --port 8000
```
Frontend (dev):
```powershell
cd Safety-Detection-YOLOv8\frontend-react
npm install
npm run dev
```
Navigate to dev URL and start stream.

---
If you need Docker, WebSocket streaming, or advanced analytics panels, open an issue or request an enhancement.

---
## 20. Deployment (Render)

This repo includes a `render.yaml` that defines two services:

Service | Type | Purpose
--------|------|--------
`ppe-backend` | Web (Python) | FastAPI + model inference
`ppe-frontend` | Static Site | React build output

### 20.1 Environment Variables
Variable | Service | Description | Default
---------|---------|-------------|--------
`PPE_MODEL` | backend | Model weights path | `ppe.pt`
`CORS_ORIGINS` | backend | Comma-separated allowed origins | `http://localhost:5173`
`VITE_API_BASE` | frontend | Backend base URL for API calls | `http://localhost:8000`

Copy `.env.example` to `.env` (and frontend-react/.env.example likewise) for local overrides.

### 20.2 Steps on Render
1. Create a new Web Service from your repo using `render.yaml` autodetect.
2. After first deploy note backend URL (e.g., `https://ppe-backend.onrender.com`).
3. Update the frontend service environment variable `VITE_API_BASE` to the backend URL.
4. Update backend `CORS_ORIGINS` to include the frontend URL.
5. Redeploy both services.

### 20.3 Verifying Deployment
Run health:
```bash
curl https://<backend-domain>/api/health
```
Open frontend and start stream; check network requests point to backend domain.

### 20.4 Production Hardening Ideas
Item | Rationale
-----|----------
Pin model version hash | Reproducible inference results
Add request rate limiting | Prevent abuse of image/video endpoints
Add auth (API key or JWT) | Secure restricted operations / streams
Use WebSocket for stats | Reduced polling overhead at scale
GPU-backed instance | Higher FPS for heavy workloads

---
## 21. Environment File Summary
File | Purpose
----|--------
`.env.example` | Backend & shared defaults
`frontend-react/.env.example` | Frontend (Vite) API base

During CI/CD set real secrets (if any) and tune origins (never rely on wildcard CORS in production).

