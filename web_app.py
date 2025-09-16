"""FastAPI web server exposing PPE detection.

Endpoints:
- GET /            : Serves static index.html (simple frontend)
- POST /api/predict/image : Multipart image -> JSON detections + annotated image (base64)
- POST /api/predict/video : Multipart video -> processed MP4 file download
- GET /stream      : Live webcam MJPEG stream with annotated frames

Run:
    uvicorn web_app:app --reload --port 8000
"""
from __future__ import annotations
import io
import os
import time
import base64
from typing import List, Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from detector_core import Detector

MODEL_PATH = os.environ.get('PPE_MODEL', 'ppe.pt')
DETECTOR = Detector(model_path=MODEL_PATH, conf=0.5, max_dim=960)

app = FastAPI(title="PPE Detection Web API")

# CORS setup (allow configured origins)
_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:5173').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend directory if exists
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), 'frontend')
if os.path.isdir(FRONTEND_DIR):
    app.mount('/static', StaticFiles(directory=FRONTEND_DIR), name='static')

INDEX_HTML = """<!doctype html><html><head><meta charset='utf-8'><title>PPE Detection</title>
<style>body{font-family:Arial;margin:20px;}section{margin-bottom:30px;}code{background:#eee;padding:2px 4px;}#stream{border:1px solid #444;max-width:640px}</style></head>
<body>
<h1>PPE Detection Web UI</h1>
<section>
<h2>Image Upload</h2>
<form id="imgForm" enctype="multipart/form-data">
<input type="file" name="file" accept="image/*" required />
<button type="submit">Detect</button>
</form>
<div id="imgResult"></div>
</section>
<section>
<h2>Video Upload</h2>
<form id="vidForm" enctype="multipart/form-data">
<input type="file" name="file" accept="video/*" required />
<button type="submit">Process Video</button>
</form>
<a id="vidLink" style="display:none;">Download Processed Video</a>
</section>
<section>
<h2>Live Stream</h2>
<img id="stream" src="/stream" alt="Live Stream" />
</section>
<script>
const imgForm = document.getElementById('imgForm');
imgForm.addEventListener('submit', async (e)=>{
 e.preventDefault();
 const fd = new FormData(imgForm);
 const r = await fetch('/api/predict/image',{method:'POST', body:fd});
 if(!r.ok){alert('Error');return;}
 const data = await r.json();
 let html = `<p>Latency: ${data.latency_ms.toFixed(1)} ms, Detections: ${data.detections.length}</p>`;
 html += `<img style='max-width:640px;border:1px solid #444' src='data:image/jpeg;base64,${data.image_b64}' />`;
 html += '<pre>'+JSON.stringify(data.detections, null, 2)+'</pre>';
 document.getElementById('imgResult').innerHTML = html;
});
const vidForm = document.getElementById('vidForm');
vidForm.addEventListener('submit', async (e)=>{
 e.preventDefault();
 const fd = new FormData(vidForm);
 const r = await fetch('/api/predict/video',{method:'POST', body:fd});
 if(!r.ok){alert('Video error');return;}
 // Expect binary mp4
 const blob = await r.blob();
 const url = URL.createObjectURL(blob);
 const link = document.getElementById('vidLink');
 link.href = url; link.download='processed.mp4'; link.style.display='inline'; link.textContent='Download Processed Video';
});
</script>
</body></html>"""

@app.get('/', response_class=HTMLResponse)
async def root():
    return HTMLResponse(INDEX_HTML)

@app.post('/api/predict/image')
async def predict_image(file: UploadFile = File(...), conf: float | None = None):
    try:
        content = await file.read()
        np_arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail='Invalid image')
        result = DETECTOR.predict(img, conf_override=conf)
        _, buf = cv2.imencode('.jpg', result['image'])
        b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        return JSONResponse({
            'detections': result['detections'],
            'latency_ms': result['latency_ms'],
            'image_b64': b64,
            'conf_used': conf if conf is not None else DETECTOR.conf
        })
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post('/api/predict/video')
async def predict_video(file: UploadFile = File(...), conf: float | None = None):
    try:
        raw = await file.read()
        in_mem = io.BytesIO(raw)
        data = np.frombuffer(in_mem.getbuffer(), np.uint8)
        cap = cv2.VideoCapture(cv2.CAP_FFMPEG)
        # Writing temp file approach for simplicity (FFmpeg decode from memory is non-trivial w/o extra libs)
        tmp_in = 'upload_input.tmp'
        with open(tmp_in, 'wb') as f:
            f.write(raw)
        cap = cv2.VideoCapture(tmp_in)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail='Could not open video')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        tmp_out = 'processed_output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(tmp_out, fourcc, fps, (width, height))
        frame_count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = DETECTOR.predict(frame, conf_override=conf)
            # Ensure output size matches writer expectation
            outf = result['image']
            if outf.shape[0] != height or outf.shape[1] != width:
                outf = cv2.resize(outf, (width, height))
            writer.write(outf)
            frame_count += 1
        writer.release(); cap.release()
        # Return as binary
        with open(tmp_out, 'rb') as f:
            data = f.read()
        # Cleanup temp files (best effort)
        try:
            os.remove(tmp_in)
            os.remove(tmp_out)
        except OSError:
            pass
        headers = { 'X-Frames-Processed': str(frame_count), 'X-Conf-Used': str(conf if conf is not None else DETECTOR.conf) }
        return Response(content=data, media_type='video/mp4', headers=headers)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get('/stream')
async def stream(conf: float | None = None, camera: int | None = None):
    def gen():
        with STATE_LOCK:
            cfg_cam = APP_STATE['config']['camera_index']
        cam_index = camera if camera is not None else cfg_cam
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            yield b""
            return
        last_time = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with STATE_LOCK:
                cfg = APP_STATE['config'].copy()
                zones = list(APP_STATE['zones'])
            result = DETECTOR.predict(frame, conf_override=conf if conf is not None else cfg['default_conf'])
            dets = result['detections']
            if cfg['tracking']:
                dets = TRACKER.update(dets)
            alerts = evaluate_alerts(dets, zones) if zones else []
            if cfg['draw_zones'] and zones:
                for z in zones:
                    pts = [(int(p[0]), int(p[1])) for p in z['polygon']]
                    if len(pts) >= 3:
                        cv2.polylines(result['image'], [np.array(pts, dtype=np.int32)], True, (0,165,255), 2)
                        cv2.putText(result['image'], z['name'], pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,140,255), 2, cv2.LINE_AA)
            now = time.time(); dt = now - last_time; last_time = now; fps = 1.0/dt if dt>0 else 0
            non_comp = sum(1 for d in dets if d['cls_name'] in NON_COMPLIANT)
            with STATE_LOCK:
                APP_STATE['stats']['frames'] += 1
                APP_STATE['stats']['detections'] = len(dets)
                APP_STATE['stats']['non_compliant'] = non_comp
                APP_STATE['stats']['recent_detections'] = dets[:15]
                APP_STATE['stats']['alerts_last'] = alerts[-10:]
                APP_STATE['stats']['fps'] = round(fps,2)
                APP_STATE['stats']['last_frame_ts'] = now
            cv2.putText(result['image'], f"FPS {fps:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2, cv2.LINE_AA)
            if alerts:
                cv2.putText(result['image'], f"ALERTS:{len(alerts)}", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2, cv2.LINE_AA)
            _, jpg = cv2.imencode('.jpg', result['image'])
            bytes_ = jpg.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + bytes_ + b"\r\n")
        cap.release()
    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/api/health')
async def health():
    return {'status': 'ok', 'model': MODEL_PATH}

@app.get('/api/live_stats')
async def live_stats():
    with STATE_LOCK:
        _out = APP_STATE['stats'].copy()
        _out['server_time'] = time.time()
        return _out

# --- Tracking & Zones (lightweight integration) ---
import math
import json
import threading

NON_COMPLIANT = {'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'}
COMPLIANT = {'Hardhat', 'Safety Vest', 'Mask'}


def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0: return 0.0
    aA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1]); aB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / float(aA + aB - inter + 1e-6)

class SimpleTracker:
    def __init__(self, iou_thresh: float = 0.4, max_age: int = 30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]]):
        assigned = set()
        for t in self.tracks.values():
            t['age'] += 1
        for det in detections:
            best_id, best_iou = None, 0.0
            for tid, tr in self.tracks.items():
                if tr['age'] > self.max_age: continue
                i = _iou(det['bbox'], tr['bbox'])
                if i > best_iou:
                    best_iou = i; best_id = tid
            if best_id and best_iou >= self.iou_thresh and best_id not in assigned:
                tr = self.tracks[best_id]; tr['bbox'] = det['bbox']; tr['age'] = 0; det['track_id'] = best_id; assigned.add(best_id)
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {'bbox': det['bbox'], 'age': 0}
                det['track_id'] = tid; assigned.add(tid)
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]['age'] > self.max_age:
                del self.tracks[tid]
        return detections

# Global state (thread-safe updates)
STATE_LOCK = threading.Lock()
APP_STATE = {
    'config': {
        'tracking': True,
        'draw_zones': True,
        'beep': False,
        'default_conf': 0.5,
        'camera_index': 0
    },
    'zones': [],  # list of {name, polygon, alert_on, min_conf}
    'stats': {
        'frames': 0,
        'detections': 0,
        'non_compliant': 0,
        'alerts_last': [],
        'recent_detections': [],
        'last_frame_ts': None
    }
}
TRACKER = SimpleTracker()


def point_in_polygon(x, y, polygon):
    inside = False
    n = len(polygon)
    for i in range(n):
        x1,y1 = polygon[i]; x2,y2 = polygon[(i+1)%n]
        cond = ((y1>y)!=(y2>y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-9)+x1)
        if cond: inside = not inside
    return inside

def evaluate_alerts(detections, zones):
    alerts = []
    for d in detections:
        cls_name = d['cls_name']; conf = d['conf']
        x1,y1,x2,y2 = d['bbox']
        cx = (x1+x2)/2; cy=(y1+y2)/2
        for z in zones:
            if cls_name in z['alert_on'] and conf >= z['min_conf']:
                if point_in_polygon(cx, cy, z['polygon']):
                    alerts.append({'zone': z['name'], 'cls_name': cls_name, 'conf': conf, 'track_id': d.get('track_id')})
    return alerts

# Config & Zones endpoints
@app.get('/api/config')
async def get_config():
    with STATE_LOCK:
        return APP_STATE['config']

@app.post('/api/config')
async def update_config(payload: Dict[str, Any]):
    with STATE_LOCK:
        for k in ['tracking','draw_zones','beep','default_conf','camera_index']:
            if k in payload:
                APP_STATE['config'][k] = payload[k]
        return APP_STATE['config']

@app.get('/api/cameras')
async def list_cameras():
    # Probe first N indices (light heuristic)
    cams = []
    for idx in range(0, 6):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cams.append({'index': idx, 'width': w, 'height': h})
            cap.release()
        else:
            cap.release()
    return {'cameras': cams}

@app.get('/api/zones')
async def get_zones():
    with STATE_LOCK:
        return APP_STATE['zones']

@app.post('/api/zones')
async def add_zone(payload: Dict[str, Any]):
    # Expected: name, polygon:[[x,y],...], alert_on:[classes], min_conf
    if 'polygon' not in payload:
        raise HTTPException(status_code=400, detail='polygon required')
    zone = {
        'name': payload.get('name', f'zone_{len(APP_STATE["zones"]) + 1}'),
        'polygon': payload['polygon'],
        'alert_on': payload.get('alert_on', list(NON_COMPLIANT)),
        'min_conf': float(payload.get('min_conf', 0.4))
    }
    with STATE_LOCK:
        APP_STATE['zones'].append(zone)
        return zone

@app.delete('/api/zones/{name}')
async def delete_zone(name: str):
    with STATE_LOCK:
        before = len(APP_STATE['zones'])
        APP_STATE['zones'] = [z for z in APP_STATE['zones'] if z['name'] != name]
        return {'removed': before - len(APP_STATE['zones'])}
