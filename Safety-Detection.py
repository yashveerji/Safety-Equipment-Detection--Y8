"""Safety Detection using a YOLOv8 PPE model.

Features added:
 - CLI arguments for model path, source (video/webcam), confidence threshold, device, save output.
 - Graceful handling of end-of-stream and missing files.
 - FPS calculation overlay.
 - Color coding for compliance vs non-compliance.
 - Optional headless mode (no imshow) and output video writer.
 - Basic self-test function (run with --self-test).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import math
import threading
import queue
import numpy as np

import cv2
import cvzone
from ultralytics import YOLO
import torch  # type: ignore

# Default class names for PPE model (order must match the model training)
CLASS_NAMES: List[str] = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
    'Safety Vest', 'machinery', 'vehicle'
]

NON_COMPLIANT = {'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'}
COMPLIANT = {'Hardhat', 'Safety Vest', 'Mask'}

import json
import csv
import os


def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)


class SimpleTracker:
    """Very lightweight IOU-based tracker (no motion model)."""
    def __init__(self, iou_thresh: float = 0.4, max_age: int = 30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        assigned = set()
        for t in self.tracks.values():
            t['age'] += 1
        for det in detections:
            best_i = 0.0
            best_id = None
            for tid, tr in self.tracks.items():
                if tr['age'] > self.max_age:
                    continue
                i = iou(det['bbox'], tr['bbox'])
                if i > best_i:
                    best_i = i
                    best_id = tid
            if best_i >= self.iou_thresh and best_id is not None and best_id not in assigned:
                tr = self.tracks[best_id]
                tr['bbox'] = det['bbox']
                tr['age'] = 0
                det['track_id'] = best_id
                assigned.add(best_id)
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {'bbox': det['bbox'], 'age': 0}
                det['track_id'] = tid
                assigned.add(tid)
        # prune
        to_del = [tid for tid, tr in self.tracks.items() if tr['age'] > self.max_age]
        for tid in to_del:
            del self.tracks[tid]
        return detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safety Detection with YOLOv8 PPE model")
    parser.add_argument('--model', type=str, default='ppe.pt', help='Path to YOLOv8 PPE model (.pt)')
    parser.add_argument('--source', type=str, default='./videos/huuman.mp4',
                        help='Video file path or integer index for webcam (e.g., 0)')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold (0-1)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device for inference: cpu, cuda, or cuda:0 etc (falls back automatically)')
    parser.add_argument('--save', type=str, default=None, help='Optional output video file path to save annotated result')
    parser.add_argument('--headless', action='store_true', help='Run without GUI window (no imshow)')
    parser.add_argument('--self-test', action='store_true', help='Run a quick self test and exit')
    parser.add_argument('--max-frames', type=int, default=None, help='Limit processing to N frames (for testing)')
    parser.add_argument('--buffer-size', type=int, default=3, help='Frame grab buffer size (larger can reduce drops)')
    parser.add_argument('--frame-stride', type=int, default=1, help='Process every Nth frame (>=1)')
    parser.add_argument('--max-dim', type=int, default=None, help='Resize longer image side to this (keeps aspect) before inference')
    parser.add_argument('--half', action='store_true', help='Use half precision (CUDA only)')
    parser.add_argument('--imgsz', type=int, default=None, help='Explicit inference image size (square) fed to YOLO')
    parser.add_argument('--set-num-threads', type=int, default=None, help='Set torch & OpenCV thread count')
    parser.add_argument('--warmup', type=int, default=1, help='Number of warm-up inferences before timing')
    parser.add_argument('--no-fps-overlay', action='store_true', help='Disable drawing FPS overlay')
    parser.add_argument('--async-infer', action='store_true', help='Run inference in a separate thread to reduce UI latency')
    parser.add_argument('--target-fps', type=float, default=None, help='Desired display FPS; system will drop frames to approach this')
    parser.add_argument('--dynamic-skip', action='store_true', help='Adaptively skip frames if processing falls behind target FPS')
    parser.add_argument('--export-json', type=str, default=None, help='Path to write detections as NDJSON (one JSON object per frame)')
    parser.add_argument('--track', action='store_true', help='Enable simple IOU-based tracking for stable IDs')
    parser.add_argument('--stats-interval', type=int, default=50, help='Interval (processed frames) for console stats (sync mode)')
    parser.add_argument('--zones', type=str, default=None, help='Path to JSON file defining polygon zones for alerts')
    parser.add_argument('--draw-zones', action='store_true', help='Draw defined zones on output frames')
    parser.add_argument('--alert-log', type=str, default=None, help='Path to NDJSON file logging alert events')
    parser.add_argument('--export-csv', type=str, default=None, help='Path to CSV file with per-frame compliance counts')
    parser.add_argument('--beep-on-alert', action='store_true', help='Play system beep when an alert triggers (Windows only)')
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        if not Path(source).exists():
            print(f"[ERROR] Source video not found: {source}", file=sys.stderr)
            sys.exit(1)
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source: {source}", file=sys.stderr)
        sys.exit(1)
    return cap


def load_model(model_path: str, device: Optional[str] = None) -> YOLO:
    if not Path(model_path).exists():
        print(f"[ERROR] Model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)
    model = YOLO(model_path)
    # Device selection is handled internally by ultralytics; we can still attempt to set via .to(device) if provided.
    if device:
        try:
            model.to(device)
        except Exception as e:  # noqa: BLE001 broad for user convenience
            print(f"[WARN] Unable to move model to device '{device}': {e}. Continuing on default.")
    return model


def determine_color(cls_name: str) -> Tuple[int, int, int]:
    if cls_name in NON_COMPLIANT:
        return 0, 0, 255  # Red
    if cls_name in COMPLIANT:
        return 0, 255, 0  # Green
    return 255, 0, 0      # Blue for others


def extract_detections(results, conf_threshold: float) -> List[Dict[str, Any]]:
    dets: List[Dict[str, Any]] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls = int(box.cls[0])
            if cls >= len(CLASS_NAMES):
                continue
            cls_name = CLASS_NAMES[cls]
            dets.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'cls': cls,
                'cls_name': cls_name,
                'compliance': 'non-compliant' if cls_name in NON_COMPLIANT else ('compliant' if cls_name in COMPLIANT else 'other')
            })
    return dets


def annotate_detections(img, detections: List[Dict[str, Any]]) -> None:
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls_name = det['cls_name']
        conf = det['conf']
        color = determine_color(cls_name)
        display_conf = math.ceil(conf * 100) / 100
        label = f"{cls_name} {display_conf:.2f}"
        if 'track_id' in det:
            label = f"{det['track_id']}:{label}"
        cvzone.putTextRect(
            img,
            label,
            (max(0, x1), max(35, y1)),
            scale=1,
            thickness=1,
            colorB=color,
            colorT=(255, 255, 255),
            colorR=color,
            offset=5,
        )
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)


# -------------------- ZONE & ALERT UTILITIES --------------------

def load_zones(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return []
    if not Path(path).exists():
        print(f"[WARN] Zones file not found: {path}; ignoring.")
        return []
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        # Expected schema: [{"name": "Zone A", "polygon": [[x,y],...], "alert_on": ["NO-Hardhat","NO-Safety Vest"], "min_conf": 0.4}]
        zones = []
        for z in data:
            if 'polygon' in z and isinstance(z['polygon'], list):
                zones.append({
                    'name': z.get('name', f'zone_{len(zones)+1}'),
                    'polygon': z['polygon'],
                    'alert_on': set(z.get('alert_on', list(NON_COMPLIANT))),
                    'min_conf': float(z.get('min_conf', 0.3))
                })
        print(f"[INFO] Loaded {len(zones)} zone(s).")
        return zones
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed parsing zones file: {e}")
        return []


def point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    # Ray casting algorithm
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1)
        if cond:
            inside = not inside
    return inside


def evaluate_alerts(detections: List[Dict[str, Any]], zones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    alerts = []
    if not zones or not detections:
        return alerts
    for det in detections:
        cls_name = det['cls_name']
        conf = det['conf']
        x1, y1, x2, y2 = det['bbox']
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        for zone in zones:
            if cls_name in zone['alert_on'] and conf >= zone['min_conf']:
                if point_in_polygon(cx, cy, zone['polygon']):
                    alerts.append({
                        'zone': zone['name'],
                        'cls_name': cls_name,
                        'conf': conf,
                        'center': (cx, cy),
                        'track_id': det.get('track_id')
                    })
    return alerts


def draw_zones(frame, zones: List[Dict[str, Any]]):
    for zone in zones:
        pts = np.array([[int(p[0]), int(p[1])] for p in zone['polygon']], dtype=np.int32)
        if len(pts) >= 3:
            cv2.polylines(frame, [pts], True, (0, 165, 255), 2)
            cv2.putText(frame, zone['name'], (pts[0][0], pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2, cv2.LINE_AA)


def maybe_beep(enabled: bool):
    if not enabled:
        return
    try:
        if os.name == 'nt':  # Windows
            import winsound
            winsound.MessageBeep()  # default system sound
        else:
            print('\a', end='')  # bell char
    except Exception:
        pass


class FrameGrabber:
    """Threaded frame grabber to minimize I/O blocking."""

    def __init__(self, cap: cv2.VideoCapture, buffer_size: int = 3):
        self.cap = cap
        self.buffer_size = max(1, buffer_size)
        self.q: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            success, frame = self.cap.read()
            if not success:
                self._stop.set()
                break
            try:
                if self.q.full():
                    # Drop oldest frame to keep latest (low-latency approach)
                    try:
                        _ = self.q.get_nowait()
                    except queue.Empty:
                        pass
                self.q.put_nowait(frame)
            except Exception:
                pass

    def read(self):
        # Always return the most recent frame available
        frame = None
        try:
            while True:
                frame = self.q.get(timeout=0.2)
                # Drain extras to get latest
                if self.q.empty():
                    break
        except queue.Empty:
            return None
        return frame

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=1)


class InferenceWorker:
    """Asynchronous inference executor.

    Receives raw frames, performs model inference, returns annotated frames.
    Maintains only the latest frame to minimize latency.
    """

    def __init__(self, model: YOLO, conf: float, imgsz: Optional[int], max_dim: Optional[int], tracker: Optional[SimpleTracker], export_writer):
        self.model = model
        self.conf = conf
        self.imgsz = imgsz
        self.max_dim = max_dim
        self.tracker = tracker
        self.export_writer = export_writer
        self.in_queue: queue.Queue = queue.Queue(maxsize=1)
        self.out_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.latest_detections: List[Dict[str, Any]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def submit(self, frame):
        # Keep only newest frame
        if self.in_queue.full():
            try:
                _ = self.in_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self.in_queue.put_nowait(frame)
        except Exception:
            pass

    def get_latest(self):
        try:
            frame = self.out_queue.get_nowait()
            return frame, self.latest_detections
        except queue.Empty:
            return None, []

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=1)

    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self.in_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if frame is None:
                continue
            proc_frame = resize_keep_aspect(frame, self.max_dim)
            try:
                results = (self.model(proc_frame, stream=True, verbose=False, imgsz=self.imgsz)
                           if self.imgsz else self.model(proc_frame, stream=True, verbose=False))
                detections = extract_detections(results, self.conf)
                if self.tracker:
                    detections = self.tracker.update(detections)
                self.latest_detections = detections
                annotate_detections(proc_frame, detections)
                if self.export_writer:
                    payload = {
                        'ts': time.time(),
                        'detections': [
                            {
                                'bbox': det['bbox'],
                                'cls': det['cls'],
                                'cls_name': det['cls_name'],
                                'conf': det['conf'],
                                'track_id': det.get('track_id'),
                                'compliance': det['compliance']
                            } for det in detections
                        ]
                    }
                    self.export_writer.write(json.dumps(payload) + '\n')
                    self.export_writer.flush()
            except Exception as e:  # noqa: BLE001
                cvzone.putTextRect(proc_frame, f'InferErr: {e}', (10, 40), scale=0.7, thickness=1,
                                   colorB=(0, 0, 255), colorT=(255, 255, 255), colorR=(0, 0, 255), offset=3)
            # Keep only newest annotated frame
            if self.out_queue.full():
                try:
                    _ = self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.out_queue.put_nowait(proc_frame)
            except Exception:
                pass


def resize_keep_aspect(frame, max_dim: int):
    if max_dim is None:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / float(longest)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def maybe_set_threads(n_threads: Optional[int]):
    if not n_threads:
        return
    try:
        torch.set_num_threads(n_threads)
    except Exception:
        pass
    try:
        cv2.setNumThreads(n_threads)
    except Exception:
        pass


def run_detection(args: argparse.Namespace) -> None:
    maybe_set_threads(args.set_num_threads)
    model = load_model(args.model, args.device)
    cap = open_source(args.source)

    # Half precision if requested and on CUDA
    if args.half:
        try:
            if torch.cuda.is_available():
                model.model.half()  # type: ignore[attr-defined]
                print('[INFO] Using half precision.')
            else:
                print('[WARN] --half requested but CUDA not available; skipping.')
        except Exception as e:  # noqa: BLE001
            print(f'[WARN] Could not enable half precision: {e}')

    # Warm-up
    if args.warmup > 0:
        import numpy as np  # local, only if needed
        dummy = (255 * np.ones((320, 320, 3), dtype=np.uint8))
        for _ in range(args.warmup):
            _ = model.predict(dummy, verbose=False)
        torch.cuda.synchronize() if torch.cuda.is_available() else None  # type: ignore
        print(f'[INFO] Warm-up done ({args.warmup} iterations).')

    grabber = FrameGrabber(cap, buffer_size=args.buffer_size).start()

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[WARN] Failed to open writer at {args.save}; skipping save.")
            writer = None

    prev_time = time.time()
    frame_count = 0
    processed_frames = 0
    start_time = time.time()

    # Export + tracking setup
    export_writer = None
    if args.export_json:
        Path(args.export_json).parent.mkdir(parents=True, exist_ok=True)
        export_writer = open(args.export_json, 'w', encoding='utf-8')
        print(f'[INFO] Exporting detections to {args.export_json} (NDJSON).')

    tracker: Optional[SimpleTracker] = SimpleTracker() if args.track else None
    if tracker:
        print('[INFO] Simple IOU tracker enabled.')

    # Zones, CSV, alert log setup
    zones = load_zones(args.zones)
    alert_log_writer = None
    if args.alert_log:
        Path(args.alert_log).parent.mkdir(parents=True, exist_ok=True)
        alert_log_writer = open(args.alert_log, 'a', encoding='utf-8')
        print(f'[INFO] Alert log -> {args.alert_log}')
    csv_writer = None
    csv_file_handle = None
    if args.export_csv:
        Path(args.export_csv).parent.mkdir(parents=True, exist_ok=True)
        csv_file_handle = open(args.export_csv, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file_handle)
        csv_writer.writerow(['frame','ts','total','compliant','non_compliant','alerts'])
        print(f'[INFO] CSV export -> {args.export_csv}')

    infer_worker = None
    if args.async_infer:
        infer_worker = InferenceWorker(model, args.conf, args.imgsz, args.max_dim, tracker, export_writer).start()
        print('[INFO] Asynchronous inference enabled.')

    try:
        dynamic_skip_mod = 0
        while True:
            frame = grabber.read()
            if frame is None:
                if not grabber.thread.is_alive():
                    print('[INFO] No more frames; exiting.')
                    break
                else:
                    continue

            frame_count += 1
            if args.max_frames and frame_count > args.max_frames:
                print(f'[INFO] Reached max-frames ({args.max_frames}). Stopping.')
                break

            # Dynamic skip logic (adaptive)
            if args.dynamic_skip and args.target_fps:
                elapsed_total = time.time() - start_time
                current_display_fps = frame_count / elapsed_total if elapsed_total > 0 else 0
                if current_display_fps < args.target_fps * 0.85:
                    dynamic_skip_mod = min(dynamic_skip_mod + 1, 5)
                elif current_display_fps > args.target_fps * 1.05 and dynamic_skip_mod > 0:
                    dynamic_skip_mod -= 1
            else:
                dynamic_skip_mod = 0

            effective_stride = max(1, args.frame_stride + dynamic_skip_mod)
            if effective_stride > 1 and (frame_count % effective_stride != 0):
                if not args.headless and frame_count % (effective_stride * 2) == 0:
                    cv2.imshow('Safety Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print('[INFO] Quit signal received (q).')
                        break
                continue

            if infer_worker:
                infer_worker.submit(frame)
                annotated, frame_dets = infer_worker.get_latest()
                if annotated is None:
                    annotated = resize_keep_aspect(frame, args.max_dim)
                    frame_dets = []
                detections_for_alert = frame_dets
            else:
                infer_frame = resize_keep_aspect(frame, args.max_dim)
                results = (model(infer_frame, stream=True, verbose=False, imgsz=args.imgsz)
                           if args.imgsz else model(infer_frame, stream=True, verbose=False))
                detections = extract_detections(results, args.conf)
                if tracker:
                    detections = tracker.update(detections)
                annotate_detections(infer_frame, detections)
                if export_writer:
                    payload = {
                        'ts': time.time(),
                        'detections': [
                            {
                                'bbox': det['bbox'],
                                'cls': det['cls'],
                                'cls_name': det['cls_name'],
                                'conf': det['conf'],
                                'track_id': det.get('track_id'),
                                'compliance': det['compliance']
                            } for det in detections
                        ]
                    }
                    export_writer.write(json.dumps(payload) + '\n')
                annotated = infer_frame
                processed_frames += 1
                detections_for_alert = detections

            # Zone drawing & alert evaluation
            if args.draw_zones and zones:
                draw_zones(annotated, zones)
            alerts = evaluate_alerts(detections_for_alert, zones) if zones else []
            if alerts:
                maybe_beep(args.beep_on_alert)
                if alert_log_writer:
                    for a in alerts:
                        alert_log_writer.write(json.dumps({
                            'ts': time.time(),
                            'zone': a['zone'],
                            'cls_name': a['cls_name'],
                            'conf': a['conf'],
                            'track_id': a.get('track_id'),
                            'center': a['center']
                        }) + '\n')
                if csv_writer:
                    pass  # CSV is per-frame, handled below

            # CSV per-frame counts (compliance summary)
            if csv_writer and detections_for_alert is not None:
                total = len(detections_for_alert)
                compliant = sum(1 for d in detections_for_alert if d['compliance'] == 'compliant')
                non_compliant = sum(1 for d in detections_for_alert if d['compliance'] == 'non-compliant')
                csv_writer.writerow([
                    frame_count,
                    time.time(),
                    total,
                    compliant,
                    non_compliant,
                    len(alerts)
                ])

            # FPS overlay (display loop time)
            now = time.time()
            inst_fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now
            if not args.no_fps_overlay:
                overlay_text = f'FPS: {inst_fps:.1f}'
                if args.dynamic_skip and args.target_fps:
                    overlay_text += f' tgt:{args.target_fps:.0f} ds:{dynamic_skip_mod}'
                cvzone.putTextRect(annotated, overlay_text, (20, 20), scale=1, thickness=1,
                                    colorB=(0, 0, 0), colorT=(0, 255, 0), colorR=(0, 255, 0), offset=5)

            if writer:
                if annotated.shape[1] != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or annotated.shape[0] != int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                    out_frame = cv2.resize(annotated, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                else:
                    out_frame = annotated
                writer.write(out_frame)

            if not args.headless:
                cv2.imshow('Safety Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print('[INFO] Quit signal received (q).')
                    break

            if not args.async_infer and processed_frames and processed_frames % 50 == 0:
                elapsed = time.time() - start_time
                avg_fps = processed_frames / elapsed if elapsed > 0 else 0
                print(f'[STATS] Frames read: {frame_count} | Processed: {processed_frames} | Avg FPS: {avg_fps:.2f} | Stride:{effective_stride}')
    finally:
        grabber.stop()
        if infer_worker:
            infer_worker.stop()
        cap.release()
        if writer:
            writer.release()
        if export_writer:
            export_writer.close()
        if alert_log_writer:
            alert_log_writer.close()
        if 'csv_file_handle' in locals() and csv_file_handle:
            csv_file_handle.close()
        if not args.headless:
            cv2.destroyAllWindows()


def self_test(model_path: str) -> int:
    """Basic self-test: loads model and creates a blank frame to ensure pipeline runs."""
    try:
        model = YOLO(model_path)
        import numpy as np  # local import to avoid unused if not called
        dummy = (255 * np.ones((320, 320, 3), dtype=np.uint8))
        _ = model.predict(dummy, verbose=False)
        print('[SELF-TEST] Model loaded and dummy inference succeeded.')
        return 0
    except Exception as e:  # noqa: BLE001 broad for diagnostics
        print(f'[SELF-TEST] Failure: {e}', file=sys.stderr)
        return 1


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test(args.model)
    run_detection(args)
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
