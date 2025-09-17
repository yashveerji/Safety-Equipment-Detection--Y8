"""Reusable detection core for PPE YOLOv8 model.

This module factors out lightweight functionality from the CLI script so it can be
imported by a web server or other integrations without pulling in argument parsing.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import math
import time
from pathlib import Path
import os
from urllib.request import urlretrieve

import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

# Re-declare or import; for simplicity we redefine constants (keep in sync with main script)
CLASS_NAMES: List[str] = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
    'Safety Vest', 'machinery', 'vehicle'
]
NON_COMPLIANT = {'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'}
COMPLIANT = {'Hardhat', 'Safety Vest', 'Mask'}


def determine_color(cls_name: str) -> Tuple[int, int, int]:
    if cls_name in NON_COMPLIANT:
        return 0, 0, 255
    if cls_name in COMPLIANT:
        return 0, 255, 0
    return 255, 0, 0


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


def resize_keep_aspect(frame, max_dim: Optional[int]):
    if not max_dim:
        return frame
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return frame
    scale = max_dim / float(longest)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class Detector:
    def __init__(self, model_path: str = 'ppe.pt', device: Optional[str] = None, conf: float = 0.5, max_dim: Optional[int] = None, imgsz: Optional[int] = None):
        """Initialize detector.

        Behavior when model file is missing:
        1. If env PPE_MODEL_URL provided and model_path ends with .pt, attempt download to that path.
        2. If still missing, load fallback weight name from env PPE_MODEL_FALLBACK (default 'yolov8n.pt').
        3. Emit warning to stdout so platform logs show reason instead of hard crash.
        """
        p = Path(model_path)
        if not p.exists():
            # Attempt download if URL provided
            dl_url = os.getenv('PPE_MODEL_URL')
            if dl_url and model_path.endswith('.pt'):
                try:
                    print(f"[INFO] Model '{model_path}' missing. Downloading from PPE_MODEL_URL: {dl_url}")
                    urlretrieve(dl_url, model_path)
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] Download failed ({e}). Will try fallback model.")
        if not p.exists():
            fallback_name = os.getenv('PPE_MODEL_FALLBACK', 'yolov8n.pt')
            print(f"[WARN] Model file '{model_path}' not found after attempts. Using fallback '{fallback_name}'.")
            try:
                self.model = YOLO(fallback_name)
            except Exception as e:  # pragma: no cover
                raise FileNotFoundError(f"Could not load fallback model '{fallback_name}': {e}") from e
        else:
            self.model = YOLO(model_path)
        if device:
            try:
                self.model.to(device)
            except Exception:  # pragma: no cover
                pass
        self.conf = conf
        self.max_dim = max_dim
        self.imgsz = imgsz

    def predict(self, frame, conf_override: Optional[float] = None) -> Dict[str, Any]:
        t0 = time.time()
        proc = resize_keep_aspect(frame, self.max_dim)
        results = (self.model(proc, stream=True, verbose=False, imgsz=self.imgsz)
                   if self.imgsz else self.model(proc, stream=True, verbose=False))
        dets = extract_detections(results, self.conf if conf_override is None else conf_override)
        annotate_detections(proc, dets)
        return {
            'image': proc,
            'detections': dets,
            'latency_ms': (time.time() - t0) * 1000.0
        }

    def predict_raw(self, frame, conf_override: Optional[float] = None) -> List[Dict[str, Any]]:
        proc = resize_keep_aspect(frame, self.max_dim)
        results = (self.model(proc, stream=True, verbose=False, imgsz=self.imgsz)
                   if self.imgsz else self.model(proc, stream=True, verbose=False))
        return extract_detections(results, self.conf if conf_override is None else conf_override)


__all__ = [
    'Detector',
    'extract_detections',
    'annotate_detections',
    'determine_color',
    'CLASS_NAMES',
]
