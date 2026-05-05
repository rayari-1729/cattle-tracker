"""YOLOv8-based cattle detection and tracking using Ultralytics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class IDs we treat as "cattle-like" so the app gracefully handles
# demo footage that may not contain literal cows.
# 19 = cow, 17 = horse, 18 = sheep
CATTLE_CLASS_IDS = {19, 17, 18}
PRIMARY_COW_CLASS_ID = 19

MODEL_DIR = Path(os.environ.get("YOLO_MODEL_DIR", "data/models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Railway CPU containers need a small default. Users can still select larger
# weights when they have enough memory/CPU budget.
DEFAULT_MODEL = os.environ.get("CATTLE_MODEL", "yolov8n.pt")

# Model catalog presented in the UI. Each entry is (label, weights_filename,
# approximate size MB, short blurb). All weights live in the Ultralytics
# GitHub assets and are downloaded on first use.
MODEL_CATALOG: list[dict] = [
    {"id": "yolov8n.pt", "family": "YOLOv8", "size": "n", "mb": 6,   "desc": "Fastest. Lower accuracy."},
    {"id": "yolov8s.pt", "family": "YOLOv8", "size": "s", "mb": 22,  "desc": "Small, quick."},
    {"id": "yolov8m.pt", "family": "YOLOv8", "size": "m", "mb": 52,  "desc": "Balanced. Good default on CPU."},
    {"id": "yolov8l.pt", "family": "YOLOv8", "size": "l", "mb": 87,  "desc": "Higher accuracy, slower."},
    {"id": "yolov8x.pt", "family": "YOLOv8", "size": "x", "mb": 136, "desc": "Most accurate v8, slowest."},

    {"id": "yolov9t.pt", "family": "YOLOv9", "size": "t", "mb": 4,   "desc": "Tiny v9."},
    {"id": "yolov9s.pt", "family": "YOLOv9", "size": "s", "mb": 15,  "desc": "Small v9."},
    {"id": "yolov9m.pt", "family": "YOLOv9", "size": "m", "mb": 38,  "desc": "Medium v9."},
    {"id": "yolov9c.pt", "family": "YOLOv9", "size": "c", "mb": 49,  "desc": "Compact v9."},
    {"id": "yolov9e.pt", "family": "YOLOv9", "size": "e", "mb": 112, "desc": "Extra-accurate v9."},

    {"id": "yolov10n.pt", "family": "YOLOv10", "size": "n", "mb": 6,   "desc": "Fastest v10."},
    {"id": "yolov10s.pt", "family": "YOLOv10", "size": "s", "mb": 16,  "desc": "Small v10."},
    {"id": "yolov10m.pt", "family": "YOLOv10", "size": "m", "mb": 32,  "desc": "Balanced v10."},
    {"id": "yolov10b.pt", "family": "YOLOv10", "size": "b", "mb": 40,  "desc": "Big v10."},
    {"id": "yolov10l.pt", "family": "YOLOv10", "size": "l", "mb": 51,  "desc": "Large v10."},
    {"id": "yolov10x.pt", "family": "YOLOv10", "size": "x", "mb": 62,  "desc": "Most accurate v10."},

    {"id": "yolo11n.pt", "family": "YOLO11", "size": "n", "mb": 5,   "desc": "Fastest YOLO11."},
    {"id": "yolo11s.pt", "family": "YOLO11", "size": "s", "mb": 19,  "desc": "Small YOLO11."},
    {"id": "yolo11m.pt", "family": "YOLO11", "size": "m", "mb": 39,  "desc": "Balanced YOLO11."},
    {"id": "yolo11l.pt", "family": "YOLO11", "size": "l", "mb": 50,  "desc": "Large YOLO11."},
    {"id": "yolo11x.pt", "family": "YOLO11", "size": "x", "mb": 110, "desc": "Most accurate YOLO11."},

    {"id": "yolo12n.pt", "family": "YOLO12", "size": "n", "mb": 5,   "desc": "Fastest YOLO12."},
    {"id": "yolo12s.pt", "family": "YOLO12", "size": "s", "mb": 19,  "desc": "Small YOLO12."},
    {"id": "yolo12m.pt", "family": "YOLO12", "size": "m", "mb": 40,  "desc": "Balanced YOLO12."},
    {"id": "yolo12l.pt", "family": "YOLO12", "size": "l", "mb": 53,  "desc": "Large YOLO12."},
    {"id": "yolo12x.pt", "family": "YOLO12", "size": "x", "mb": 113, "desc": "Most accurate YOLO12."},

    {"id": "yolo26n.pt", "family": "YOLO26", "size": "n", "mb": 5,   "desc": "Fastest YOLO26 (newest family)."},
    {"id": "yolo26s.pt", "family": "YOLO26", "size": "s", "mb": 19,  "desc": "Small YOLO26."},
    {"id": "yolo26m.pt", "family": "YOLO26", "size": "m", "mb": 40,  "desc": "Balanced YOLO26."},
    {"id": "yolo26l.pt", "family": "YOLO26", "size": "l", "mb": 53,  "desc": "Large YOLO26."},
    {"id": "yolo26x.pt", "family": "YOLO26", "size": "x", "mb": 113, "desc": "Most accurate YOLO26."},
]


_model_cache: dict[str, YOLO] = {}


def get_model(weights: Optional[str] = None) -> YOLO:
    """Load and cache the YOLO model by weights filename."""
    name = weights or DEFAULT_MODEL
    if name in _model_cache:
        return _model_cache[name]

    model_path = MODEL_DIR / name
    if not model_path.exists():
        # Let Ultralytics download to CWD, then move into our cache.
        tmp = YOLO(name)
        src = Path(name)
        if src.exists():
            src.replace(model_path)
            model = YOLO(str(model_path))
        else:
            model = tmp
    else:
        model = YOLO(str(model_path))

    _model_cache[name] = model
    return model


def track_video(
    video_path: str,
    output_path: str,
    conf: float = 0.35,
    iou: float = 0.5,
    imgsz: int = 640,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    weights: Optional[str] = None,
) -> Iterator[dict]:
    """Run detection + ByteTrack on the video, write annotated MP4, and yield
    progress events.

    Yields dicts of one of:
        {"type": "meta", "fps": float, "total": int, "width": int, "height": int}
        {"type": "progress", "frame": int, "tracks": list[TrackObs]}
        {"type": "done", "tracks": dict[int, list[TrackObs]]}

    Each TrackObs is {"track_id": int, "class_id": int, "conf": float,
        "bbox": [x1,y1,x2,y2], "frame": int}.
    """
    model = get_model(weights)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_stride = max(1, int(frame_stride))
    if max_frames is not None:
        total = min(total, max_frames)

    processed_total = max(1, int(np.ceil(total / frame_stride))) if total else 0

    yield {
        "type": "meta",
        "fps": fps,
        "output_fps": fps / frame_stride,
        "total": total,
        "processed_total": processed_total,
        "width": width,
        "height": height,
    }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps / frame_stride, (width, height))

    track_observations: dict[int, list[dict]] = {}
    best_crops: dict[int, dict] = {}
    frame_idx = 0

    try:
        # Use the streaming generator from ultralytics for efficient frame iteration.
        results = model.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(CATTLE_CLASS_IDS),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            vid_stride=frame_stride,
            verbose=False,
        )

        for res in results:
            source_frame_idx = frame_idx * frame_stride
            if max_frames is not None and source_frame_idx >= max_frames:
                break

            frame = res.orig_img.copy()
            obs_this_frame: list[dict] = []

            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                cls = res.boxes.cls.cpu().numpy().astype(int)
                ids = (
                    res.boxes.id.cpu().numpy().astype(int)
                    if res.boxes.id is not None
                    else np.array([-1] * len(xyxy))
                )

                for box, c, k, tid in zip(xyxy, confs, cls, ids):
                    if tid < 0:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box]
                    obs = {
                        "track_id": int(tid),
                        "class_id": int(k),
                        "conf": float(c),
                        "bbox": [x1, y1, x2, y2],
                        "frame": source_frame_idx,
                    }
                    obs_this_frame.append(obs)
                    track_observations.setdefault(int(tid), []).append(obs)

                    if (
                        int(tid) not in best_crops
                        or float(c) > best_crops[int(tid)]["obs"]["conf"]
                    ):
                        crop = _crop_from_frame(res.orig_img, [x1, y1, x2, y2])
                        if crop is not None:
                            best_crops[int(tid)] = {"obs": obs, "crop": crop}

                    # Draw box + label
                    label = f"#{int(tid)}  {c:.2f}"
                    color = _color_for_id(int(tid))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame,
                        (x1, max(0, y1 - th - 8)),
                        (x1 + tw + 8, y1),
                        color,
                        -1,
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            writer.write(frame)
            frame_idx += 1

            if frame_idx % 5 == 0 or frame_idx == processed_total:
                yield {
                    "type": "progress",
                    "frame": frame_idx,
                    "tracks": obs_this_frame,
                }
    finally:
        cap.release()
        writer.release()

    yield {
        "type": "done",
        "tracks": track_observations,
        "best_crops": best_crops,
        "frames_processed": frame_idx,
    }


def _color_for_id(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(tid * 9973 + 7)
    h = int(rng.integers(0, 180))
    hsv = np.uint8([[[h, 200, 230]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def crop_from_video(
    video_path: str, frame_idx: int, bbox: list[int]
) -> Optional[np.ndarray]:
    """Read a specific frame from the source video and return the cropped BGR patch."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return _crop_from_frame(frame, bbox)
    finally:
        cap.release()


def _crop_from_frame(frame: np.ndarray, bbox: list[int]) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2].copy()
