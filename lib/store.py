"""Simple JSON-backed herd store with image snapshot persistence."""

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

DATA_DIR = Path("data")
HERD_FILE = DATA_DIR / "herd.json"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
MAX_FINGERPRINTS_PER_COW = 6
MAX_SNAPSHOTS_PER_COW = 6

DATA_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _read() -> dict:
    if not HERD_FILE.exists():
        return {"cattle": []}
    try:
        return json.loads(HERD_FILE.read_text())
    except Exception:
        return {"cattle": []}


def _write(data: dict) -> None:
    HERD_FILE.write_text(json.dumps(data, indent=2))


def list_cattle() -> list[dict]:
    return _read().get("cattle", [])


def get_cow(cow_id: str) -> Optional[dict]:
    for c in list_cattle():
        if c["id"] == cow_id:
            return c
    return None


def _save_snapshot(cow_id: str, crop_bgr: np.ndarray) -> str:
    cow_dir = SNAPSHOT_DIR / cow_id
    cow_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}.jpg"
    path = cow_dir / fname
    # store at modest size
    h, w = crop_bgr.shape[:2]
    if max(h, w) > 360:
        scale = 360 / max(h, w)
        crop_bgr = cv2.resize(
            crop_bgr, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(path), crop_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return str(path)


def add_cow(name: str, crop_bgr: np.ndarray, fingerprint: np.ndarray) -> dict:
    data = _read()
    cow_id = uuid.uuid4().hex[:12]
    snap_path = _save_snapshot(cow_id, crop_bgr)
    cow = {
        "id": cow_id,
        "name": name.strip() or "Unnamed",
        "snapshots": [snap_path],
        "fingerprints": [fingerprint.tolist()],
        "createdAt": int(time.time()),
        "lastSeen": int(time.time()),
    }
    data.setdefault("cattle", []).append(cow)
    _write(data)
    return cow


def add_observation(
    cow_id: str, crop_bgr: np.ndarray, fingerprint: np.ndarray
) -> Optional[dict]:
    data = _read()
    for cow in data.get("cattle", []):
        if cow["id"] != cow_id:
            continue
        snap_path = _save_snapshot(cow_id, crop_bgr)
        snaps = cow.get("snapshots", []) + [snap_path]
        prints = cow.get("fingerprints", []) + [fingerprint.tolist()]
        cow["snapshots"] = snaps[-MAX_SNAPSHOTS_PER_COW:]
        cow["fingerprints"] = prints[-MAX_FINGERPRINTS_PER_COW:]
        cow["lastSeen"] = int(time.time())
        _write(data)
        return cow
    return None


def rename_cow(cow_id: str, new_name: str) -> bool:
    data = _read()
    for cow in data.get("cattle", []):
        if cow["id"] == cow_id:
            cow["name"] = new_name.strip() or cow["name"]
            _write(data)
            return True
    return False


def delete_cow(cow_id: str) -> bool:
    data = _read()
    before = len(data.get("cattle", []))
    data["cattle"] = [c for c in data.get("cattle", []) if c["id"] != cow_id]
    _write(data)
    # Best-effort cleanup of snapshot folder
    cow_dir = SNAPSHOT_DIR / cow_id
    if cow_dir.exists():
        for p in cow_dir.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
        try:
            cow_dir.rmdir()
        except Exception:
            pass
    return len(data["cattle"]) < before


def encode_image_b64(path: str) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("ascii")
