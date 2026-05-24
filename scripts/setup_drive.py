"""
00_setup_drive.py
=================
Run this FIRST in every Colab session.

Your Google Drive layout (create this manually once):
  MyDrive/
  └── cattle-analytics/
      ├── data/
      │   ├── v1/          ← folder per video-version
      │   │   ├── *.mp4
      │   │   └── annotations.xml
      │   ├── v2/
      │   │   ├── *.mp4
      │   │   └── annotations.xml
      │   └── ...          ← up to v10 for Phase-1 experiment
      ├── processed/       ← auto-created by pipeline
      ├── models/
      │   └── checkpoints/ ← weights persist across sessions
      └── logs/
          └── experiments/ ← JSON logs per run

Usage (Colab cell):
    exec(open('scripts/00_setup_drive.py').read())
    # or simply import it:
    import scripts.setup_drive as cfg; print(cfg.DRIVE_ROOT)
"""

import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Mount Google Drive (Colab only — skipped when running locally)
# ─────────────────────────────────────────────────────────────────────────────
IS_COLAB = "google.colab" in sys.modules or os.path.exists("/content")

if IS_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        print("✅ Google Drive mounted at /content/drive")
    except Exception as e:
        print(f"⚠️  Drive mount failed: {e}. Check Colab runtime permissions.")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Path constants — edit DRIVE_ROOT if your folder is named differently
# ─────────────────────────────────────────────────────────────────────────────
if IS_COLAB:
    DRIVE_ROOT   = Path("/content/drive/MyDrive/cattle-analytics")
else:
    # Local dev: adjust to wherever you cloned the project
    DRIVE_ROOT = Path(__file__).resolve().parents[2]

# Subdirectory layout
RAW_DATA_DIR    = DRIVE_ROOT / "data"            # v1/, v2/, ... folders live here
PROCESSED_DIR   = DRIVE_ROOT / "processed"
MODELS_DIR      = DRIVE_ROOT / "models" / "checkpoints"
LOGS_DIR        = DRIVE_ROOT / "logs" / "experiments"

# Processed sub-dirs
FRAMES_DIR      = PROCESSED_DIR / "frames"
YOLO_DIR        = PROCESSED_DIR / "yolo_detection"
CROPS_DIR       = PROCESSED_DIR / "behavior_crops"
SPLITS_DIR      = PROCESSED_DIR / "splits"

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Create all directories
# ─────────────────────────────────────────────────────────────────────────────
ALL_DIRS = [
    RAW_DATA_DIR, PROCESSED_DIR, MODELS_DIR, LOGS_DIR,
    FRAMES_DIR, CROPS_DIR, SPLITS_DIR,
    YOLO_DIR / "images" / "train",
    YOLO_DIR / "images" / "val",
    YOLO_DIR / "labels" / "train",
    YOLO_DIR / "labels" / "val",
]
for d in ALL_DIRS:
    d.mkdir(parents=True, exist_ok=True)

print(f"✅ Drive root : {DRIVE_ROOT}")
print(f"   Raw data  : {RAW_DATA_DIR}")
print(f"   Processed : {PROCESSED_DIR}")
print(f"   Models    : {MODELS_DIR}")
print(f"   Logs      : {LOGS_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Discover version folders (v1, v2, …)
# ─────────────────────────────────────────────────────────────────────────────
def get_version_folders(max_versions: int = 10) -> list[Path]:
    """
    Returns sorted list of version dirs that actually exist on Drive.
    Each folder must contain at least one .mp4 AND an annotations.xml.
    """
    found = []
    for i in range(1, max_versions + 1):
        vdir = RAW_DATA_DIR / f"v{i}"
        if not vdir.exists():
            continue
        mp4s = list(vdir.glob("*.mp4"))
        xmls = list(vdir.glob("*.xml"))
        if mp4s and xmls:
            found.append(vdir)
        else:
            print(f"  ⚠️  v{i} found but missing mp4 or xml — skipping")
    print(f"\n📁 Version folders ready for processing: {[d.name for d in found]}")
    return found


VERSION_FOLDERS = get_version_folders(max_versions=10)
