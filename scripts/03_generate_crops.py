"""
03_generate_crops.py
This is version 2 
=======================
Generates behavior crops from every version folder.

KEY CHANGE vs v1
----------------
CROP_SIZE is now 512×512 (was 224×224 hardcoded).

Why?
  The original script cropped the bbox out of a 1280×720 frame, then
  immediately downsampled to 224px — discarding resolution at save time.
  That forced you to re-run this expensive script every time you wanted
  to experiment with a different imgsz in the classifier (224, 288, 320…).

  Now we save at 512px (still well below the native bbox size on 4K source
  scaled to 1280×720), and let the YOLO trainer handle the final resize
  via imgsz= at training time.  One crop run → experiment freely.

  512px is chosen because:
    • Larger than any imgsz you'd realistically test (224 / 288 / 320 / 384)
    • Still ~4× smaller than storing raw crops → manageable Drive quota
    • cv2.resize from bbox→512 is higher quality than bbox→224

Usage:
    # Normal run (incremental, skips already-done versions):
    python scripts/03_generate_crops_v2.py

    # Override crop size (if you want even larger):
    python scripts/03_generate_crops_v2.py --crop_size 640

    # Force redo specific versions:
    python scripts/03_generate_crops_v2.py --force v13 v14

    # Force redo ALL:
    python scripts/03_generate_crops_v2.py --force

Output:
  processed/behavior_crops/
  ├── standing/      ← 512×512 JPEGs
  ├── walking/
  ├── eating/
  ├── drinking/
  ├── lying_down/
  ├── rumination/
  ├── sleeping/
  └── crop_stats.json
"""

import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import json
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.setup_drive import VERSION_FOLDERS, FRAMES_DIR, CROPS_DIR  # type: ignore

BEHAVIORS = [
    "standing", "eating", "walking", "drinking",
    "lying_down", "rumination", "sleeping",
]

# ── Changed: 512 instead of 224 ──────────────────────────────────────────────
# Trainer will resize to imgsz= (288 recommended) at training time.
# You can pass --crop_size 224 to reproduce old behaviour exactly.
DEFAULT_CROP_SIZE = 512


# ─────────────────────────────────────────────────────────────────────────────
def generate_crops_for_version(
    version_dir: Path,
    frames_root: Path,
    crops_out: Path,
    padding: float = 0.15,
    target_size: tuple[int, int] = (1280, 720),
    crop_size: int = DEFAULT_CROP_SIZE,
) -> dict[str, int]:
    """
    Generates behavior crops for all annotations in a version folder.
    Returns per-behavior crop counts.

    crop_size: square output size in pixels (both width and height).
    """
    xml_files = sorted(version_dir.glob("*.xml"))
    if not xml_files:
        print(f"  ⚠️  No XML in {version_dir.name}")
        return {}

    xml_path = xml_files[0]
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Resolve original resolution
    try:
        orig_w = int(root.find("meta/original_size/width").text)
        orig_h = int(root.find("meta/original_size/height").text)
    except (AttributeError, TypeError):
        orig_w, orig_h = 3840, 2160

    W, H = target_size
    scale_x = W / orig_w
    scale_y = H / orig_h

    # Find frames directory for this version
    version_frames_root = frames_root / version_dir.name
    stem_dirs = [d for d in version_frames_root.iterdir() if d.is_dir()] \
        if version_frames_root.exists() else []
    frames_dir = stem_dirs[0] if stem_dirs else version_frames_root

    behavior_counts: dict[str, int] = {b: 0 for b in BEHAVIORS}
    CROP_SIZE = (crop_size, crop_size)

    for track in root.findall("track"):
        track_id = track.get("id")

        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue

            frame_id = int(box.get("frame"))

            behavior = "standing"
            for attr in box.findall("attribute"):
                if attr.get("name") == "behavior":
                    behavior = (attr.text or "standing").strip()

            img_path = frames_dir / f"frame_{frame_id:06d}.jpg"
            if not img_path.exists():
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            # Scale bbox from annotation coords to frame coords + add padding
            xtl = int(float(box.get("xtl")) * scale_x)
            ytl = int(float(box.get("ytl")) * scale_y)
            xbr = int(float(box.get("xbr")) * scale_x)
            ybr = int(float(box.get("ybr")) * scale_y)

            pw = int((xbr - xtl) * padding)
            ph = int((ybr - ytl) * padding)
            xtl = max(0, xtl - pw)
            ytl = max(0, ytl - ph)
            xbr = min(W - 1, xbr + pw)
            ybr = min(H - 1, ybr + ph)

            crop = frame[ytl:ybr, xtl:xbr]
            if crop.size == 0:
                continue

            # Use INTER_AREA for downscaling (best quality for shrinking),
            # INTER_LINEAR for upscaling (rare, but safe fallback)
            h_crop, w_crop = crop.shape[:2]
            interp = cv2.INTER_AREA if (w_crop > crop_size or h_crop > crop_size) \
                     else cv2.INTER_LINEAR
            crop_resized = cv2.resize(crop, CROP_SIZE, interpolation=interp)

            beh_dir = crops_out / behavior
            beh_dir.mkdir(parents=True, exist_ok=True)

            count = behavior_counts.get(behavior, 0)
            fname = f"{version_dir.name}_track{track_id}_f{frame_id:06d}_{count:04d}.jpg"
            # Quality 95 — slightly higher than cv2 default (95 vs 95, explicit)
            cv2.imwrite(str(beh_dir / fname), crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            behavior_counts[behavior] = count + 1

    return behavior_counts


# ─────────────────────────────────────────────────────────────────────────────
# Incremental processing helpers  (unchanged from v1)
# ─────────────────────────────────────────────────────────────────────────────
def _crop_done_path(version_name: str, crop_size: int) -> Path:
    """Sentinel includes crop_size so changing size triggers a fresh run."""
    return CROPS_DIR / f".done_{version_name}_s{crop_size}.json"


def is_cropped(version_name: str, crop_size: int) -> bool:
    return _crop_done_path(version_name, crop_size).exists()


def load_crop_record(version_name: str, crop_size: int) -> dict:
    return json.loads(_crop_done_path(version_name, crop_size).read_text())


def save_crop_record(version_name: str, crop_size: int, counts: dict) -> None:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    _crop_done_path(version_name, crop_size).write_text(json.dumps(counts, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding",   type=float, default=0.15,
                        help="Bbox padding ratio (default 0.15 = 15%%)")
    parser.add_argument("--target_w",  type=int,   default=1280)
    parser.add_argument("--target_h",  type=int,   default=720)
    parser.add_argument("--crop_size", type=int,   default=DEFAULT_CROP_SIZE,
                        help=f"Square output crop size in px (default {DEFAULT_CROP_SIZE}). "
                             "Pass 224 to reproduce v1 behaviour exactly.")
    parser.add_argument(
        "--force", nargs="*", metavar="VERSION",
        help=(
            "Force re-crop even if already done. "
            "Pass specific versions (e.g. --force v13 v14) or bare --force for ALL."
        ),
    )
    args = parser.parse_args()

    force_all = args.force is not None and len(args.force) == 0
    force_set = set(args.force) if args.force else set()

    target_size = (args.target_w, args.target_h)
    crop_size   = args.crop_size
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n✂️  Behavior Crop Generation  [incremental]")
    print(f"   Output     : {CROPS_DIR}")
    print(f"   Crop size  : {crop_size}×{crop_size}px  ← trainer will resize to imgsz= at train time")
    print(f"   Padding    : {args.padding}\n")

    for vdir in VERSION_FOLDERS:
        force_this = force_all or (vdir.name in force_set)

        if is_cropped(vdir.name, crop_size) and not force_this:
            cached = load_crop_record(vdir.name, crop_size)
            print(f"  ⏭️  [{vdir.name}] already cropped at {crop_size}px — skipping")
            continue

        counts = generate_crops_for_version(
            vdir, FRAMES_DIR, CROPS_DIR, args.padding, target_size, crop_size
        )
        if counts:
            save_crop_record(vdir.name, crop_size, counts)
            total_v = sum(counts.values())
            print(f"  ✅ [{vdir.name}] {total_v} crops → {counts}")

    # ── Rebuild global crop_stats.json ────────────────────────────────────────
    merged: dict[str, int] = {b: 0 for b in BEHAVIORS}
    for vdir in VERSION_FOLDERS:
        if is_cropped(vdir.name, crop_size):
            for b, c in load_crop_record(vdir.name, crop_size).items():
                merged[b] = merged.get(b, 0) + c

    (CROPS_DIR / "crop_stats.json").write_text(json.dumps(merged, indent=2))

    print(f"\n✅ Crop generation complete  (crop_size={crop_size}px):")
    total = sum(merged.values())
    for b, c in sorted(merged.items(), key=lambda x: -x[1]):
        pct    = 100 * c / max(total, 1)
        status = "❌ MISSING" if c == 0 else ("⚠️  LOW" if c < 200 else "✅")
        print(f"   {b:15s}: {c:5d} ({pct:5.1f}%)  {status}")

    print(f"\n   Total crops : {total}")
    print(f"   Stats saved : {CROPS_DIR / 'crop_stats.json'}")
    print(f"\n   ℹ️  Crops saved at {crop_size}px. Train with any imgsz ≤ {crop_size}:")
    print(f"      python train_classifier_yolo_v2.py --imgsz 288")
    print(f"      python train_classifier_yolo_v2.py --imgsz 224  # reproduce v1")

    if any(v < 200 for v in merged.values()):
        print("\n⚠️  WARNING: Some classes have < 200 samples.")
        print("   Annotate more videos before training the classifier.")


if __name__ == "__main__":
    main()