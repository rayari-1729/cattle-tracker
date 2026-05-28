"""
03_generate_crops.py
====================
Generates 224×224 behavior crops from every version folder.
These crops become the training dataset for the MobileNetV3 behavior classifier.

Output:
  processed/behavior_crops/
  ├── standing/
  ├── walking/
  ├── eating/
  ├── drinking/
  ├── lying_down/
  ├── rumination/
  ├── sleeping/
  └── crop_stats.json

Usage:
    python scripts/03_generate_crops.py --padding 0.15
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
CROP_SIZE = (224, 224)


# ─────────────────────────────────────────────────────────────────────────────
def generate_crops_for_version(
    version_dir: Path,
    frames_root: Path,
    crops_out: Path,
    padding: float = 0.15,
    target_size: tuple[int, int] = (1280, 720),
) -> dict[str, int]:
    """
    Generates behavior crops for all annotations in a version folder.
    Returns per-behavior crop counts.
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

            # Scale + add padding
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

            crop_resized = cv2.resize(crop, CROP_SIZE)

            beh_dir = crops_out / behavior
            beh_dir.mkdir(parents=True, exist_ok=True)

            count = behavior_counts.get(behavior, 0)
            fname = f"{version_dir.name}_track{track_id}_f{frame_id:06d}_{count:04d}.jpg"
            cv2.imwrite(str(beh_dir / fname), crop_resized)
            behavior_counts[behavior] = count + 1

    return behavior_counts


# ─────────────────────────────────────────────────────────────────────────────
# Incremental processing helpers
# ─────────────────────────────────────────────────────────────────────────────
def _crop_done_path(version_name: str) -> Path:
    """Path to the per-version done sentinel inside CROPS_DIR."""
    return CROPS_DIR / f".done_{version_name}.json"


def is_cropped(version_name: str) -> bool:
    return _crop_done_path(version_name).exists()


def load_crop_record(version_name: str) -> dict:
    return json.loads(_crop_done_path(version_name).read_text())


def save_crop_record(version_name: str, counts: dict) -> None:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    _crop_done_path(version_name).write_text(json.dumps(counts, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding",  type=float, default=0.15,
                        help="Bbox padding ratio (default 0.15 = 15%)")
    parser.add_argument("--target_w", type=int, default=1280)
    parser.add_argument("--target_h", type=int, default=720)
    parser.add_argument(
        "--force", nargs="*", metavar="VERSION",
        help=(
            "Force re-crop even if already done. "
            "Pass specific versions (e.g. --force v13 v14) or bare --force for ALL."
        ),
    )
    args = parser.parse_args()

    # Resolve force flags
    force_all = args.force is not None and len(args.force) == 0
    force_set = set(args.force) if args.force else set()

    target_size = (args.target_w, args.target_h)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n✂️  Behavior Crop Generation  [incremental]  — padding={args.padding}")
    print(f"   Output: {CROPS_DIR}\n")

    global_counts: dict[str, int] = {b: 0 for b in BEHAVIORS}

    for vdir in VERSION_FOLDERS:
        force_this = force_all or (vdir.name in force_set)

        # ── Skip if already done ─────────────────────────────────────────────
        if is_cropped(vdir.name) and not force_this:
            cached = load_crop_record(vdir.name)
            print(f"  ⏭️  [{vdir.name}] already cropped — skipping "
                  f"(use --force {vdir.name} to redo)")
            for b, c in cached.items():
                global_counts[b] = global_counts.get(b, 0) + c
            continue

        # ── Generate crops ───────────────────────────────────────────────────
        counts = generate_crops_for_version(
            vdir, FRAMES_DIR, CROPS_DIR, args.padding, target_size
        )
        if counts:
            save_crop_record(vdir.name, counts)   # ← write sentinel
            print(f"  ✅ [{vdir.name}] crops: {counts}")
            for b, c in counts.items():
                global_counts[b] = global_counts.get(b, 0) + c

    # ── Always rebuild global crop_stats.json from all sentinels ─────────────
    # (ensures the file is complete even when old versions were skipped)
    merged: dict[str, int] = {b: 0 for b in BEHAVIORS}
    for vdir in VERSION_FOLDERS:
        if is_cropped(vdir.name):
            for b, c in load_crop_record(vdir.name).items():
                merged[b] = merged.get(b, 0) + c

    (CROPS_DIR / "crop_stats.json").write_text(json.dumps(merged, indent=2))

    print(f"\n✅ Crop generation complete:")
    total = sum(merged.values())
    for b, c in sorted(merged.items(), key=lambda x: -x[1]):
        pct    = 100 * c / max(total, 1)
        status = "❌ MISSING" if c == 0 else ("⚠️ LOW" if c < 200 else "✅")
        print(f"   {b:15s}: {c:5d} ({pct:5.1f}%)  {status}")

    print(f"\n   Total crops: {total}")
    print(f"   Stats saved: {CROPS_DIR / 'crop_stats.json'}")

    if any(v < 200 for v in merged.values()):
        print("\n⚠️  WARNING: Some classes have < 200 samples.")
        print("   Annotate more videos before training the classifier.")


if __name__ == "__main__":
    main()
