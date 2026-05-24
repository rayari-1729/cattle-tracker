"""
02_cvat_to_yolo.py
==================
Converts CVAT XML annotations from ALL version folders into YOLO detection format.

Detection task: single class (class_id = 0 = calf).
Behavior labels are captured separately into a side-car JSON for the classifier.

Key design decisions:
  • Train/Val split is done AT VIDEO LEVEL, not frame level.
    → Prevents temporal leakage (frames from same video spanning both sets).
  • Version folders (v1..v10) are split proportionally.
  • Bounding box coordinates are scaled from original resolution to target resolution.

Output:
  processed/
  ├── yolo_detection/
  │   ├── images/train/   ← JPEG frames
  │   ├── images/val/
  │   ├── labels/train/   ← YOLO .txt files
  │   ├── labels/val/
  │   ├── dataset.yaml
  │   └── dataset_stats.json
  └── splits/
      ├── train_versions.txt
      └── val_versions.txt

Usage:
    python scripts/02_cvat_to_yolo.py --val_split 0.2
"""

import xml.etree.ElementTree as ET
import shutil
import json
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.setup_drive import (  # type: ignore
    VERSION_FOLDERS, FRAMES_DIR, YOLO_DIR, SPLITS_DIR,
)

BEHAVIORS = [
    "standing", "eating", "walking", "drinking",
    "lying_down", "rumination", "sleeping",
]
BEHAVIOR_TO_IDX = {b: i for i, b in enumerate(BEHAVIORS)}


# ─────────────────────────────────────────────────────────────────────────────
def parse_cvat_xml(
    xml_path: Path,
    frames_dir: Path,
    orig_size: tuple[int, int] = (3840, 2160),
    target_size: tuple[int, int] = (1280, 720),
) -> dict[int, list[tuple]]:
    """
    Parse a CVAT XML and return frame_data:
      { frame_id: [(xtl, ytl, xbr, ybr, behavior), ...] }

    Coordinates are already scaled to target_size.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Try to read orig size from XML meta
    try:
        orig_w = int(root.find("meta/original_size/width").text)
        orig_h = int(root.find("meta/original_size/height").text)
    except (AttributeError, TypeError):
        orig_w, orig_h = orig_size

    scale_x = target_size[0] / orig_w
    scale_y = target_size[1] / orig_h

    frame_data: dict[int, list] = {}

    for track in root.findall("track"):
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            frame_id = int(box.get("frame"))

            xtl = float(box.get("xtl")) * scale_x
            ytl = float(box.get("ytl")) * scale_y
            xbr = float(box.get("xbr")) * scale_x
            ybr = float(box.get("ybr")) * scale_y

            behavior = "standing"
            for attr in box.findall("attribute"):
                if attr.get("name") == "behavior":
                    behavior = attr.text.strip() if attr.text else "standing"

            if frame_id not in frame_data:
                frame_data[frame_id] = []
            frame_data[frame_id].append((xtl, ytl, xbr, ybr, behavior))

    return frame_data, (scale_x, scale_y)


def write_yolo_labels(
    frame_data: dict,
    frames_src_dir: Path,
    yolo_out: Path,
    split: str,
    W: int,
    H: int,
) -> dict:
    """Copy image + write YOLO label for every frame in frame_data."""
    behavior_stats = {b: 0 for b in BEHAVIORS}
    written = 0

    for frame_id, boxes in frame_data.items():
        img_name = f"frame_{frame_id:06d}.jpg"
        img_src = frames_src_dir / img_name
        if not img_src.exists():
            continue

        shutil.copy(img_src, yolo_out / "images" / split / img_name)

        label_path = yolo_out / "labels" / split / img_name.replace(".jpg", ".txt")
        with open(label_path, "w") as f:
            for xtl, ytl, xbr, ybr, behavior in boxes:
                cx = max(0.0, min(1.0, ((xtl + xbr) / 2) / W))
                cy = max(0.0, min(1.0, ((ytl + ybr) / 2) / H))
                bw = max(0.0, min(1.0, (xbr - xtl) / W))
                bh = max(0.0, min(1.0, (ybr - ytl) / H))
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                behavior_stats[behavior] = behavior_stats.get(behavior, 0) + 1
        written += 1

    return behavior_stats, written


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of version folders for validation (default 0.2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_w", type=int, default=1280)
    parser.add_argument("--target_h", type=int, default=720)
    args = parser.parse_args()

    random.seed(args.seed)
    W, H = args.target_w, args.target_h
    target_size = (W, H)

    print(f"\n🔄 CVAT XML → YOLO Conversion")
    print(f"   Version folders: {[v.name for v in VERSION_FOLDERS]}")
    print(f"   Val split: {args.val_split} (video-level)\n")

    # ── Video-level train/val split ──────────────────────────────────────────
    all_versions = list(VERSION_FOLDERS)
    random.shuffle(all_versions)
    n_val = max(1, int(len(all_versions) * args.val_split))
    val_versions  = set(v.name for v in all_versions[:n_val])
    train_versions = set(v.name for v in all_versions[n_val:])

    (SPLITS_DIR).mkdir(parents=True, exist_ok=True)
    (SPLITS_DIR / "train_versions.txt").write_text("\n".join(sorted(train_versions)))
    (SPLITS_DIR / "val_versions.txt").write_text("\n".join(sorted(val_versions)))

    print(f"  Train versions : {sorted(train_versions)}")
    print(f"  Val   versions : {sorted(val_versions)}")

    # ── Process each version folder ─────────────────────────────────────────
    global_stats = {
        "train": {"frames": 0, "behaviors": {b: 0 for b in BEHAVIORS}},
        "val":   {"frames": 0, "behaviors": {b: 0 for b in BEHAVIORS}},
    }

    for vdir in VERSION_FOLDERS:
        xml_files = sorted(vdir.glob("*.xml"))
        if not xml_files:
            print(f"  ⚠️  No XML in {vdir.name} — skipping")
            continue

        xml_path = xml_files[0]
        split = "val" if vdir.name in val_versions else "train"

        # Find the frames directory for this version
        # (may have sub-dirs per video stem; take first)
        version_frames_root = FRAMES_DIR / vdir.name
        stem_dirs = [d for d in version_frames_root.iterdir() if d.is_dir()] \
            if version_frames_root.exists() else []
        frames_dir = stem_dirs[0] if stem_dirs else version_frames_root

        frame_data, (sx, sy) = parse_cvat_xml(xml_path, frames_dir,
                                               target_size=target_size)

        b_stats, n_written = write_yolo_labels(
            frame_data, frames_dir, YOLO_DIR, split, W, H
        )

        print(f"  ✅ [{vdir.name}] → {split} | {n_written} frames | behaviors: {b_stats}")

        global_stats[split]["frames"] += n_written
        for b, cnt in b_stats.items():
            global_stats[split]["behaviors"][b] = \
                global_stats[split]["behaviors"].get(b, 0) + cnt

    # ── dataset.yaml ────────────────────────────────────────────────────────
    yaml_content = f"""# Cattle Calf Detection Dataset — auto-generated
path: {str(YOLO_DIR.absolute())}
train: images/train
val: images/val

nc: 1
names:
  0: calf
"""
    (YOLO_DIR / "dataset.yaml").write_text(yaml_content)

    # ── dataset_stats.json ──────────────────────────────────────────────────
    stats_out = {
        "train_versions": sorted(train_versions),
        "val_versions": sorted(val_versions),
        "train_frames": global_stats["train"]["frames"],
        "val_frames": global_stats["val"]["frames"],
        "behavior_distribution": global_stats,
        "target_resolution": f"{W}x{H}",
    }
    (YOLO_DIR / "dataset_stats.json").write_text(json.dumps(stats_out, indent=2))

    print(f"\n✅ YOLO dataset ready at: {YOLO_DIR}")
    print(f"   Train: {global_stats['train']['frames']} frames")
    print(f"   Val  : {global_stats['val']['frames']} frames")
    print(f"   Behavior dist (train): {global_stats['train']['behaviors']}")


if __name__ == "__main__":
    main()
