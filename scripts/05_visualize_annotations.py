"""
05_visualize_annotations.py
===========================
Draws bounding boxes + behavior labels onto frames and saves annotated images.
Useful for sanity-checking annotations before training.

Usage:
    python scripts/05_visualize_annotations.py --version v1 --max_frames 20
"""

import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import sys
import random

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.00_setup_drive import FRAMES_DIR, PROCESSED_DIR, VERSION_FOLDERS  # type: ignore

BEHAVIOR_COLORS = {
    "standing":   (0,   220,   0),
    "walking":    (0,   220, 220),
    "eating":     (255, 165,   0),
    "drinking":   (255,  50,  50),
    "lying_down": (180,   0, 180),
    "sleeping":   ( 50,  50, 255),
    "rumination": (255, 180, 200),
    "unknown":    (128, 128, 128),
}


def visualize_version(
    version_dir: Path,
    frames_root: Path,
    out_root: Path,
    max_frames: int = 20,
    target_size: tuple[int, int] = (1280, 720),
):
    xml_files = sorted(version_dir.glob("*.xml"))
    if not xml_files:
        print(f"  ⚠️  No XML in {version_dir.name}")
        return

    xml_path = xml_files[0]
    tree = ET.parse(xml_path)
    root = tree.getroot()

    try:
        orig_w = int(root.find("meta/original_size/width").text)
        orig_h = int(root.find("meta/original_size/height").text)
    except (AttributeError, TypeError):
        orig_w, orig_h = 3840, 2160

    W, H = target_size
    scale_x = W / orig_w
    scale_y = H / orig_h

    # Build frame_id → boxes map
    frame_data: dict[int, list] = {}
    for track in root.findall("track"):
        track_id = track.get("id")
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            fid = int(box.get("frame"))
            xtl = int(float(box.get("xtl")) * scale_x)
            ytl = int(float(box.get("ytl")) * scale_y)
            xbr = int(float(box.get("xbr")) * scale_x)
            ybr = int(float(box.get("ybr")) * scale_y)

            behavior = "unknown"
            for attr in box.findall("attribute"):
                if attr.get("name") == "behavior":
                    behavior = (attr.text or "unknown").strip()

            occluded = box.get("occluded") == "1"

            if fid not in frame_data:
                frame_data[fid] = []
            frame_data[fid].append((xtl, ytl, xbr, ybr, behavior, track_id, occluded))

    # Find frames dir
    version_frames_root = frames_root / version_dir.name
    stem_dirs = [d for d in version_frames_root.iterdir() if d.is_dir()] \
        if version_frames_root.exists() else []
    frames_dir = stem_dirs[0] if stem_dirs else version_frames_root

    # Sample frames
    sample_ids = sorted(random.sample(list(frame_data.keys()),
                                      min(max_frames, len(frame_data))))

    out_dir = out_root / "viz" / version_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for fid in sample_ids:
        img_path = frames_dir / f"frame_{fid:06d}.jpg"
        if not img_path.exists():
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        for xtl, ytl, xbr, ybr, behavior, tid, occluded in frame_data[fid]:
            color = BEHAVIOR_COLORS.get(behavior, BEHAVIOR_COLORS["unknown"])
            thickness = 1 if occluded else 2
            cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), color, thickness)

            label = f"T{tid}:{behavior}"
            if occluded:
                label += "(occ)"
            cv2.putText(frame, label, (xtl, max(ytl - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imwrite(str(out_dir / f"frame_{fid:06d}_viz.jpg"), frame)
        saved += 1

    print(f"  ✅ [{version_dir.name}] Saved {saved} annotated frames → {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None,
                        help="Specific version folder (e.g. v1). Default: all.")
    parser.add_argument("--max_frames", type=int, default=20,
                        help="Max frames to visualize per version")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_root = PROCESSED_DIR

    print(f"\n🖼️  Annotation Visualizer")

    if args.version:
        target = next((v for v in VERSION_FOLDERS if v.name == args.version), None)
        if not target:
            print(f"  ❌ Version '{args.version}' not found in: {[v.name for v in VERSION_FOLDERS]}")
            return
        versions = [target]
    else:
        versions = VERSION_FOLDERS

    for vdir in versions:
        visualize_version(vdir, FRAMES_DIR, out_root, args.max_frames)

    print(f"\n✅ Done. Visualized frames saved to {out_root}/viz/")


if __name__ == "__main__":
    main()
