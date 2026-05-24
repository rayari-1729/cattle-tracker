"""
01_extract_frames.py
====================
Extracts ONLY the annotated frames from each video version folder.

Supports v1, v2, … vN folder layout:
  data/
  ├── v1/
  │   ├── <video_name>.mp4
  │   └── annotations.xml
  ├── v2/
  │   ├── <video_name>.mp4
  │   └── annotations.xml
  └── ...

Output: processed/frames/v1/<video_stem>/frame_XXXXXX.jpg
        processed/frames/v2/<video_stem>/frame_XXXXXX.jpg

Usage:
    python scripts/01_extract_frames.py --max_versions 10 --width 1280 --height 720
"""

import cv2
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import json
import sys

# Allow running from experiment/ root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.setup_drive import FRAMES_DIR, VERSION_FOLDERS   # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
def extract_frames_for_version(
    version_dir: Path,
    out_root: Path,
    resize: tuple[int, int] = (1280, 720),
    jpeg_quality: int = 95,
) -> dict:
    """
    Extract annotated frames from all mp4s in a version folder.

    Returns a stats dict per video.
    """
    mp4s = sorted(version_dir.glob("*.mp4"))
    xmls = sorted(version_dir.glob("*.xml"))

    if not mp4s or not xmls:
        raise FileNotFoundError(f"Missing mp4 or xml in {version_dir}")

    # Match mp4 to xml: prefer same stem, fallback to first xml found
    xml_path = xmls[0]  # CVAT exports one XML with all tracks
    if len(xmls) > 1:
        print(f"  ⚠️  Multiple XMLs found in {version_dir.name}; using: {xml_path.name}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Collect annotated frame IDs (across all tracks)
    frame_ids: set[int] = set()
    for track in root.findall("track"):
        for box in track.findall("box"):
            if box.get("outside") == "0":
                frame_ids.add(int(box.get("frame")))

    # Original resolution from XML meta
    try:
        orig_w = int(root.find("meta/original_size/width").text)
        orig_h = int(root.find("meta/original_size/height").text)
    except (AttributeError, TypeError):
        # Fallback — will infer from first frame
        orig_w, orig_h = None, None

    version_stats = {"version": version_dir.name, "videos": []}

    for mp4 in mp4s:
        stem = mp4.stem
        out_dir = out_root / version_dir.name / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(mp4))
        if orig_w is None:
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        scale_x = resize[0] / orig_w
        scale_y = resize[1] / orig_h

        frame_idx = 0
        saved = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in frame_ids:
                frame_resized = cv2.resize(frame, resize)
                out_file = out_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(
                    str(out_file),
                    frame_resized,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                saved += 1
            frame_idx += 1

        cap.release()

        video_stat = {
            "video": mp4.name,
            "orig_resolution": f"{orig_w}x{orig_h}",
            "target_resolution": f"{resize[0]}x{resize[1]}",
            "total_frames_in_video": frame_idx,
            "annotated_frame_ids": len(frame_ids),
            "frames_saved": saved,
            "scale_x": round(scale_x, 6),
            "scale_y": round(scale_y, 6),
            "output_dir": str(out_dir),
        }
        version_stats["videos"].append(video_stat)
        print(
            f"  ✅ [{version_dir.name}] {mp4.name}: "
            f"{saved}/{len(frame_ids)} frames → {out_dir}"
        )

    # Save stats JSON alongside frames
    stats_path = out_root / version_dir.name / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(version_stats, f, indent=2)

    return version_stats


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Extract annotated frames for all version folders")
    parser.add_argument("--max_versions", type=int, default=10,
                        help="Maximum number of version folders to process (default: 10)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 0-100")
    args = parser.parse_args()

    resize = (args.width, args.height)
    print(f"\n📽️  Frame Extraction — Target: {resize[0]}×{resize[1]} JPEG Q{args.quality}")
    print(f"   Processing {len(VERSION_FOLDERS)} version folders...\n")

    all_stats = []
    for vdir in VERSION_FOLDERS:
        try:
            stats = extract_frames_for_version(vdir, FRAMES_DIR, resize, args.quality)
            all_stats.append(stats)
        except Exception as e:
            print(f"  ❌ Error processing {vdir.name}: {e}")

    # Global summary
    total_saved = sum(
        vid["frames_saved"]
        for s in all_stats
        for vid in s["videos"]
    )
    print(f"\n✅ Done. Total frames extracted: {total_saved}")
    print(f"   Output root: {FRAMES_DIR}")


if __name__ == "__main__":
    main()
