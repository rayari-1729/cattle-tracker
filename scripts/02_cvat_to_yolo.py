# """
# 02_cvat_to_yolo.py
# ==================
# Converts CVAT XML annotations from ALL version folders into YOLO detection format.

# Detection task: single class (class_id = 0 = calf).
# Behavior labels are captured separately into a side-car JSON for the classifier.

# Key design decisions:
#   • Train/Val split is done AT VIDEO LEVEL, not frame level.
#     → Prevents temporal leakage (frames from same video spanning both sets).
#   • Version folders (v1..v10) are split proportionally.
#   • Bounding box coordinates are scaled from original resolution to target resolution.

# Output:
#   processed/
#   ├── yolo_detection/
#   │   ├── images/train/   ← JPEG frames
#   │   ├── images/val/
#   │   ├── labels/train/   ← YOLO .txt files
#   │   ├── labels/val/
#   │   ├── dataset.yaml
#   │   └── dataset_stats.json
#   └── splits/
#       ├── train_versions.txt
#       └── val_versions.txt

# Usage:
#     python scripts/02_cvat_to_yolo.py --val_split 0.2
# """

# import xml.etree.ElementTree as ET
# import shutil
# import json
# import argparse
# import random
# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# from scripts.setup_drive import (  # type: ignore
#     VERSION_FOLDERS, FRAMES_DIR, YOLO_DIR, SPLITS_DIR,
# )

# BEHAVIORS = [
#     "standing", "eating", "walking", "drinking",
#     "lying_down", "rumination", "sleeping",
# ]
# BEHAVIOR_TO_IDX = {b: i for i, b in enumerate(BEHAVIORS)}


# # ─────────────────────────────────────────────────────────────────────────────
# def parse_cvat_xml(
#     xml_path: Path,
#     frames_dir: Path,
#     orig_size: tuple[int, int] = (3840, 2160),
#     target_size: tuple[int, int] = (1280, 720),
# ) -> dict[int, list[tuple]]:
#     """
#     Parse a CVAT XML and return frame_data:
#       { frame_id: [(xtl, ytl, xbr, ybr, behavior), ...] }

#     Coordinates are already scaled to target_size.
#     """
#     tree = ET.parse(xml_path)
#     root = tree.getroot()

#     # Try to read orig size from XML meta
#     try:
#         orig_w = int(root.find("meta/original_size/width").text)
#         orig_h = int(root.find("meta/original_size/height").text)
#     except (AttributeError, TypeError):
#         orig_w, orig_h = orig_size

#     scale_x = target_size[0] / orig_w
#     scale_y = target_size[1] / orig_h

#     frame_data: dict[int, list] = {}

#     for track in root.findall("track"):
#         for box in track.findall("box"):
#             if box.get("outside") == "1":
#                 continue
#             frame_id = int(box.get("frame"))

#             xtl = float(box.get("xtl")) * scale_x
#             ytl = float(box.get("ytl")) * scale_y
#             xbr = float(box.get("xbr")) * scale_x
#             ybr = float(box.get("ybr")) * scale_y

#             behavior = "standing"
#             for attr in box.findall("attribute"):
#                 if attr.get("name") == "behavior":
#                     behavior = attr.text.strip() if attr.text else "standing"

#             if frame_id not in frame_data:
#                 frame_data[frame_id] = []
#             frame_data[frame_id].append((xtl, ytl, xbr, ybr, behavior))

#     return frame_data, (scale_x, scale_y)


# def write_yolo_labels(
#     frame_data: dict,
#     frames_src_dir: Path,
#     yolo_out: Path,
#     split: str,
#     W: int,
#     H: int,
# ) -> dict:
#     """Copy image + write YOLO label for every frame in frame_data."""
#     behavior_stats = {b: 0 for b in BEHAVIORS}
#     written = 0

#     for frame_id, boxes in frame_data.items():
#         img_name = f"frame_{frame_id:06d}.jpg"
#         img_src = frames_src_dir / img_name
#         if not img_src.exists():
#             continue

#         shutil.copy(img_src, yolo_out / "images" / split / img_name)

#         label_path = yolo_out / "labels" / split / img_name.replace(".jpg", ".txt")
#         with open(label_path, "w") as f:
#             for xtl, ytl, xbr, ybr, behavior in boxes:
#                 cx = max(0.0, min(1.0, ((xtl + xbr) / 2) / W))
#                 cy = max(0.0, min(1.0, ((ytl + ybr) / 2) / H))
#                 bw = max(0.0, min(1.0, (xbr - xtl) / W))
#                 bh = max(0.0, min(1.0, (ybr - ytl) / H))
#                 f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
#                 behavior_stats[behavior] = behavior_stats.get(behavior, 0) + 1
#         written += 1

#     return behavior_stats, written


# # ─────────────────────────────────────────────────────────────────────────────
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--val_split", type=float, default=0.2,
#                         help="Fraction of version folders for validation (default 0.2)")
#     parser.add_argument("--seed", type=int, default=42)
#     parser.add_argument("--target_w", type=int, default=1280)
#     parser.add_argument("--target_h", type=int, default=720)
#     args = parser.parse_args()

#     random.seed(args.seed)
#     W, H = args.target_w, args.target_h
#     target_size = (W, H)

#     print(f"\n🔄 CVAT XML → YOLO Conversion")
#     print(f"   Version folders: {[v.name for v in VERSION_FOLDERS]}")
#     print(f"   Val split: {args.val_split} (video-level)\n")

#     # ── Video-level train/val split ──────────────────────────────────────────
#     all_versions = list(VERSION_FOLDERS)
#     random.shuffle(all_versions)
#     n_val = max(1, int(len(all_versions) * args.val_split))
#     ### Debug ###
#     # val_versions  = set(v.name for v in all_versions[:n_val])
#     # train_versions = set(v.name for v in all_versions[n_val:])

#     # Hardcoded video-level split — v7 is ~20% of total frames
#     VAL_VERSIONS  = {"v7"}
#     TRAIN_VERSIONS = {v.name for v in all_versions if v.name not in VAL_VERSIONS}

#     val_versions   = VAL_VERSIONS
#     train_versions = TRAIN_VERSIONS
#     ############### Debig End ######

#     (SPLITS_DIR).mkdir(parents=True, exist_ok=True)
#     (SPLITS_DIR / "train_versions.txt").write_text("\n".join(sorted(train_versions)))
#     (SPLITS_DIR / "val_versions.txt").write_text("\n".join(sorted(val_versions)))

#     print(f"  Train versions : {sorted(train_versions)}")
#     print(f"  Val   versions : {sorted(val_versions)}")

#     # ── Process each version folder ─────────────────────────────────────────
#     global_stats = {
#         "train": {"frames": 0, "behaviors": {b: 0 for b in BEHAVIORS}},
#         "val":   {"frames": 0, "behaviors": {b: 0 for b in BEHAVIORS}},
#     }

#     for vdir in VERSION_FOLDERS:
#         xml_files = sorted(vdir.glob("*.xml"))
#         if not xml_files:
#             print(f"  ⚠️  No XML in {vdir.name} — skipping")
#             continue

#         xml_path = xml_files[0]
#         split = "val" if vdir.name in val_versions else "train"

#         # Find the frames directory for this version
#         # (may have sub-dirs per video stem; take first)
#         version_frames_root = FRAMES_DIR / vdir.name
#         stem_dirs = [d for d in version_frames_root.iterdir() if d.is_dir()] \
#             if version_frames_root.exists() else []
#         frames_dir = stem_dirs[0] if stem_dirs else version_frames_root

#         frame_data, (sx, sy) = parse_cvat_xml(xml_path, frames_dir,
#                                                target_size=target_size)

#         b_stats, n_written = write_yolo_labels(
#             frame_data, frames_dir, YOLO_DIR, split, W, H
#         )

#         print(f"  ✅ [{vdir.name}] → {split} | {n_written} frames | behaviors: {b_stats}")

#         global_stats[split]["frames"] += n_written
#         for b, cnt in b_stats.items():
#             global_stats[split]["behaviors"][b] = \
#                 global_stats[split]["behaviors"].get(b, 0) + cnt

#     # ── dataset.yaml ────────────────────────────────────────────────────────
#     yaml_content = f"""# Cattle Calf Detection Dataset — auto-generated
# path: {str(YOLO_DIR.absolute())}
# train: images/train
# val: images/val

# nc: 1
# names:
#   0: calf
# """
#     (YOLO_DIR / "dataset.yaml").write_text(yaml_content)

#     # ── dataset_stats.json ──────────────────────────────────────────────────
#     stats_out = {
#         "train_versions": sorted(train_versions),
#         "val_versions": sorted(val_versions),
#         "train_frames": global_stats["train"]["frames"],
#         "val_frames": global_stats["val"]["frames"],
#         "behavior_distribution": global_stats,
#         "target_resolution": f"{W}x{H}",
#     }
#     (YOLO_DIR / "dataset_stats.json").write_text(json.dumps(stats_out, indent=2))

#     print(f"\n✅ YOLO dataset ready at: {YOLO_DIR}")
#     print(f"   Train: {global_stats['train']['frames']} frames")
#     print(f"   Val  : {global_stats['val']['frames']} frames")
#     print(f"   Behavior dist (train): {global_stats['train']['behaviors']}")


# if __name__ == "__main__":
#     main()

"""
02_cvat_to_yolo.py
==================
Converts CVAT XML annotations from ALL version folders into YOLO detection format.

Detection task: single class (class_id = 0 = calf).
Behavior labels are captured separately into a side-car JSON for the classifier.

Key design decisions:
  • Train/Val split is done AT VIDEO LEVEL, not frame level.
    → Prevents temporal leakage (frames from same video spanning both sets).
  • Two split modes controlled by --split_mode flag:
      auto   → picks versions by actual frame count to hit --val_split target
      manual → uses --val_versions list (e.g. --val_versions v7)
  • Output filenames prefixed with version name to prevent collisions.
  • Bounding box coordinates are scaled from original resolution to target resolution.

Output:
  processed/
  ├── yolo_detection/
  │   ├── images/train/   ← JPEG frames (named v1_000001.jpg etc.)
  │   ├── images/val/
  │   ├── labels/train/   ← YOLO .txt files
  │   ├── labels/val/
  │   ├── dataset.yaml
  │   └── dataset_stats.json
  └── splits/
      ├── train_versions.txt
      └── val_versions.txt

Usage:
    # Manual mode (default) — fastest, no Drive scans for split
    python scripts/02_cvat_to_yolo.py
    python scripts/02_cvat_to_yolo.py --split_mode manual --val_versions v7

    # Manual with multiple val versions
    python scripts/02_cvat_to_yolo.py --split_mode manual --val_versions v7 v11

    # Auto mode — picks versions by frame count to hit ~20% val
    # (slower — scans Drive to count frames)
    python scripts/02_cvat_to_yolo.py --split_mode auto --val_split 0.2
"""

import xml.etree.ElementTree as ET
import shutil
import json
import argparse
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

# Cache for frame counts — avoids repeated slow Drive scans in auto mode
_frame_count_cache: dict[str, int] = {}


# ─────────────────────────────────────────────────────────────────────────────
def count_version_frames(vdir: Path) -> int:
    """Count actual .jpg frames on disk — cached to avoid repeated Drive scans."""
    if vdir.name not in _frame_count_cache:
        vframes = FRAMES_DIR / vdir.name
        if not vframes.exists():
            _frame_count_cache[vdir.name] = 0
        else:
            _frame_count_cache[vdir.name] = sum(1 for _ in vframes.rglob("*.jpg"))
    return _frame_count_cache[vdir.name]


def select_val_versions_auto(all_versions: list, val_split: float) -> set:
    """
    Automatically pick val versions by frame count.
    Sorts versions by size ascending, adds until val_split% is reached.
    Frame counts cached — Drive scanned once per version total.
    """
    total_frames = sum(count_version_frames(v) for v in all_versions)
    target_val   = total_frames * val_split

    sorted_versions = sorted(all_versions, key=count_version_frames)

    val_versions    = set()
    val_frame_count = 0

    for v in sorted_versions:
        if val_frame_count >= target_val:
            break
        val_versions.add(v.name)
        val_frame_count += count_version_frames(v)  # free — already cached

    total_val_pct = 100 * val_frame_count / max(total_frames, 1)
    print(f"  [auto] Val versions : {sorted(val_versions)}")
    print(f"  [auto] Val frames   : {val_frame_count} / {total_frames} "
          f"({total_val_pct:.1f}%)")

    return val_versions


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
    version_name: str,
) -> dict:
    """
    Copy image + write YOLO label for every frame in frame_data.
    Output filename: {version_name}_{frame_id:06d}.jpg
    e.g. v1_000042.jpg — guarantees no collisions across versions.
    """
    behavior_stats = {b: 0 for b in BEHAVIORS}
    written = 0

    for frame_id, boxes in frame_data.items():
        src_name = f"frame_{frame_id:06d}.jpg"           # original name on disk
        out_name = f"{version_name}_{frame_id:06d}.jpg"  # unique output name

        img_src = frames_src_dir / src_name
        if not img_src.exists():
            continue

        shutil.copy(img_src, yolo_out / "images" / split / out_name)

        label_path = yolo_out / "labels" / split / out_name.replace(".jpg", ".txt")
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
# Incremental processing helpers
# ─────────────────────────────────────────────────────────────────────────────
# Per-version proc records live in YOLO_DIR/.proc/vX.json
# Each record stores: split assignment, frame count, behavior stats.
# This lets us skip already-processed versions AND still merge their
# stats into the final dataset_stats.json correctly.

def _proc_record_path(version_name: str) -> Path:
    return YOLO_DIR / ".proc" / f"{version_name}.json"


def is_converted(version_name: str) -> bool:
    """Return True if this version has a saved proc record (already converted)."""
    return _proc_record_path(version_name).exists()


def load_proc_record(version_name: str) -> dict:
    """Load the saved proc record for an already-converted version."""
    return json.loads(_proc_record_path(version_name).read_text())


def save_proc_record(version_name: str, split: str, n_written: int, b_stats: dict) -> None:
    """Save a proc record after successful conversion."""
    rec_path = _proc_record_path(version_name)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps({
        "version"  : version_name,
        "split"    : split,
        "frames"   : n_written,
        "behaviors": b_stats,
    }, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split_mode", choices=["auto", "manual"], default="manual",
        help=(
            "auto   = pick val versions by frame count to hit --val_split target\n"
            "manual = explicitly pass --val_versions (default: v7)"
        ),
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Target val fraction for auto mode (default: 0.2)",
    )
    parser.add_argument(
        "--val_versions", nargs="+", default=["v7"],
        help="Version(s) to use as val in manual mode (default: v7)",
    )
    parser.add_argument("--target_w", type=int, default=1280)
    parser.add_argument("--target_h", type=int, default=720)
    parser.add_argument(
        "--force", nargs="*", metavar="VERSION",
        help=(
            "Force re-conversion even if already done. "
            "Pass specific versions (e.g. --force v13 v14) or bare --force for ALL."
        ),
    )
    args = parser.parse_args()

    # Resolve force flags
    force_all = args.force is not None and len(args.force) == 0
    force_set = set(args.force) if args.force else set()

    W, H         = args.target_w, args.target_h
    target_size  = (W, H)
    all_versions = list(VERSION_FOLDERS)

    print(f"\n🔄 CVAT XML → YOLO Conversion  [incremental]")
    print(f"   Discovered      : {[v.name for v in all_versions]}")
    print(f"   Split mode      : {args.split_mode}")
    print(f"   Target size     : {W}x{H}\n")

    # ── Partition into already-done vs new/forced ─────────────────────────────
    old_versions = [
        v for v in all_versions
        if is_converted(v.name) and not force_all and v.name not in force_set
    ]
    new_versions = [
        v for v in all_versions
        if not is_converted(v.name) or force_all or v.name in force_set
    ]
    print(f"  Already done : {[v.name for v in old_versions]}  (skipping)")
    print(f"  New / forced : {[v.name for v in new_versions]}\n")

    # ── Pick split assignment for NEW videos only ─────────────────────────────
    # Already-done versions keep their split locked in their proc record.
    if new_versions:
        if args.split_mode == "auto":
            new_val_versions = select_val_versions_auto(new_versions, args.val_split)
        else:
            new_val_versions = set(args.val_versions) & {v.name for v in new_versions}
            print(f"  [manual] New val versions : {sorted(new_val_versions)}")
    else:
        new_val_versions = set()

    # ── Accumulate global stats (old cached + new processed) ─────────────────
    global_stats = {
        "train": {"frames": 0, "behaviors": {b: 0 for b in BEHAVIORS}},
        "val":   {"frames": 0, "behaviors": {b: 0 for b in BEHAVIORS}},
    }

    # Merge stats from already-processed versions (free — just reads tiny JSONs)
    for vdir in old_versions:
        rec = load_proc_record(vdir.name)
        sp  = rec["split"]
        global_stats[sp]["frames"] += rec["frames"]
        for b, cnt in rec.get("behaviors", {}).items():
            global_stats[sp]["behaviors"][b] = \
                global_stats[sp]["behaviors"].get(b, 0) + cnt
        print(f"  ⏭️  [{vdir.name}] → {sp} | {rec['frames']} frames  (cached)")

    # Process new versions
    for vdir in new_versions:
        xml_files = sorted(vdir.glob("*.xml"))
        if not xml_files:
            print(f"  ⚠️  No XML in {vdir.name} — skipping")
            continue

        xml_path = xml_files[0]
        split    = "val" if vdir.name in new_val_versions else "train"

        version_frames_root = FRAMES_DIR / vdir.name
        stem_dirs = [d for d in version_frames_root.iterdir() if d.is_dir()] \
            if version_frames_root.exists() else []
        frames_dir = stem_dirs[0] if stem_dirs else version_frames_root

        frame_data, (sx, sy) = parse_cvat_xml(
            xml_path, frames_dir, target_size=target_size
        )

        b_stats, n_written = write_yolo_labels(
            frame_data, frames_dir, YOLO_DIR, split, W, H,
            version_name=vdir.name,
        )

        save_proc_record(vdir.name, split, n_written, b_stats)  # ← write sentinel
        print(f"  ✅ [{vdir.name}] → {split} | {n_written} frames | behaviors: {b_stats}")

        global_stats[split]["frames"] += n_written
        for b, cnt in b_stats.items():
            global_stats[split]["behaviors"][b] = \
                global_stats[split]["behaviors"].get(b, 0) + cnt

    # ── Rebuild split manifests from ALL proc records ─────────────────────────
    all_train = sorted(
        v.name for v in all_versions
        if is_converted(v.name) and load_proc_record(v.name)["split"] == "train"
    )
    all_val = sorted(
        v.name for v in all_versions
        if is_converted(v.name) and load_proc_record(v.name)["split"] == "val"
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    (SPLITS_DIR / "train_versions.txt").write_text("\n".join(all_train))
    (SPLITS_DIR / "val_versions.txt").write_text("\n".join(all_val))

    # ── dataset.yaml ─────────────────────────────────────────────────────────
    yaml_content = f"""# Cattle Calf Detection Dataset — auto-generated
path: {str(YOLO_DIR.absolute())}
train: images/train
val: images/val

nc: 1
names:
  0: calf
"""
    (YOLO_DIR / "dataset.yaml").write_text(yaml_content)

    # ── dataset_stats.json (always reflects ALL versions) ─────────────────────
    total = global_stats["train"]["frames"] + global_stats["val"]["frames"]
    stats_out = {
        "split_mode"           : args.split_mode,
        "train_versions"       : all_train,
        "val_versions"         : all_val,
        "train_frames"         : global_stats["train"]["frames"],
        "val_frames"           : global_stats["val"]["frames"],
        "train_pct"            : round(100 * global_stats["train"]["frames"] / max(total, 1), 1),
        "val_pct"              : round(100 * global_stats["val"]["frames"]   / max(total, 1), 1),
        "behavior_distribution": global_stats,
        "target_resolution"    : f"{W}x{H}",
    }
    (YOLO_DIR / "dataset_stats.json").write_text(json.dumps(stats_out, indent=2))

    print(f"\n{'='*55}")
    print(f"  ✅ YOLO dataset ready at: {YOLO_DIR}")
    print(f"  Total train : {global_stats['train']['frames']} frames  ({stats_out['train_pct']}%)")
    print(f"  Total val   : {global_stats['val']['frames']} frames  ({stats_out['val_pct']}%)")
    print(f"  Train vers  : {all_train}")
    print(f"  Val   vers  : {all_val}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()