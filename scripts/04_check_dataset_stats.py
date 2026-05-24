"""
04_check_dataset_stats.py
=========================
Audits the processed dataset and prints a rich health report.

Checks:
  • Frame counts per split
  • Behavior distribution + class imbalance warnings
  • Missing YOLO labels
  • Crop counts per behavior
  • Recommendations before training

Usage:
    python scripts/04_check_dataset_stats.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.setup_drive import YOLO_DIR, CROPS_DIR  # type: ignore

BEHAVIORS = [
    "standing", "eating", "walking", "drinking",
    "lying_down", "rumination", "sleeping",
]

MIN_CROPS_FOR_TRAINING = 200
MIN_FRAMES_FOR_TRAINING = 500


def bar(value: int, total: int, width: int = 30) -> str:
    """Simple ASCII bar chart."""
    if total == 0:
        return "[" + " " * width + "]"
    filled = int(width * value / total)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def main():
    print("\n" + "=" * 65)
    print("  🐄  Cattle Analytics — Dataset Health Report")
    print("=" * 65)

    # ── YOLO detection stats ─────────────────────────────────────────────────
    stats_file = YOLO_DIR / "dataset_stats.json"
    if not stats_file.exists():
        print("\n❌ dataset_stats.json not found. Run 02_cvat_to_yolo.py first.")
    else:
        with open(stats_file) as f:
            stats = json.load(f)

        print("\n📊 YOLO Detection Dataset")
        print(f"   Train versions : {stats.get('train_versions', [])}")
        print(f"   Val   versions : {stats.get('val_versions', [])}")
        print(f"   Train frames   : {stats.get('train_frames', 0)}")
        print(f"   Val   frames   : {stats.get('val_frames', 0)}")

        total_f = stats.get("train_frames", 0) + stats.get("val_frames", 0)
        if total_f < MIN_FRAMES_FOR_TRAINING:
            print(f"\n  ⚠️  Only {total_f} total frames. "
                  f"Recommend ≥{MIN_FRAMES_FOR_TRAINING} before training.")

        # Behavior distribution
        dist = stats.get("behavior_distribution", {})
        train_beh = dist.get("train", {}).get("behaviors", {})
        if train_beh:
            print("\n   Behavior distribution (train frames):")
            total_b = sum(train_beh.values())
            for b in BEHAVIORS:
                c = train_beh.get(b, 0)
                pct = 100 * c / max(total_b, 1)
                flag = "⚠️ " if c < 100 else ("❌" if c == 0 else "  ")
                print(f"   {flag} {b:15s}: {c:5d} ({pct:5.1f}%) {bar(c, total_b, 25)}")

    # ── Behavior crop stats ──────────────────────────────────────────────────
    crop_stats_file = CROPS_DIR / "crop_stats.json"
    print("\n📦 Behavior Crop Counts (for Classifier)")

    if not crop_stats_file.exists():
        print("   ❌ crop_stats.json not found. Run 03_generate_crops.py first.")
    else:
        with open(crop_stats_file) as f:
            crop_stats = json.load(f)

        total_c = sum(crop_stats.values())
        for b in BEHAVIORS:
            c = crop_stats.get(b, 0)
            pct = 100 * c / max(total_c, 1)
            status = "✅" if c >= MIN_CROPS_FOR_TRAINING else ("⚠️ " if c > 0 else "❌ MISSING")
            print(f"   {status} {b:15s}: {c:5d} ({pct:5.1f}%) {bar(c, total_c, 25)}")

        print(f"\n   Total crops: {total_c}")

        missing = [b for b in BEHAVIORS if crop_stats.get(b, 0) == 0]
        low     = [b for b in BEHAVIORS if 0 < crop_stats.get(b, 0) < MIN_CROPS_FOR_TRAINING]

        if missing:
            print(f"\n   ❌ Missing classes: {missing}")
            print("      → Annotate videos containing these behaviors before classifier training.")
        if low:
            print(f"\n   ⚠️  Low sample classes (<{MIN_CROPS_FOR_TRAINING}): {low}")
            print("      → Use weighted sampling + augmentation (already coded in train script).")

    # ── YOLO label integrity check ───────────────────────────────────────────
    print("\n🔍 YOLO Label Integrity")
    for split in ["train", "val"]:
        img_dir   = YOLO_DIR / "images" / split
        label_dir = YOLO_DIR / "labels" / split
        if not img_dir.exists():
            print(f"   ⚠️  {split} image dir missing")
            continue
        imgs   = set(p.stem for p in img_dir.glob("*.jpg"))
        labels = set(p.stem for p in label_dir.glob("*.txt"))
        orphan_imgs   = imgs - labels
        orphan_labels = labels - imgs
        print(f"   {split}: {len(imgs)} images | {len(labels)} labels", end="")
        if orphan_imgs:
            print(f" | ⚠️  {len(orphan_imgs)} images without labels", end="")
        if orphan_labels:
            print(f" | ⚠️  {len(orphan_labels)} labels without images", end="")
        print()

    print("\n" + "=" * 65)
    print("  Done. Fix any ❌/⚠️  issues above before training.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
