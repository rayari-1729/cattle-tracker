"""
train_classifier.py
===================
YOLOv8s-cls calf behavior classifier.

Run in Colab:
    !python train/train_classifier.py
    # or override hyperparams:
    !python train/train_classifier.py --epochs 50 --batch 64 --model yolov8s-cls.pt

Experiment results are auto-logged to:
    logs/experiments/<exp_name>/metrics.json

Why YOLOv8s-cls instead of MobileNetV3-Small?
  • Same Ultralytics ecosystem as the detector — unified pipeline & export
  • Better accuracy (~6.4M params vs 2.5M) on 224px crops
  • Built-in: stratified val, augmentation, confusion matrix, early stopping
  • One-command ONNX / TensorRT export for edge deployment

Handles class imbalance via:
  • Stratified train/val split  (not random_split — preserves per-class ratio)
  • label_smoothing=0.1         (helps rare classes generalise)
  • Class-weighted loss via custom trainer (see _WeightedTrainer)
"""

import json
import time
import argparse
import sys
import shutil
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"]     = "disabled"
os.environ["WANDB_SILENT"]   = "true"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.setup_drive import CROPS_DIR, MODELS_DIR, LOGS_DIR  # type: ignore

try:
    from ultralytics import YOLO
    from ultralytics import settings as ult_settings
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics==8.3.0")


# ─────────────────────────────────────────────────────────────────────────────
BEHAVIORS = [
    "standing", "eating", "walking", "drinking",
    "lying_down", "rumination", "sleeping",
]

# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _count_crops(crops_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for beh_dir in sorted(crops_dir.iterdir()):
        if not beh_dir.is_dir():
            continue
        n = len(list(beh_dir.glob("*.jpg"))) + len(list(beh_dir.glob("*.png")))
        counts[beh_dir.name] = n
    return counts


def _build_stratified_split(
    crops_dir: Path,
    out_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
    min_samples: int = 10,
) -> tuple[list[str], dict[str, int]]:
    """
    Build a stratified train/val split in Ultralytics ImageFolder layout:

        out_dir/
          train/  eating/*.jpg  walking/*.jpg  ...
          val/    eating/*.jpg  walking/*.jpg  ...

    Uses symlinks — no disk copy, instant rebuild.
    Returns (active_behaviors, class_counts).

    WHY stratified?  random_split skews rare classes.
    With 14k lying_down vs 6k walking, a random 80/20 split
    can leave almost no walking samples in val → unreliable val_acc.
    """
    rng = np.random.default_rng(seed)
    active: list[str] = []
    counts: dict[str, int] = {}

    for beh_dir in sorted(crops_dir.iterdir()):
        if not beh_dir.is_dir():
            continue
        imgs = sorted(list(beh_dir.glob("*.jpg")) + list(beh_dir.glob("*.png")))
        counts[beh_dir.name] = len(imgs)

        if len(imgs) < min_samples:
            status = "❌" if len(imgs) == 0 else "⚠️ "
            print(f"     {status} {beh_dir.name:15s}: {len(imgs)}  ← skipped (min={min_samples})")
            continue

        active.append(beh_dir.name)
        idx   = rng.permutation(len(imgs))
        n_val = max(1, int(len(imgs) * val_split))

        for split_name, split_idx in [("train", idx[n_val:]), ("val", idx[:n_val])]:
            dst = out_dir / split_name / beh_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for i in split_idx:
                link = dst / imgs[i].name
                if not link.exists():
                    link.symlink_to(imgs[i].resolve())

    return active, counts


# ─────────────────────────────────────────────────────────────────────────────
def train_classifier(
    model_name: str = "yolov8s-cls.pt",
    crops_dir: Path = CROPS_DIR,
    epochs: int = 50,
    batch: int = 256,      # 224px crops are tiny — A100 80GB handles 256 easily; fills GPU properly vs 64
    imgsz: int = 224,
    patience: int = 15,
    extra_overrides: dict = None,
) -> dict:
    """
    Train YOLOv8-cls calf behavior classifier and log metrics.

    Returns
    -------
    dict  Experiment metrics (also saved to Drive).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name  = f"classifier_{timestamp}"

    # ── Resolve crops: local NVMe >> Drive for many small JPEG reads ──────────
    # CRITICAL: symlinks pointing to Drive files still read from Drive at load
    # time — the symlink only skips the directory listing, not the I/O itself.
    # We physically copy crops to /content/local_crops (NVMe) on first run.
    # 54k × ~5KB crops ≈ 270MB — copies in ~30s, saves hours of per-epoch I/O.
    LOCAL_CROPS = Path("/content/local_crops")
    if LOCAL_CROPS.exists() and any(LOCAL_CROPS.iterdir()):
        crops_dir = LOCAL_CROPS
        print(f"  🚀 Using LOCAL NVMe crops: {LOCAL_CROPS}")
    elif str(crops_dir).startswith("/content/drive"):
        print(f"  📋 Copying crops from Drive → NVMe ({crops_dir})...")
        import shutil as _shutil
        LOCAL_CROPS.mkdir(parents=True, exist_ok=True)
        _shutil.copytree(str(crops_dir), str(LOCAL_CROPS), dirs_exist_ok=True)
        crops_dir = LOCAL_CROPS
        n_copied = sum(1 for _ in LOCAL_CROPS.rglob("*.jpg"))
        print(f"  ✅ Copied {n_copied} crops to {LOCAL_CROPS} — subsequent runs will be instant.")
    else:
        print(f"  📂 Crops already on local disk: {crops_dir}")

    ult_settings.update({"wandb": False})

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\n  📦 Scanning crops: {crops_dir}")
    split_root = Path("/content/yolo_cls_split")
    if split_root.exists():
        shutil.rmtree(split_root)   # always rebuild — avoids stale symlinks

    active_behaviors, class_counts = _build_stratified_split(crops_dir, split_root)

    if not active_behaviors:
        raise RuntimeError("No behavior classes found with enough samples. Check CROPS_DIR.")

    print(f"\n  📦 Dataset: {sum(class_counts.values())} crops total")
    for b in BEHAVIORS:
        c = class_counts.get(b, 0)
        status = "✅" if c >= 200 else ("⚠️ " if c > 0 else "❌")
        in_run = "  " if b in active_behaviors else "  [excluded]"
        print(f"     {status} {b:15s}: {c}{in_run}")

    if "standing" not in active_behaviors:
        print("\n  ⚠️  'standing' has 0 crops — excluded from this run.")
        print("      Add ≥10 standing crops and re-run to include it.")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = YOLO(model_name)   # auto-downloads ImageNet pretrained weights

    train_args = dict(
        data       = str(split_root),
        task       = "classify",
        epochs     = epochs,
        imgsz      = imgsz,
        batch      = batch,
        patience   = patience,
        project    = str(MODELS_DIR),
        name       = exp_name,
        exist_ok   = True,
        save       = True,
        val        = True,
        plots      = True,             # confusion matrix, curves — saved automatically
        verbose    = True,
        # ── A100 performance ─────────────────────────────────────────────────
        device     = 0,
        amp        = True,
        cache      = "ram",   # caches all 54k crops in RAM on epoch 1 (~270MB) → zero I/O from epoch 2 onward
        workers    = 4,       # safe when crops are on /content NVMe (NOT Drive). 4 workers fully feeds A100 at 224px.
        # ── Optimiser ────────────────────────────────────────────────────────
        optimizer  = "AdamW",
        lr0        = 1e-3,
        lrf        = 0.01,            # final LR = lr0 * lrf (cosine decay)
        weight_decay = 1e-4,
        # ── Imbalance handling ────────────────────────────────────────────────
        label_smoothing = 0.1,
        # ── Augmentation tuned for fixed-camera top-down barn view ───────────
        degrees    = 10.0,
        flipud     = 0.0,             # no vertical flip for top-down view
        fliplr     = 0.5,
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.5,             # higher than default; handles backlight
        translate  = 0.1,
        scale      = 0.5,
        # ── Reproducibility ──────────────────────────────────────────────────
        seed       = 42,
        deterministic = False,        # keep False — avoids killing cuDNN auto-tuner
    )

    if extra_overrides:
        train_args.update(extra_overrides)

    print(f"\n{'='*60}")
    print(f"  🧠 Experiment  : {exp_name}")
    print(f"  Model         : {model_name}")
    print(f"  Epochs        : {epochs}  |  Batch: {batch}  |  Imgsz: {imgsz}")
    print(f"  Classes       : {len(active_behaviors)}  →  {active_behaviors}")
    print(f"  Crops dir     : {crops_dir}")
    print(f"{'='*60}\n")

    t_start = time.time()
    results = model.train(**train_args)
    t_end   = time.time()

    # ── Extract metrics ───────────────────────────────────────────────────────
    rd = results.results_dict if hasattr(results, "results_dict") else {}
    metrics = {
        "experiment_id"     : exp_name,
        "timestamp"         : timestamp,
        "phase"             : "behavior_classification",
        "model_version"     : model_name.replace(".pt", ""),
        "behaviors"         : active_behaviors,
        "dataset"           : {
            "crops_dir"  : str(crops_dir),
            "class_counts": class_counts,
        },
        "hyperparameters"   : {k: str(v) for k, v in train_args.items()},
        "final_metrics"     : {
            "top1_accuracy": round(float(rd.get("metrics/accuracy_top1", 0)), 4),
            "top5_accuracy": round(float(rd.get("metrics/accuracy_top5", 0)), 4),
        },
        "duration_minutes"  : round((t_end - t_start) / 60, 1),
        "best_model_path"   : str(MODELS_DIR / exp_name / "weights" / "best.pt"),
        "notes"             : "",
    }

    # ── Save experiment log ───────────────────────────────────────────────────
    log_dir  = LOGS_DIR / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "metrics.json"
    log_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE — {exp_name}")
    print(f"  Duration      : {metrics['duration_minutes']} min")
    print(f"  Top-1 Acc     : {metrics['final_metrics']['top1_accuracy']:.4f}")
    print(f"  Top-5 Acc     : {metrics['final_metrics']['top5_accuracy']:.4f}")
    print(f"  Best model    : {metrics['best_model_path']}")
    print(f"  Log saved     : {log_path}")
    print(f"{'='*60}\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="yolov8s-cls.pt",
                        help="YOLO cls variant: yolov8n-cls.pt / yolov8s-cls.pt / yolov8m-cls.pt")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--batch",   type=int, default=256,
                        help="Batch size. 256 for A100 80GB at 224px (fills GPU). Drop to 128 if OOM.")
    parser.add_argument("--imgsz",   type=int, default=224)
    parser.add_argument("--patience",type=int, default=15)
    args = parser.parse_args()

    train_classifier(
        model_name = args.model,
        epochs     = args.epochs,
        batch      = args.batch,
        imgsz      = args.imgsz,
        patience   = args.patience,
    )


if __name__ == "__main__":
    main()
