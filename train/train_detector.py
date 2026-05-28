"""
train_detector.py
=================
YOLOv8 calf detector training.

Run in Colab:
    !python train/train_detector.py
    # or override hyperparams:
    !python train/train_detector.py --epochs 50 --batch 4 --model yolov8s.pt

Experiment results are auto-logged to:
    logs/experiments/<exp_name>/metrics.json
"""

import json
import time
import argparse
import sys
from datetime import datetime
from pathlib import Path
import os

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"]     = "disabled"
os.environ["WANDB_SILENT"]   = "true"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.setup_drive import YOLO_DIR, MODELS_DIR, LOGS_DIR  # type: ignore

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics==8.3.0")


# ─────────────────────────────────────────────────────────────────────────────
def train_detector(
    model_name: str = "yolov8s.pt",
    epochs: int = 100,
    batch: int = 8,
    imgsz: int = 1280,
    patience: int = 20,
    extra_overrides: dict = None,
) -> dict:
    """
    Train YOLOv8 calf detector and log metrics.

    Returns
    -------
    dict  Experiment metrics (also saved to Drive).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name  = f"detector_{timestamp}"

    yaml_path = YOLO_DIR / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"dataset.yaml not found at {yaml_path}. "
            "Run 02_cvat_to_yolo.py first."
        )

    model = YOLO(model_name)

    # Persist wandb=off to ultralytics settings
    from ultralytics import settings as ult_settings
    ult_settings.update({"wandb": False})

    train_args = dict(
        data       = str(yaml_path),
        epochs     = epochs,
        imgsz      = imgsz,
        batch      = batch,
        patience   = patience,
        project    = str(MODELS_DIR),
        name       = exp_name,
        exist_ok   = True,
        save       = True,
        save_period= 10,
        val        = True,
        plots      = True,
        verbose    = True,
        # Augmentation tuned for indoor fixed-camera top-down view
        degrees    = 5.0,
        flipud     = 0.0,
        fliplr     = 0.5,
        mosaic     = 0.5,
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.5,      # higher than default; handles backlight
        translate  = 0.1,
        scale      = 0.5,
        shear      = 2.0,
        perspective= 0.0005,
        # Reproducibility
        seed       = 42,
        deterministic = True,
    )

    if extra_overrides:
        train_args.update(extra_overrides)

    print(f"\n{'='*60}")
    print(f"  🏋️  Experiment  : {exp_name}")
    print(f"  Model         : {model_name}")
    print(f"  Epochs        : {epochs}  |  Batch: {batch}  |  Imgsz: {imgsz}")
    print(f"  Dataset YAML  : {yaml_path}")
    print(f"{'='*60}\n")

    t_start = time.time()
    results = model.train(**train_args)
    t_end   = time.time()

    # ── Extract metrics ──────────────────────────────────────────────────────
    rd = results.results_dict if hasattr(results, "results_dict") else {}
    metrics = {
        "experiment_id"    : exp_name,
        "timestamp"        : timestamp,
        "phase"            : "detection",
        "model_version"    : model_name.replace(".pt", ""),
        "dataset"          : {
            "yaml"     : str(yaml_path),
            "n_versions": len(list(YOLO_DIR.glob("images/train/*.jpg"))) // 100,  # estimate
        },
        "hyperparameters"  : {k: str(v) for k, v in train_args.items()},
        "final_metrics"    : {
            "mAP50"    : round(float(rd.get("metrics/mAP50(B)",    0)), 4),
            "mAP50_95" : round(float(rd.get("metrics/mAP50-95(B)", 0)), 4),
            "precision": round(float(rd.get("metrics/precision(B)",0)), 4),
            "recall"   : round(float(rd.get("metrics/recall(B)",   0)), 4),
        },
        "duration_minutes" : round((t_end - t_start) / 60, 1),
        "best_model_path"  : str(MODELS_DIR / exp_name / "weights" / "best.pt"),
        "notes"            : "",
    }

    # ── Save experiment log ──────────────────────────────────────────────────
    log_dir = LOGS_DIR / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "metrics.json"
    log_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE — {exp_name}")
    print(f"  Duration    : {metrics['duration_minutes']} min")
    print(f"  mAP@0.5     : {metrics['final_metrics']['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {metrics['final_metrics']['mAP50_95']:.4f}")
    print(f"  Precision   : {metrics['final_metrics']['precision']:.4f}")
    print(f"  Recall      : {metrics['final_metrics']['recall']:.4f}")
    print(f"  Log saved   : {log_path}")
    print(f"{'='*60}\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="yolov8s.pt",
                        help="YOLO model variant (yolov8n.pt / yolov8s.pt / yolov8m.pt)")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--batch",   type=int, default=8,
                        help="Batch size. Use 4 if OOM on T4, 16 on A100.")
    parser.add_argument("--imgsz",   type=int, default=1280)
    parser.add_argument("--patience",type=int, default=20)
    args = parser.parse_args()

    train_detector(
        model_name=args.model,
        epochs    =args.epochs,
        batch     =args.batch,
        imgsz     =args.imgsz,
        patience  =args.patience,
    )


if __name__ == "__main__":
    main()