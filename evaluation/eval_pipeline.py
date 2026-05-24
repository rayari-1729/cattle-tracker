"""
eval_pipeline.py
================
Comprehensive evaluation suite:
  • Detector metrics  (from YOLO val)
  • Classifier metrics (confusion matrix, per-class report)
  • Experiment comparison table (all runs in logs/)

Usage:
    python evaluation/eval_pipeline.py --phase all
    python evaluation/eval_pipeline.py --phase detector
    python evaluation/eval_pipeline.py --phase classifier --model_dir <path>
"""

import json
import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless (Colab compatible)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, balanced_accuracy_score
)
from torch.utils.data import DataLoader
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.00_setup_drive import LOGS_DIR, CROPS_DIR, MODELS_DIR  # type: ignore
from train.train_classifier import (                                  # type: ignore
    BehaviorClassifier, BehaviorCropDataset, VAL_TRANSFORM, BEHAVIORS
)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ─────────────────────────────────────────────────────────────────────────────
# Classifier Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_classifier(model_path: Path, device: str = "cuda", batch_size: int = 64):
    """Load classifier and run full evaluation on the entire crops dir."""
    print(f"\n🔍 Classifier Evaluation — {model_path.name}")

    model = BehaviorClassifier()
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model = model.to(device)
    model.eval()

    dataset = BehaviorCropDataset(CROPS_DIR, transform=VAL_TRANSFORM)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outs = model(imgs)
            preds = outs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Confusion matrix
    cm    = confusion_matrix(all_labels, all_preds, labels=list(range(len(BEHAVIORS))))
    cm_n  = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=BEHAVIORS, yticklabels=BEHAVIORS, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Behavior Classification — Normalized Confusion Matrix")
    plt.tight_layout()

    out_dir = model_path.parent
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Classification report
    report = classification_report(
        all_labels, all_preds, target_names=BEHAVIORS, output_dict=True
    )
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    print(classification_report(all_labels, all_preds, target_names=BEHAVIORS))
    print(f"  Balanced accuracy: {bal_acc:.4f}")

    report_path = out_dir / "classification_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"  Report saved: {report_path}")
    print(f"  Confusion matrix saved: {out_dir / 'confusion_matrix.png'}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# Experiment Comparison Table
# ─────────────────────────────────────────────────────────────────────────────
def summarize_experiments(log_root: Path = LOGS_DIR):
    """Print a comparison table of all experiment runs."""
    print(f"\n📊 Experiment Summary — {log_root}\n")

    rows = []
    for exp_dir in sorted(log_root.iterdir()):
        mf = exp_dir / "metrics.json"
        if not mf.exists():
            continue
        m    = json.loads(mf.read_text())
        phase = m.get("phase", m.get("task", "?"))

        if phase == "detection":
            fm = m.get("final_metrics", {})
            rows.append([
                m.get("experiment_id", exp_dir.name)[:40],
                "detect",
                m.get("model_version", "?"),
                f"{fm.get('mAP50',    0):.4f}",
                f"{fm.get('mAP50_95', 0):.4f}",
                f"{fm.get('precision',0):.4f}",
                f"{fm.get('recall',   0):.4f}",
                f"{m.get('duration_minutes', 0):.1f}m",
            ])
        elif phase == "behavior_classification":
            rows.append([
                m.get("experiment_id", exp_dir.name)[:40],
                "classify",
                m.get("model", "?"),
                "—", "—", "—",
                f"acc={m.get('best_val_accuracy', 0):.4f}",
                f"{m.get('duration_minutes', 0):.1f}m",
            ])

    headers = ["Experiment", "Phase", "Model", "mAP50", "mAP50-95", "Prec", "Recall/Acc", "Time"]

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print("\t".join(headers))
        for r in rows:
            print("\t".join(str(x) for x in r))


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["detector", "classifier", "summary", "all"],
        default="summary",
    )
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to experiment output dir with best_classifier.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.phase in ("summary", "all"):
        summarize_experiments()

    if args.phase in ("classifier", "all"):
        if args.model_dir:
            model_path = Path(args.model_dir) / "best_classifier.pth"
        else:
            # Find most recent classifier
            classifier_dirs = sorted(
                [d for d in MODELS_DIR.iterdir() if d.name.startswith("classifier_")]
            )
            if not classifier_dirs:
                print("No classifier checkpoints found. Train first.")
                return
            model_path = classifier_dirs[-1] / "best_classifier.pth"

        evaluate_classifier(model_path, device=args.device)


if __name__ == "__main__":
    main()
