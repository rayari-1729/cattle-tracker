"""
train_classifier_yolo_v2.py
============================
YOLOv8/YOLO11-cls calf behavior classifier — v2

What's new vs v3
-----------------
1.  Focal loss  — added alongside weighted CE for direct comparison.
    --loss ce          plain cross-entropy (baseline, no class weights)
    --loss weighted_ce inverse-frequency class weights  (v3 behaviour)
    --loss focal       focal loss, no class weights
    --loss focal_weighted  focal loss + class weights  ← recommended start

    FocalLoss(gamma, alpha):
      gamma  controls how hard to focus on uncertain samples.
             gamma=0  → reduces to plain CE.
             gamma=2  → standard focal (Lin et al. 2017).  Good default.
             gamma=3  → more aggressive; helps if easy majority class is
                        still dominating after gamma=2.
      alpha  per-class weight tensor (same inverse-frequency as weighted CE).
             Used only with focal_weighted mode.

    Why focal on top of weighted CE?
      Weighted CE fixes  → how often each CLASS appears in gradient updates.
      Focal fixes        → how hard each SAMPLE is weighted within a class.
      eating/drinking/sleeping all have easy samples (clear posture) AND
      hard samples (ambiguous head angle, partial occlusion from another calf).
      Focal specifically up-weights the hard ones.  Combined they attack
      imbalance from two orthogonal directions.

    Recommended experiment order:
      Run 1: --loss weighted_ce       (v3 baseline, your current best)
      Run 2: --loss focal             (focal only, no class weights)
      Run 3: --loss focal_weighted    (both combined — likely best)
      Compare per-class F1 in metrics.json.

2.  YOLO11 / any future Ultralytics cls model — just pass --model:
      --model yolov8s-cls.pt     (current default)
      --model yolov8m-cls.pt     (bigger, better accuracy, slower)
      --model yolo11s-cls.pt     (YOLO11 — same API, no code change needed)
      --model yolo11m-cls.pt
    The ClassificationTrainer API is identical across all Ultralytics cls
    variants, so focal loss + weighted sampler work unchanged.

All v3 features kept
---------------------
  ✅ WeightedRandomSampler     — oversampling minority classes
  ✅ Video-aware split          — by video ID, not frame (no leakage)
  ✅ save_period=10             — checkpoint every 10 epochs (Colab safety)
  ✅ workers=8/0 auto           — 8 on NVMe, 0 on Drive
  ✅ scale=0.2, erasing=0.3    — safe augmentation for head-based behaviors
  ✅ patience=25                — minority classes need longer to converge
  ✅ Per-class F1 logged        — metrics.json has full per-class breakdown

Run examples:
    # Recommended comparison sequence:
    python train/train_classifier_yolo_v2.py --loss weighted_ce
    python train/train_classifier_yolo_v2.py --loss focal_weighted
    python train/train_classifier_yolo_v2.py --loss focal

    # Try YOLO11 (same flags, just different model):
    python train/train_classifier_yolo_v2.py --model yolo11s-cls.pt --loss focal_weighted

    # Tune focal gamma:
    python train/train_classifier_yolo_v2.py --loss focal_weighted --focal_gamma 3.0
"""

import json, time, argparse, sys, shutil, os, re
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"]     = "disabled"
os.environ["WANDB_SILENT"]   = "true"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.setup_drive import CROPS_DIR, MODELS_DIR, LOGS_DIR  # type: ignore

try:
    from ultralytics import YOLO
    from ultralytics import settings as ult_settings
    from ultralytics.models.yolo.classify import ClassificationTrainer
except ImportError:
    raise ImportError("ultralytics not installed. Run: pip install ultralytics==8.3.0")

try:
    from sklearn.metrics import classification_report
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("  ⚠️  scikit-learn not found — per-class F1 skipped. pip install scikit-learn")

# ─────────────────────────────────────────────────────────────────────────────
BEHAVIORS = [
    "standing", "eating", "walking", "drinking",
    "lying_down", "rumination", "sleeping",
]

LOCAL_CROPS = Path("/content/local_crops")

LOSS_MODES = ("ce", "weighted_ce", "focal", "focal_weighted")


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Multi-class focal loss (Lin et al., 2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma : float
        Focusing parameter.  gamma=0 → plain CE.  gamma=2 is the standard
        default from the original paper and works well here.
        Increase to 3 if the model still ignores minority classes after
        a full run at gamma=2.
    alpha : Tensor | None
        Per-class weight vector of shape (n_classes,).  Pass class_weights
        tensor for focal_weighted mode; None for plain focal.
    label_smoothing : float
        Applied before focal weighting, same as nn.CrossEntropyLoss.

    Implementation notes
    --------------------
    We compute CE via F.cross_entropy (numerically stable), then apply the
    focal modulation factor (1 - p_t)^gamma on the per-sample CE values.
    label_smoothing is handled by F.cross_entropy directly.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma           = gamma
        self.alpha           = alpha          # (C,) or None
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (N, C)  raw model output (pre-softmax)
        targets : (N,)    integer class indices
        """
        # ── Per-sample CE loss (unreduced) ────────────────────────────────────
        # alpha=None here — we apply class weights via the focal alpha term below,
        # NOT via F.cross_entropy weight, to keep the focal modulation correct.
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction      = "none",
            label_smoothing= self.label_smoothing,
        )

        # ── p_t: probability of the correct class ─────────────────────────────
        # Use detached softmax so focal weight doesn't affect gradient of p_t
        with torch.no_grad():
            probs = F.softmax(logits.detach(), dim=1)
            p_t   = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # ── Focal modulation ──────────────────────────────────────────────────
        focal_weight = (1.0 - p_t) ** self.gamma   # (N,)
        focal_loss   = focal_weight * ce_loss       # (N,)

        # ── Per-class alpha weighting (only in focal_weighted mode) ───────────
        if self.alpha is not None:
            alpha_t     = self.alpha.to(logits.device)[targets]   # (N,)
            focal_loss  = alpha_t * focal_loss

        return focal_loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Unified trainer base — swappable loss via loss_mode
# ─────────────────────────────────────────────────────────────────────────────

class _FlexTrainer(ClassificationTrainer):
    """
    Supports four loss modes via loss_mode:
      ce             — plain cross-entropy (Ultralytics default)
      weighted_ce    — inverse-frequency class-weighted CE
      focal          — focal loss, uniform class weights
      focal_weighted — focal loss + inverse-frequency class weights
    """

    def __init__(
        self,
        *args,
        loss_mode: str = "weighted_ce",
        class_weights: torch.Tensor = None,
        focal_gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert loss_mode in LOSS_MODES, f"loss_mode must be one of {LOSS_MODES}"
        self._loss_mode     = loss_mode
        self._class_weights = class_weights   # (C,) float32 tensor or None
        self._focal_gamma   = focal_gamma
        self._loss_fn       = None            # built lazily on first forward pass

    def _build_loss_fn(self, device: torch.device):
        """Build loss function once we know the device."""
        ls = self.args.label_smoothing

        if self._loss_mode == "ce":
            self._loss_fn = nn.CrossEntropyLoss(label_smoothing=ls)

        elif self._loss_mode == "weighted_ce":
            w = self._class_weights.to(device) if self._class_weights is not None else None
            self._loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=ls)

        elif self._loss_mode == "focal":
            self._loss_fn = FocalLoss(
                gamma=self._focal_gamma,
                alpha=None,
                label_smoothing=ls,
            )

        elif self._loss_mode == "focal_weighted":
            alpha = self._class_weights if self._class_weights is not None else None
            self._loss_fn = FocalLoss(
                gamma=self._focal_gamma,
                alpha=alpha,
                label_smoothing=ls,
            )

    def get_loss(self, preds, batch):
        labels = batch["cls"].long()
        if self._loss_fn is None:
            self._build_loss_fn(preds.device)
        return self._loss_fn(preds, labels)


# ─────────────────────────────────────────────────────────────────────────────
# WeightedRandomSampler  (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

def _make_weighted_sampler(dataset) -> WeightedRandomSampler:
    targets      = [s[1] for s in dataset.samples]
    class_counts = np.bincount(targets)
    weights      = [1.0 / class_counts[t] for t in targets]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(weights),
        num_samples=len(dataset),
        replacement=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Video-aware train/val split  (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

def _build_video_aware_split(
    crops_dir: Path,
    out_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
    min_samples: int = 10,
) -> tuple[list[str], dict[str, int]]:
    rng    = np.random.default_rng(seed)
    active: list[str]      = []
    counts: dict[str, int] = {}

    for beh_dir in sorted(crops_dir.iterdir()):
        if not beh_dir.is_dir():
            continue
        imgs = sorted(list(beh_dir.glob("*.jpg")) + list(beh_dir.glob("*.png")))
        counts[beh_dir.name] = len(imgs)

        if len(imgs) < min_samples:
            tag = "❌" if len(imgs) == 0 else "⚠️ "
            print(f"     {tag} {beh_dir.name:15s}: {len(imgs)}  ← skipped")
            continue

        active.append(beh_dir.name)

        vid_to_imgs: dict[str, list[Path]] = defaultdict(list)
        has_vid = False
        for img in imgs:
            m = re.match(r"(v\d+)_", img.name)
            if m:
                vid_to_imgs[m.group(1)].append(img)
                has_vid = True
            else:
                vid_to_imgs["_no_vid_"].append(img)

        if has_vid and len(vid_to_imgs) > 1:
            vids = sorted(vid_to_imgs.keys())
            rng.shuffle(vids)
            n_val   = max(1, int(len(vids) * val_split))
            val_set = set(vids[:n_val])
            trn_set = set(vids[n_val:])
            split_map = {
                "val":   [img for v in val_set for img in vid_to_imgs[v]],
                "train": [img for v in trn_set for img in vid_to_imgs[v]],
            }
            print(f"     📹 {beh_dir.name:15s}: {len(imgs):,} frames | "
                  f"{len(vids)} vids → {len(val_set)} val / {len(trn_set)} train")
        else:
            idx   = rng.permutation(len(imgs))
            n_val = max(1, int(len(imgs) * val_split))
            split_map = {
                "val":   [imgs[i] for i in idx[:n_val]],
                "train": [imgs[i] for i in idx[n_val:]],
            }
            print(f"     🖼️  {beh_dir.name:15s}: {len(imgs):,} frames (random split fallback)")

        for split_name, split_imgs in split_map.items():
            dst = out_dir / split_name / beh_dir.name
            dst.mkdir(parents=True, exist_ok=True)
            for img in split_imgs:
                link = dst / img.name
                if not link.exists():
                    link.symlink_to(img.resolve())

    return active, counts


# ─────────────────────────────────────────────────────────────────────────────
# Class weight computation  (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_class_weights(
    active_behaviors: list[str],
    class_counts: dict[str, int],
    smoothing: float = 0.15,
) -> torch.Tensor:
    counts   = np.array([class_counts[b] for b in active_behaviors], dtype=np.float32)
    smoothed = counts + smoothing * counts.mean()
    weights  = 1.0 / smoothed
    weights  = weights / weights.sum() * len(active_behaviors)
    print(f"\n  ⚖️  Class weights (smoothing={smoothing}):")
    for b, w, c in zip(active_behaviors, weights, counts):
        bar = "█" * int(w * 4)
        print(f"     {b:15s}: {w:.3f}  {bar}  (n={int(c):,})")
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Per-class F1 evaluation  (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────

def _eval_per_class_f1(
    best_pt: Path,
    split_root: Path,
    active_behaviors: list[str],
    imgsz: int,
) -> dict:
    if not HAS_SKLEARN or not best_pt.exists():
        return {}

    print("\n  📊 Running per-class F1 on val split...")
    eval_model = YOLO(str(best_pt))
    val_dir    = split_root / "val"
    cls_to_idx = {c: i for i, c in enumerate(active_behaviors)}
    y_true, y_pred = [], []

    for cls_dir in sorted(val_dir.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in cls_to_idx:
            continue
        imgs_in_class = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png"))
        if not imgs_in_class:
            continue
        preds = eval_model.predict(imgs_in_class, imgsz=imgsz, device=0, verbose=False)
        for p in preds:
            y_true.append(cls_to_idx[cls_dir.name])
            y_pred.append(int(p.probs.top1))

    if not y_true:
        return {}

    report = classification_report(
        y_true, y_pred,
        target_names=active_behaviors,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        b: {
            "precision": round(report[b]["precision"], 3),
            "recall"   : round(report[b]["recall"],    3),
            "f1"       : round(report[b]["f1-score"],  3),
            "support"  : int(report[b]["support"]),
        }
        for b in active_behaviors if b in report
    }

    print(f"\n  Per-class F1 (val):")
    for b, m in per_class.items():
        bar = "█" * int(m["f1"] * 20)
        print(f"     {b:15s}: F1={m['f1']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  n={m['support']}  {bar}")

    return per_class


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(
    model_name:   str   = "yolov8s-cls.pt",
    crops_dir:    Path  = CROPS_DIR,
    epochs:       int   = 80,
    batch:        int   = 64,
    imgsz:        int   = 288,
    patience:     int   = 25,
    loss_mode:    str   = "focal_weighted",   # NEW
    focal_gamma:  float = 2.0,               # NEW
    extra_overrides: dict = None,
) -> dict:

    assert loss_mode in LOSS_MODES, f"loss_mode must be one of {LOSS_MODES}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name  = f"cls_v4_{loss_mode}_{timestamp}"

    # ── NVMe / Drive detection ─────────────────────────────────────────────────
    on_local_nvme = LOCAL_CROPS.exists() and any(LOCAL_CROPS.iterdir())
    if on_local_nvme:
        crops_dir = LOCAL_CROPS
        print(f"  🚀 Using LOCAL NVMe crops: {LOCAL_CROPS}")
    else:
        print(f"  📂 Using Drive crops (slow). Run 'Copy to Local NVMe' cell for 5× faster I/O.")

    ult_settings.update({"wandb": False})

    # ── Dataset split ──────────────────────────────────────────────────────────
    print(f"\n  📦 Scanning crops: {crops_dir}")
    split_root = Path("/content/yolo_cls_split_v4")
    if split_root.exists():
        shutil.rmtree(split_root)

    active_behaviors, class_counts = _build_video_aware_split(crops_dir, split_root)

    if not active_behaviors:
        raise RuntimeError("No behavior classes found with enough samples.")

    print(f"\n  📦 Dataset: {sum(class_counts.values()):,} crops total")
    for b in BEHAVIORS:
        c      = class_counts.get(b, 0)
        status = "✅" if c >= 200 else ("⚠️ " if c > 0 else "❌")
        excl   = "" if b in active_behaviors else "  [excluded]"
        print(f"     {status} {b:15s}: {c:>6,}{excl}")

    # ── Class weights (used by weighted_ce and focal_weighted modes) ───────────
    needs_weights = loss_mode in ("weighted_ce", "focal_weighted")
    class_weights = _compute_class_weights(active_behaviors, class_counts) \
                    if needs_weights else None

    if not needs_weights:
        print(f"\n  ⚖️  loss_mode='{loss_mode}' — no class weights applied.")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = YOLO(model_name)

    n_workers = 8 if on_local_nvme else 0

    train_args = dict(
        data            = str(split_root),
        task            = "classify",
        epochs          = epochs,
        imgsz           = imgsz,
        batch           = batch,
        patience        = patience,
        project         = str(MODELS_DIR),
        name            = exp_name,
        exist_ok        = True,
        save            = True,
        save_period     = 10,
        val             = True,
        plots           = True,
        verbose         = True,
        device          = 0,
        amp             = True,
        cache           = "ram",
        workers         = n_workers,
        optimizer       = "AdamW",
        lr0             = 5e-4,
        lrf             = 0.01,
        weight_decay    = 1e-4,
        label_smoothing = 0.1,
        # Augmentation — fixed-camera top-down barn view
        degrees         = 15.0,
        flipud          = 0.0,
        fliplr          = 0.5,
        hsv_h           = 0.015,
        hsv_s           = 0.7,
        hsv_v           = 0.5,
        translate       = 0.1,
        scale           = 0.2,
        erasing         = 0.3,
        seed            = 42,
        deterministic   = False,
    )

    if extra_overrides:
        train_args.update(extra_overrides)

    print(f"\n{'='*60}")
    print(f"  🧠 Experiment   : {exp_name}")
    print(f"  Model          : {model_name}")
    print(f"  Loss mode      : {loss_mode}"
          + (f"  (gamma={focal_gamma})" if "focal" in loss_mode else ""))
    print(f"  Epochs         : {epochs}  |  Batch: {batch}  |  Imgsz: {imgsz}")
    print(f"  Patience       : {patience}  |  save_period: 10  |  workers: {n_workers}")
    print(f"  Classes ({len(active_behaviors)})    : {active_behaviors}")
    print(f"{'='*60}\n")

    # ── Build patched trainer with chosen loss + WeightedRandomSampler ─────────
    _loss_mode    = loss_mode
    _class_weights = class_weights
    _focal_gamma  = focal_gamma

    class _PatchedTrainer(_FlexTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(
                *args,
                loss_mode     = _loss_mode,
                class_weights = _class_weights,
                focal_gamma   = _focal_gamma,
                **kwargs,
            )

        def get_dataloader(self, dataset_path, batch_size, rank=0, mode="train"):
            loader = super().get_dataloader(dataset_path, batch_size, rank, mode)
            if mode == "train":
                sampler = _make_weighted_sampler(loader.dataset)
                loader  = torch.utils.data.DataLoader(
                    loader.dataset,
                    batch_size  = batch_size,
                    sampler     = sampler,
                    num_workers = loader.num_workers,
                    pin_memory  = loader.pin_memory,
                    collate_fn  = loader.collate_fn,
                )
                print(f"  🎲 WeightedRandomSampler active  |  loss={_loss_mode}")
            return loader

    t_start = time.time()
    results  = model.train(trainer=_PatchedTrainer, **train_args)
    t_end    = time.time()

    # ── Metrics ────────────────────────────────────────────────────────────────
    rd = results.results_dict if hasattr(results, "results_dict") else {}
    best_pt = Path(MODELS_DIR) / exp_name / "weights" / "best.pt"

    per_class_f1 = _eval_per_class_f1(best_pt, split_root, active_behaviors, imgsz)

    metrics = {
        "experiment_id"   : exp_name,
        "timestamp"       : timestamp,
        "phase"           : "behavior_classification",
        "model_version"   : model_name.replace(".pt", ""),
        "loss_mode"       : loss_mode,
        "focal_gamma"     : focal_gamma if "focal" in loss_mode else None,
        "behaviors"       : active_behaviors,
        "dataset"         : {"crops_dir": str(crops_dir), "class_counts": class_counts},
        "hyperparameters" : {k: str(v) for k, v in train_args.items()},
        "final_metrics"   : {
            "top1_accuracy": round(float(rd.get("metrics/accuracy_top1", 0)), 4),
            "top5_accuracy": round(float(rd.get("metrics/accuracy_top5", 0)), 4),
        },
        "per_class_f1"    : per_class_f1,
        "duration_minutes": round((t_end - t_start) / 60, 1),
        "best_model_path" : str(best_pt),
        "notes"           : f"v4: loss={loss_mode} gamma={focal_gamma}",
    }

    log_dir  = LOGS_DIR / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "metrics.json"
    log_path.write_text(json.dumps(metrics, indent=2))

    print(f"\n{'='*60}")
    print(f"  ✅ TRAINING COMPLETE — {exp_name}")
    print(f"  Loss mode     : {loss_mode}")
    print(f"  Duration      : {metrics['duration_minutes']} min")
    print(f"  Top-1 Acc     : {metrics['final_metrics']['top1_accuracy']:.4f}")
    print(f"  Top-5 Acc     : {metrics['final_metrics']['top5_accuracy']:.4f}")
    print(f"  Best model    : {metrics['best_model_path']}")
    print(f"  Log saved     : {log_path}")
    print(f"{'='*60}\n")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Loss mode comparison (run all three, compare per-class F1):
  --loss weighted_ce      inverse-frequency class weights in CE loss
  --loss focal            focal loss, no class weights (gamma=2 default)
  --loss focal_weighted   focal loss + class weights  ← recommended first try

Model options (all use the same script, no code changes needed):
  --model yolov8s-cls.pt   default
  --model yolov8m-cls.pt   bigger / more accurate
  --model yolo11s-cls.pt   YOLO11 small  (pip install ultralytics>=8.3.50)
  --model yolo11m-cls.pt   YOLO11 medium
        """,
    )
    parser.add_argument("--model",       default="yolov8s-cls.pt")
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--batch",       type=int,   default=64)
    parser.add_argument("--imgsz",       type=int,   default=288)
    parser.add_argument("--patience",    type=int,   default=25)
    parser.add_argument(
        "--loss",
        default="focal_weighted",
        choices=LOSS_MODES,
        help="Loss function. See epilog for details.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma. Ignored for ce/weighted_ce. Default=2.0.",
    )
    args = parser.parse_args()

    train_classifier(
        model_name   = args.model,
        epochs       = args.epochs,
        batch        = args.batch,
        imgsz        = args.imgsz,
        patience     = args.patience,
        loss_mode    = args.loss,
        focal_gamma  = args.focal_gamma,
    )

if __name__ == "__main__":
    main()