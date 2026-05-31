# 🐄 Cattle Analytics — Project Update Log

> **Project**: Calf Behavior Analytics  
> **Camera**: Reolink TrackMix WiFi-1 — indoor pen, top-down/angled view, 4K @ 25 fps  
> **Annotation Tool**: CVAT (XML 1.1, interpolation mode)  
> **Training Platform**: Google Colab Pro+ (A100 GPU)  
> **Maintained by**: Aritra & Nimai

---

## 🏗️ What This Project Is

Automated cattle behavior monitoring using computer vision. The camera records calves in an indoor pen 24/7. The pipeline automatically detects each calf, tracks it across frames, and classifies its behavior — without any manual observation.

### The 7 Behaviors Being Tracked
`standing` · `walking` · `lying_down` · `sleeping` · `eating` · `drinking` · `rumination`

---

## 🧠 Core Architecture — The Main Idea

**Why not one model?**  
A single YOLO model with 7 behavior classes was considered and rejected because:
- Behaviors are **temporal** — rumination vs standing looks identical in a single frame
- A single head mixes detection gradients with behavior gradients → poor box regression
- Track identity is lost across frames

**Chosen: Detect → Track → Classify (3-stage pipeline)**

```
Video Frame
    │
    ▼
[Stage 1]  YOLOv8s — Calf Detector
               outputs: bounding boxes + confidence
    │
    ▼
[Stage 2]  ByteTrack — Multi-Object Tracker
               outputs: track_id per box, temporal trajectory buffer
    │
    ▼
[Stage 3]  MobileNetV3-Small — Behavior Classifier
               input:  last N cropped frames per track_id
               output: behavior label + confidence
    │
    ▼
Final Output: { track_id, bbox, behavior, confidence, timestamp }
```

| Component | Choice | Reason |
|-----------|--------|--------|
| Detector | YOLOv8s | Fast, ONNX-exportable, handles occluded boxes |
| Tracker | ByteTrack | Best for crowded/occluded scenes (62% occlusion rate in this dataset) |
| Classifier | MobileNetV3-Small | Lightweight, runs on crops, fast enough for real-time |

---

## 📁 Project Structure

```
experiment/
├── scripts/
│   ├── setup_drive.py            ← Drive mount + path constants (shared by all scripts)
│   ├── 01_extract_frames.py      ← Extract annotated frames from each video version
│   ├── 02_cvat_to_yolo.py        ← CVAT XML → YOLO detection labels + dataset.yaml
│   ├── 03_generate_crops.py      ← Generate 224×224 behavior crops for classifier
│   ├── 04_check_dataset_stats.py ← Dataset health check (run before every training)
│   └── 05_visualize_annotations.py
│
├── train/
│   ├── train_detector.py         ← YOLOv8 training (A100 optimized)
│   └── train_classifier.py       ← MobileNetV3 training (A100 optimized)
│
├── tracking/
│   └── bytetrack_wrapper.py      ← ByteTrack + rolling crop buffer per track_id
│
├── inference/
│   └── pipeline.py               ← End-to-end: Detect → Track → Classify → JSON + video
│
├── evaluation/
│   └── eval_pipeline.py          ← Confusion matrix, per-class metrics, experiment table
│
├── notebooks/
│   └── cattle_analytics_experiment.ipynb  ← Main Colab notebook
│
└── cookbook/
    └── CATTLE_ANALYTICS_COOKBOOK.md       ← Full architecture reference
```

---

## 🗂️ Data Flow (Step by Step)

```
1. Record video (Reolink 4K)
        ↓
2. Annotate in CVAT → export as XML 1.1
        ↓
3. Upload vX/ folder to Google Drive (mp4 + annotations.xml)
        ↓
4. [Script 01] Extract annotated frames → processed/frames/vX/
        ↓
5. [Script 02] Convert XML → YOLO labels → processed/yolo_detection/
        ↓
6. [Script 03] Crop calves per behavior → processed/behavior_crops/<behavior>/
        ↓
7. [Script 04] Health check — verify frame counts, class balance, label integrity
        ↓
8. [Cell 8b]  Copy dataset to local NVMe (/content/) — 100× faster than Drive I/O
        ↓
9. Train YOLOv8s detector    → models/checkpoints/detector_<ts>/
        ↓
10. Train MobileNetV3 classifier → models/checkpoints/classifier_<ts>/
        ↓
11. Evaluate → confusion matrix, per-class report, experiment comparison table
        ↓
12. Inference on new video → annotated .mp4 + per-frame JSON
```

---

## 📅 Update Log

---

### ✅ v0.1 — Initial Pipeline Setup
**Date**: May 24, 2026  
**Status**: Foundation complete

#### What was built
- Full project scaffold: `scripts/`, `train/`, `tracking/`, `inference/`, `evaluation/`, `notebooks/`
- `setup_drive.py` — centralized Drive path management, version folder discovery
- `01_extract_frames.py` — frame extraction from annotated CVAT videos
- `02_cvat_to_yolo.py` — CVAT XML 1.1 → YOLO label conversion with video-level train/val split
- `03_generate_crops.py` — 224×224 behavior crop generation with 15% padding
- `04_check_dataset_stats.py` — dataset health report
- `train_detector.py` — YOLOv8s training with augmentation tuned for indoor top-down pen view
- `train_classifier.py` — MobileNetV3-Small with `WeightedRandomSampler` + class-weighted loss
- `bytetrack_wrapper.py` — ByteTrack integration with per-track crop buffer
- `pipeline.py` — end-to-end inference (Detect → Track → Classify → JSON + video)
- `eval_pipeline.py` — confusion matrix, per-class metrics, experiment comparison table
- `CATTLE_ANALYTICS_COOKBOOK.md` — full architecture reference (1,494 lines)

#### Key design decisions
- **Video-level train/val split** enforced to prevent temporal leakage
- **`standing`** used as the primary YOLO detection class (1 class detector)
- Augmentation disabled vertical flip (`flipud=0`) — camera orientation is fixed
- `hsv_v=0.5` (higher than default) to handle window backlight
- Dataset: 12 videos at launch (`v1`–`v12`)

---

### ✅ v0.2 — Dataset Cap Bug Fix + Detector Path Fix
**Date**: May 28, 2026  
**Reason**: Pipeline was silently skipping most videos; model weights not saving to Drive

#### Bugs fixed

**Bug 1 — `setup_drive.py`: `max_versions=10` hard cap**
- **Problem**: `setup_drive.py` had `max_versions=10` limit. With 34 videos loaded, only 10 were ever discovered. The other 24 were silently ignored on every run.
- **Fix**: Bumped to `max_versions=50` — covers current 34 videos + future growth.
- **Impact**: All 34 videos now discovered and processed correctly.

**Bug 2 — `train_detector.py`: wrong model output directory**
- **Problem**: `project` argument was hardcoded to a static string instead of `MODELS_DIR`. Model weights were saving to Colab's ephemeral `/content/` folder and being lost when the session ended.
- **Fix**: Changed to `project=str(MODELS_DIR)` which points to Google Drive.
- **Impact**: Model checkpoints now persist across sessions.

---

### ✅ v0.3 — Classifier Training Hang — 3 Bugs Fixed
**Date**: May 28, 2026  
**Reason**: Classifier was running but producing zero output; appeared to hang indefinitely

#### Bugs fixed

**Bug 1 (Critical) — `num_workers=2` deadlock on Colab**
- **Problem**: `DataLoader(num_workers=2)` — forked worker processes cannot access Google Drive's FUSE mount. Workers spin forever waiting for data. No error, no output, complete silence.
- **Fix**: `num_workers=0` — single-process loading. Works correctly with Drive.
- **Impact**: This was the primary cause of the training hang.

**Bug 2 — `shuffle=True` conflicting with `WeightedRandomSampler`**
- **Problem**: `DataLoader(shuffle=True, sampler=train_sampler)` — PyTorch does not allow both. The sampler is responsible for shuffling; having `shuffle=True` creates a silent conflict.
- **Fix**: Removed `shuffle=True` from train loader. Sampler handles ordering.

**Bug 3 — `WeightedRandomSampler` defined but never wired**
- **Problem**: `get_weighted_sampler()` method existed on the dataset class but was never called. The DataLoader just used `shuffle=True` with no sampler — all class imbalance protection was completely bypassed.
- **Fix**: Build `train_sample_weights` from `train_ds.indices` and pass `sampler=train_sampler` to the DataLoader.
- **Impact**: Rare behavior classes (`sleeping`, `drinking`, `rumination`) now get proportionally more training samples.

---

### ✅ v0.4 — Incremental Processing (Skip Already-Done Versions)
**Date**: May 28, 2026  
**Reason**: Every time a new video was added, all 34 videos had to be re-processed from scratch. This was slow and unnecessary.

#### What changed
All three data pipeline scripts (01, 02, 03) now use **sentinel files** to track which versions have already been processed.

| Script | Sentinel | Content |
|--------|----------|---------|
| `01_extract_frames.py` | `processed/frames/vX/.done` | `"ok"` |
| `02_cvat_to_yolo.py` | `processed/yolo_detection/.proc/vX.json` | split assignment, frame count, behavior stats |
| `03_generate_crops.py` | `processed/behavior_crops/.done_vX.json` | per-behavior crop counts |

**Key behaviour**:
- Adding `v35` → only `v35` is processed. `v1`–`v34` are skipped in ~0 seconds.
- `dataset_stats.json` is always rebuilt from **all** proc records (old + new) so it stays complete.
- Split assignments for already-processed versions are **frozen** — never changed on subsequent runs.
- `--force vX` flag on all three scripts to selectively reprocess a version (e.g. after fixing annotations).

---

### ✅ v0.5 — A100 GPU Performance Optimizations
**Date**: May 30, 2026  
**Reason**: GPU was not being fully utilized; training was slower than expected on A100

#### Detector (`train_detector.py`)

| Change | Before | After | Why |
|--------|--------|-------|-----|
| `device` arg | missing (CPU fallback risk) | `device=0` explicit | Never silently train on CPU |
| `amp=True` | missing | Added | bfloat16 on A100 — ~2× faster |
| `cache="ram"` | missing | Added | Frames in RAM, no Drive I/O during training |
| `workers=8` | missing | Added | Saturates Drive→RAM loader pipeline |
| `deterministic=True` | enabled | `False` | Was killing cuDNN auto-tuner → 30%+ slower |
| Batch size | 8 | **16** | A100 40GB handles it at imgsz=1280 |
| GPU fail-fast | missing | `_verify_gpu()` | Raises immediately if no GPU found |

#### Classifier (`train_classifier.py`)

| Change | Before | After | Why |
|--------|--------|-------|-----|
| AMP | missing | `GradScaler` + `autocast` | bfloat16 forward pass ~2× faster |
| `torch.compile` | missing | Added (graceful fallback) | 10–30% throughput via Triton kernels |
| `.to(device)` | blocking | `non_blocking=True` | Overlaps CPU→GPU transfer with compute |
| `zero_grad()` | memset-based | `set_to_none=True` | Less memory bandwidth |
| Batch size | 64 | **256** | MobileNetV3-Small is tiny; A100 fits 256 |
| GPU fail-fast | missing | `_verify_gpu()` | Raises immediately if no GPU |
| `cudnn.benchmark` | off | `True` | Auto-picks fastest conv kernels |

---

### ✅ v0.6 — TF32 + Local NVMe Dataset Cache
**Date**: May 30, 2026  
**Reason**: TF32 was disabled (PyTorch default); Drive I/O was the training bottleneck

#### Change 1 — TF32 enabled in both training scripts

```python
torch.backends.cuda.matmul.allow_tf32 = True   # ~3× faster matmul on A100
torch.backends.cudnn.allow_tf32       = True   # conv layers too
```

PyTorch disables TF32 by default for IEEE reproducibility. Enabling it is safe for training — the tiny precision difference (TF32 uses 10-bit mantissa vs float32's 23-bit) has no meaningful impact on model accuracy.

#### Change 2 — Local NVMe dataset copy (Notebook Cell 8b)

- **Problem**: Google Drive throttles after ~3,000 small file reads. With 27,000+ YOLO frame files and thousands of crop JPEGs, Drive I/O drops to ~3 seconds per image during DataLoader iteration. This was the primary training speed bottleneck.
- **Solution**: Copy datasets from Drive to `/content/` (local NVMe SSD) once per session before training. NVMe is 100× faster than Drive for random small-file reads.
- **Models and logs still save to Drive** — so nothing is lost if the session ends.

**Implementation**:
- Notebook **Cell 8b** (new) — copies `yolo_detection/` → `/content/local_yolo/` and `behavior_crops/` → `/content/local_crops/`
- **Critical**: After copying, `dataset.yaml` is patched to update the `path:` field from the Drive path to `/content/local_yolo`. Without this, YOLO still reads from Drive even when the files are local.
- Both training scripts **auto-detect** local copies at startup — no CLI flag needed.
- Cell is **idempotent** — safe to re-run; skips copy if already done this session.

---

## 🎯 Acceptance Criteria

### Detector
| Metric | Target (20+ videos) | Acceptable (first 10) |
|--------|--------------------|-----------------------|
| mAP@0.5 | > 0.85 | > 0.70 |
| mAP@0.5:0.95 | > 0.65 | > 0.50 |
| Precision | > 0.85 | > 0.75 |
| Recall | > 0.90 | > 0.80 |

### Classifier
| Metric | Target | Minimum |
|--------|--------|---------|
| Val Accuracy | > 0.80 | > 0.65 |
| Per-class Recall | > 0.75 | > 0.60 |
| sleep vs lying_down confusion | < 30% | < 50% |
| eating vs drinking confusion | < 20% | < 40% |

---

## ⚠️ Known Issues

| Issue | Impact | Status |
|-------|--------|--------|
| 62.5% boxes occluded | Tracker loses IDs | Mitigated via `lost_track_buffer=30` |
| Only 3/7 behaviors in early videos | Classifier can't train all classes | Annotate eating/lying/rumination/sleeping |
| Window backlight | Silhouette frames | Mitigated via `hsv_v=0.5` augmentation |
| sleep vs lying_down confusion | ~30% confusion expected | LSTM temporal classifier planned (v2) |
| `num_workers > 0` on Colab | DataLoader deadlock on Drive | **Fixed** v0.3 — `num_workers=0` |
| Drive I/O bottleneck | Slow training iteration | **Fixed** v0.6 — NVMe local copy |

---

## 🔮 Next Steps

- [ ] Video-level split for classifier (currently uses random crop-level split — temporal leakage risk)
- [ ] LSTM/temporal classifier to capture motion context for sleeping vs lying_down
- [ ] Expand dataset beyond 34 videos for detector mAP > 0.85
- [ ] ONNX export for edge deployment
- [ ] WandB integration for experiment tracking dashboard

---

*Last updated: May 30, 2026 · Phase 1 (34 videos annotated)*
