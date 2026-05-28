# 🐄 Cattle Analytics — Experiment Suite

> **Pipeline**: YOLOv8 Detection → ByteTrack Tracking → MobileNetV3 Behavior Classification  
> **Platform**: Google Colab Pro+ (A100 GPU)  
> **Data format**: Version folders (`v1/`, `v2/`, …), each with one `.mp4` + `annotations.xml`  
> **Behaviors**: `standing`, `walking`, `eating`, `drinking`, `lying_down`, `rumination`, `sleeping`  
> **Current dataset**: 34 annotated videos (`v1`–`v34`)

---

## 📁 Folder Structure

```
experiment/
├── scripts/
│   ├── setup_drive.py            ← Drive mount + all path constants (imported by all scripts)
│   ├── 01_extract_frames.py      ← Extract annotated frames from each version folder [incremental]
│   ├── 02_cvat_to_yolo.py        ← CVAT XML → YOLO detection labels + dataset.yaml [incremental]
│   ├── 03_generate_crops.py      ← Generate 224×224 behavior crops for classifier [incremental]
│   ├── 04_check_dataset_stats.py ← Dataset health check (run before every training)
│   └── 05_visualize_annotations.py ← Draw bounding boxes on frames (sanity check)
│
├── train/
│   ├── train_detector.py         ← YOLOv8 calf detector training
│   ├── train_classifier.py       ← MobileNetV3 behavior classifier training
│   └── configs/
│       └── detector.yaml         ← Annotated YOLO config (reference; overrides in script)
│
├── tracking/
│   └── bytetrack_wrapper.py      ← ByteTrack + rolling crop buffer per track_id
│
├── inference/
│   └── pipeline.py               ← End-to-end: Detect → Track → Classify → JSON + video
│
├── evaluation/
│   └── eval_pipeline.py          ← Confusion matrix, per-class report, experiment table
│
├── notebooks/
│   └── cattle_analytics_experiment.ipynb  ← Main Colab notebook (run this)
│
└── cookbook/
    └── CATTLE_ANALYTICS_COOKBOOK.md       ← Architecture decisions + full reference
```

---

## 🚀 Quick Start — Google Colab Pro+

### Step 1 — Set up Google Drive

Create this folder layout **manually once** in your Google Drive:

```
MyDrive/
└── cattle-analytics/
    ├── data/
    │   ├── v1/
    │   │   ├── <video_name>.mp4       ← your recorded video
    │   │   └── annotations.xml        ← CVAT XML 1.1 export
    │   ├── v2/
    │   │   ├── <video_name>.mp4
    │   │   └── annotations.xml
    │   └── v3/ ... vN/               ← add more versions as you annotate
    ├── processed/                     ← auto-created by pipeline
    ├── models/checkpoints/            ← weights saved here (persists across sessions)
    └── logs/experiments/              ← JSON metrics per experiment run
```

> **Important**: Each version folder must have **exactly one `.mp4`** and **one `annotations.xml`**.  
> The XML must be a CVAT XML 1.1 export in interpolation mode.

### Step 2 — Clone the Code to Drive

In Colab:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/YOUR_ORG/cattle-analytics.git \
    /content/drive/MyDrive/cattle-analytics
```

### Step 3 — Open the Notebook

Open `notebooks/cattle_analytics_experiment.ipynb` in Colab and run cells **top to bottom**.

Always set: **Runtime → Change runtime type → A100 GPU**

---

## 📋 Full Workflow (Cell-by-Cell)

| Cell | What it does | Time estimate |
|------|-------------|---------------|
| 1 | Install dependencies | 2 min |
| 2 | Mount Drive + pull code | 1 min |
| 3 | Setup paths + discover version folders | <1 min |
| 4 | Extract annotated frames *(incremental — skips already-done)* | 2–5 min for new videos only |
| 5 | CVAT XML → YOLO labels + `dataset.yaml` *(incremental)* | <1 min for new videos only |
| 6 | Generate behavior crops *(incremental)* | 2–5 min for new videos only |
| 7 | **Dataset health check** (⚠️ always run) | <1 min |
| 8 | Visualize annotations (sanity check) | <1 min |
| 9 | Train YOLOv8 detector | 30–60 min |
| 10 | Train behavior classifier | 10–20 min |
| 11 | Evaluate + compare experiments | 5 min |
| 12 | Run inference on a new video | 5–15 min |
| 13 | Add more videos + retrain | iterative |

---

## 🔁 Incremental Processing (Adding New Videos)

Scripts `01`, `02`, and `03` are **incremental** — they remember which versions they've already processed using small sentinel files on Drive, and **skip them automatically** on the next run.

### How it works

| Script | Sentinel file | Stored info |
|--------|--------------|-------------|
| `01_extract_frames.py` | `processed/frames/vX/.done` | marker only |
| `02_cvat_to_yolo.py` | `processed/yolo_detection/.proc/vX.json` | split, frame count, behavior stats |
| `03_generate_crops.py` | `processed/behavior_crops/.done_vX.json` | per-behavior crop counts |

### Adding a new video (e.g. v35)

1. Upload `v35/` to Drive (with `.mp4` + `annotations.xml`)
2. Re-run the pipeline cells — **v1–v34 are skipped instantly**, only v35 is processed:

```bash
python scripts/01_extract_frames.py          # ⏭ v1–v34 skipped | ✅ v35 extracted
python scripts/02_cvat_to_yolo.py            # ⏭ v1–v34 cached  | ✅ v35 converted
python scripts/03_generate_crops.py          # ⏭ v1–v34 skipped | ✅ v35 cropped
```

> `dataset.yaml` and `dataset_stats.json` are always **fully rebuilt** from all versions (old + new) so they stay complete and accurate.

### Specifying the train/val split for new videos

New videos default to **train**. To designate a new video as val:

```bash
# Manual mode — fastest (recommended)
python scripts/02_cvat_to_yolo.py --split_mode manual --val_versions v7 v35

# Auto mode — picks val versions by frame count to hit ~20%
python scripts/02_cvat_to_yolo.py --split_mode auto --val_split 0.2
```

> **Note**: The split for already-processed versions is **locked in** their proc record and never changed by subsequent runs. Only new versions get assigned a split.

### Force re-processing a version

Use `--force` if you re-annotated a video and need to regenerate its output:

```bash
# Re-process only v7 (e.g. you fixed annotations)
python scripts/01_extract_frames.py --force v7
python scripts/02_cvat_to_yolo.py   --force v7
python scripts/03_generate_crops.py --force v7

# Re-process everything from scratch (nuclear option)
python scripts/01_extract_frames.py --force
python scripts/02_cvat_to_yolo.py   --force
python scripts/03_generate_crops.py --force
```

---

## 📊 Experiment Tracking

Every training run logs its metrics to:
```
logs/experiments/<exp_name>/metrics.json
```

To compare all runs:
```bash
python evaluation/eval_pipeline.py --phase summary
```

Example output:
```
| Experiment                   | Phase    | Model       | mAP50  | mAP50-95 | Prec   | Recall/Acc | Time  |
|------------------------------|----------|-------------|--------|----------|--------|------------|-------|
| detector_20260524_143021     | detect   | yolov8s     | 0.7812 | 0.5930   | 0.8740 | 0.8210     | 42.3m |
| classifier_20260524_160022   | classify | MobileNetV3 | —      | —        | —      | acc=0.7340 | 15.1m |
```

---

## ⚠️ Acceptance Criteria

### Detector (before proceeding to classifier)
| Metric | Target (20+ videos) | Acceptable (first 10) |
|--------|-------------------|----------------------|
| mAP@0.5 | > 0.85 | > 0.70 |
| mAP@0.5:0.95 | > 0.65 | > 0.50 |
| Precision | > 0.85 | > 0.75 |
| Recall | > 0.90 | > 0.80 |

### Classifier
| Metric | Target | Minimum |
|--------|--------|---------| 
| Overall Val Accuracy | > 0.80 | > 0.65 |
| Per-class Recall | > 0.75 | > 0.60 |
| sleep vs lying_down confusion | < 30% | < 50% |
| eating vs drinking confusion | < 20% | < 40% |

---

## 🐛 Known Issues

| Issue | Impact | Fix |
|-------|--------|-----|
| 62.5% boxes occluded | Tracker loses IDs | `lost_track_buffer=30` (already set) |
| Only 3/7 behaviors in early videos | Classifier can't train | Annotate eating/lying/rumination/sleeping |
| Window backlight | Silhouette frames | `hsv_v=0.5` augmentation (set in config) |
| sleep vs lying_down | ~30% confusion | Use LSTM temporal classifier (next iteration) |
| `num_workers > 0` on Colab | DataLoader deadlock on Drive | Fixed: `num_workers=0` in `train_classifier.py` |

---

## 🔧 Running Scripts Directly (Without Notebook)

```bash
# From experiment/ directory:

# Step 0: health check — see what's already processed
python scripts/04_check_dataset_stats.py

# Step 1: extract frames (skips already-done versions)
python scripts/01_extract_frames.py

# Step 2: convert annotations (manual split — fast path)
python scripts/02_cvat_to_yolo.py --split_mode manual --val_versions v7

# Step 3: generate crops (skips already-done versions)
python scripts/03_generate_crops.py

# Step 4: train
python train/train_detector.py --epochs 100 --batch 8
python train/train_classifier.py --epochs 50

# Step 5: evaluate
python evaluation/eval_pipeline.py --phase all

# Step 6: inference
python inference/pipeline.py \
    --video data/v1/video.mp4 \
    --detector models/checkpoints/detector_<ts>/weights/best.pt \
    --classifier models/checkpoints/classifier_<ts>/best_classifier.pth \
    --output_video data/v1/output.mp4
```

---

## 📚 Reference

- Full architecture decisions → `cookbook/CATTLE_ANALYTICS_COOKBOOK.md`
- Annotation guidelines → `cookbook/CATTLE_ANALYTICS_COOKBOOK.md §12`
- Iteration checklist → `cookbook/CATTLE_ANALYTICS_COOKBOOK.md §13`

---

*Maintained by Aritra & Nimai · Updated: May 2026 · Version: Phase-1 (34 videos)*
