# 🐄 Cattle Analytics — Experiment Suite

> **Pipeline**: YOLOv8 Detection → ByteTrack Tracking → MobileNetV3 Behavior Classification  
> **Platform**: Google Colab Pro+ (A100 GPU)  
> **Data format**: Version folders (`v1/`, `v2/`, …), each with one `.mp4` + `annotations.xml`  
> **Behaviors**: `standing`, `walking`, `eating`, `drinking`, `lying_down`, `rumination`, `sleeping`

---

## 📁 Folder Structure

```
experiment/
├── scripts/
│   ├── 00_setup_drive.py         ← Drive mount + all path constants (run first)
│   ├── 01_extract_frames.py      ← Extract annotated frames from each version folder
│   ├── 02_cvat_to_yolo.py        ← CVAT XML → YOLO detection labels + dataset.yaml
│   ├── 03_generate_crops.py      ← Generate 224×224 behavior crops for classifier
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
    │   ├── v3/ ... v10/               ← first 10 videos for Phase 1
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
| 4 | Extract annotated frames from all `v1..v10` | 2–5 min |
| 5 | CVAT XML → YOLO labels + `dataset.yaml` | <1 min |
| 6 | Generate behavior crops for classifier | 2–5 min |
| 7 | **Dataset health check** (⚠️ always run) | <1 min |
| 8 | Visualize annotations (sanity check) | <1 min |
| 9 | Train YOLOv8 detector | 30–60 min |
| 10 | Train behavior classifier | 10–20 min |
| 11 | Evaluate + compare experiments | 5 min |
| 12 | Run inference on a new video | 5–15 min |
| 13 | Add more videos + retrain | iterative |

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

## 📈 Scaling Beyond 10 Videos

When ready to add more videos:

1. Upload `v11/`, `v12/`, etc. to Drive (same structure: mp4 + xml)
2. Re-run Cells 3–11 — they **auto-discover** new version folders
3. Cell 11 will **automatically compare** new experiment results against all previous runs

> **Note**: Video-level train/val split is re-randomized each time (seed=42).  
> This means adding more videos gradually expands both splits proportionally.

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

---

## 🔧 Running Scripts Directly (Without Notebook)

```bash
# From experiment/ directory:

# Step 1: check what Drive has
python scripts/04_check_dataset_stats.py

# Step 2: extract frames
python scripts/01_extract_frames.py --max_versions 10

# Step 3: convert annotations
python scripts/02_cvat_to_yolo.py --val_split 0.2

# Step 4: generate crops
python scripts/03_generate_crops.py

# Step 5: train
python train/train_detector.py --epochs 30 --batch 8
python train/train_classifier.py --epochs 50

# Step 6: evaluate
python evaluation/eval_pipeline.py --phase all

# Step 7: inference
python inference/pipeline.py \
    --video data/v1/video.mp4 \
    --detector models/checkpoints/detector_<ts>/weights/best.pt \
    --classifier models/checkpoints/classifier_<ts>/best_classifier.pth \
    --output_video data/v1/output.mp4
```

---

## 📚 Reference

- Full architecture decisions: Aritra 
- Annotation guidelines: Nill
- Iteration checklist: Nill

---

*Maintained by Aritra & Nimai  · Updated: May 2026 · Version: Phase-1 (10 videos)*
