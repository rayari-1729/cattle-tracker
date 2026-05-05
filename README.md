# Cattle Tracker

Streamlit app that detects and tracks cattle in uploaded videos using YOLO (v8 → v26) + ByteTrack, lets you name each animal, and auto-recognizes them in future videos via color-histogram matching.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Railway

This repo includes a Railway-ready `Dockerfile` and `railway.json`.

1. Create a new Railway project from this folder/repo.
2. Railway will build with the Dockerfile and run:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

3. For persistent herd data, mount a Railway volume at `/app/data` or replace
   `lib/store.py` with a database/object-storage backend. Without that, uploads,
   processed videos, model weights, herd JSON, and snapshots are container-local.

Recommended Railway settings:
- Start with `yolov8n.pt` or another nano model on CPU.
- Use the `Fast CPU` profile for shared CPU plans.
- Use `Small-object recall` only when cattle are distant and the instance has enough CPU budget.

## Deploy to Streamlit Community Cloud

1. Push this folder (`artifacts/cattle-tracker/`) to a public GitHub repo. The simplest path is to push the whole monorepo.
2. Go to <https://share.streamlit.io> and click **New app**.
3. Configure:
   - **Repository**: your GitHub repo
   - **Branch**: `main` (or whatever you push to)
   - **Main file path**: `artifacts/cattle-tracker/app.py`
   - **Python version**: 3.11 (matches `runtime.txt`)
4. Click **Deploy**.

Streamlit Cloud will read:
- `requirements.txt` — Python deps (already pinned)
- `packages.txt` — apt packages (`ffmpeg`, `libgl1`, `libglib2.0-0`)
- `runtime.txt` — Python version
- `.streamlit/config.toml` — server/theme config

### Notes on Streamlit Cloud limits

- **1 GB RAM** per app on the free tier. Stick to nano/small model sizes (`yolov8n`, `yolo11n`, `yolo12n`, `yolo26n`) and keep `Process at most (seconds)` under ~20s for safety. Medium/large models may OOM.
- **Ephemeral disk**: `data/herd.json` and snapshots will reset when the container restarts. For a persistent herd on Streamlit Cloud, swap `lib/store.py` to use S3, GCS, or a hosted DB.
- **Upload limit**: capped to 200 MB on Cloud (matches `maxUploadSize` in `.streamlit/config.toml`).
- **First run downloads the model weights** (~5–135 MB depending on size) and may take a minute. They cache in `data/models/` until the container restarts.

## Files

| Path | Purpose |
| ---- | ------- |
| `app.py` | Streamlit UI |
| `lib/detector.py` | YOLO + ByteTrack tracking, model catalog |
| `lib/reid.py` | HSV color-histogram fingerprint + matching |
| `lib/store.py` | JSON-backed herd + snapshots |
| `lib/video.py` | ffmpeg H.264 transcode |
| `requirements.txt` | Python deps |
| `packages.txt` | Streamlit Cloud apt deps |
| `runtime.txt` | Python version pin |
| `.streamlit/config.toml` | Streamlit server config |
