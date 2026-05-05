"""Cattle Tracker - Streamlit app powered by YOLO + ByteTrack."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from lib import detector, reid, store, video as videolib

UPLOAD_DIR = Path("data/uploads")
PROCESSED_DIR = Path("data/processed")
MAX_UPLOAD_MB = 200
RETENTION_SECONDS = 60 * 60 * 6
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="Cattle Tracker",
    page_icon="cow",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    defaults = {
        "uploaded_path": None,
        "processed_path": None,
        "raw_processed_path": None,
        "tracks": None,
        "track_meta": None,
        "track_decisions": {},
        "processing": False,
        "video_meta": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


def cleanup_runtime_files() -> None:
    """Keep Railway's ephemeral disk from filling with stale videos."""
    cutoff = time.time() - RETENTION_SECONDS
    for folder in (UPLOAD_DIR, PROCESSED_DIR):
        for path in folder.glob("*"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass


def reset_video_state() -> None:
    st.session_state.uploaded_path = None
    st.session_state.processed_path = None
    st.session_state.raw_processed_path = None
    st.session_state.tracks = None
    st.session_state.track_meta = None
    st.session_state.track_decisions = {}


def save_uploaded_video(file) -> Path | None:
    size = getattr(file, "size", None)
    if size is not None and size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Upload is larger than {MAX_UPLOAD_MB} MB.")
        return None

    suffix = Path(file.name).suffix.lower()
    if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
        st.error("Unsupported video type.")
        return None

    upload_path = UPLOAD_DIR / f"upload_{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    upload_path.write_bytes(file.read())
    return upload_path


def render_sidebar() -> None:
    st.sidebar.title("Your Herd")
    cattle = store.list_cattle()
    st.sidebar.caption(f"{len(cattle)} head on record")

    if not cattle:
        st.sidebar.info(
            "No cattle yet. Upload a video, name the animals you see, "
            "and they'll show up here."
        )
        return

    for cow in sorted(cattle, key=lambda c: c.get("lastSeen", 0), reverse=True):
        with st.sidebar.expander(cow["name"], expanded=False):
            snaps = cow.get("snapshots", [])
            if snaps:
                cols = st.columns(min(3, len(snaps)))
                for i, snapshot in enumerate(snaps[-3:]):
                    if Path(snapshot).exists():
                        cols[i % len(cols)].image(snapshot, use_container_width=True)

            last_seen = cow.get("lastSeen", 0)
            if last_seen:
                ago = max(0, int(time.time()) - last_seen)
                if ago < 60:
                    when = "just now"
                elif ago < 3600:
                    when = f"{ago // 60}m ago"
                elif ago < 86400:
                    when = f"{ago // 3600}h ago"
                else:
                    when = f"{ago // 86400}d ago"
                st.caption(f"Last seen {when}")

            new_name = st.text_input(
                "Rename", value=cow["name"], key=f"rename_{cow['id']}"
            )
            c1, c2 = st.columns(2)
            if c1.button("Save", key=f"savename_{cow['id']}", use_container_width=True):
                if new_name and new_name != cow["name"]:
                    store.rename_cow(cow["id"], new_name)
                    st.rerun()
            if c2.button(
                "Delete",
                key=f"del_{cow['id']}",
                type="secondary",
                use_container_width=True,
            ):
                store.delete_cow(cow["id"])
                st.rerun()


def render_uploader() -> None:
    cleanup_runtime_files()
    st.subheader("Upload a video of your cattle")
    st.caption(
        "We'll run YOLO + ByteTrack, draw a box around each animal, "
        "and try to recognize anyone you've named before."
    )

    file = st.file_uploader(
        "Drop a video (mp4, mov, mkv, webm, avi)",
        type=["mp4", "mov", "mkv", "webm", "avi"],
        accept_multiple_files=False,
    )

    families = []
    seen = set()
    for model in detector.MODEL_CATALOG:
        if model["family"] not in seen:
            families.append(model["family"])
            seen.add(model["family"])

    fcol, scol = st.columns([1, 1])
    with fcol:
        family = st.selectbox(
            "YOLO family",
            families,
            index=families.index("YOLOv8"),
            help="v8 is the safest baseline. Newer families may be faster or more accurate.",
        )

    family_models = [m for m in detector.MODEL_CATALOG if m["family"] == family]
    size_labels = {
        "n": "Nano",
        "t": "Tiny",
        "s": "Small",
        "m": "Medium",
        "b": "Big",
        "c": "Compact",
        "l": "Large",
        "e": "Extra",
        "x": "X-Large",
    }
    options = [
        f"{m['size'].upper()} - {size_labels.get(m['size'], m['size'])} "
        f"(~{m['mb']} MB) - {m['desc']}"
        for m in family_models
    ]
    default_size_idx = next(
        (i for i, model in enumerate(family_models) if model["size"] == "n"), 0
    )
    with scol:
        size_choice = st.selectbox(
            "Model size",
            options,
            index=default_size_idx,
            help="Larger is more accurate, but slower and heavier on Railway CPU.",
        )
    chosen = family_models[options.index(size_choice)]

    cached = (Path("data/models") / chosen["id"]).exists()
    if cached:
        st.caption(f"Using **{chosen['id']}** - already downloaded.")
    else:
        st.caption(
            f"Using **{chosen['id']}** - ~{chosen['mb']} MB will be downloaded "
            "on first use, then cached."
        )

    profile_col, imgsz_col, stride_col = st.columns([1, 1, 1])
    with profile_col:
        profile = st.selectbox(
            "Processing profile",
            ["Fast CPU", "Balanced", "Small-object recall"],
            index=0,
            help="Fast CPU is best for Railway. Small-object recall is slower.",
        )

    profile_defaults = {
        "Fast CPU": {"imgsz": 640, "stride": 2, "iou": 0.45},
        "Balanced": {"imgsz": 800, "stride": 1, "iou": 0.50},
        "Small-object recall": {"imgsz": 960, "stride": 1, "iou": 0.55},
    }
    defaults = profile_defaults[profile]

    with imgsz_col:
        imgsz_options = [640, 800, 960, 1280]
        imgsz = st.selectbox(
            "Input size",
            imgsz_options,
            index=imgsz_options.index(defaults["imgsz"]),
            help="Higher sizes improve distant cattle recall but cost more CPU.",
        )
    with stride_col:
        stride_options = [1, 2, 3, 4]
        frame_stride = st.selectbox(
            "Frame stride",
            stride_options,
            index=stride_options.index(defaults["stride"]),
            help="2 means process every second frame.",
        )

    limit_col, conf_col, iou_col = st.columns([1, 1, 1])
    with limit_col:
        max_seconds = st.number_input(
            "Process at most (seconds)",
            min_value=2,
            max_value=60,
            value=15,
            step=1,
            help="Keep this low on shared CPU deployments.",
        )
    with conf_col:
        confidence = st.slider(
            "Detection confidence",
            min_value=0.10,
            max_value=0.80,
            value=0.35,
            step=0.05,
        )
    with iou_col:
        iou = st.slider(
            "NMS IoU",
            min_value=0.25,
            max_value=0.75,
            value=float(defaults["iou"]),
            step=0.05,
            help="Lower removes duplicate boxes; higher keeps overlapping cattle.",
        )

    if file is not None and st.button(
        "Process video", type="primary", use_container_width=True
    ):
        upload_path = save_uploaded_video(file)
        if upload_path is None:
            return
        st.session_state.uploaded_path = str(upload_path)

        process_video(
            str(upload_path),
            float(confidence),
            float(iou),
            int(imgsz),
            int(frame_stride),
            int(max_seconds),
            chosen["id"],
        )


def process_video(
    src_path: str,
    conf: float,
    iou: float,
    imgsz: int,
    frame_stride: int,
    max_seconds: int,
    weights: str,
) -> None:
    raw_out = PROCESSED_DIR / f"raw_{int(time.time())}.mp4"
    final_out = PROCESSED_DIR / f"out_{int(time.time())}.mp4"

    progress_bar = st.progress(0, text=f"Warming up {weights}...")
    status = st.empty()

    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        st.error("Could not open uploaded video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    requested_frames = max(1, int(fps * max_seconds))
    max_frames = min(total, requested_frames) if total > 0 else requested_frames
    processed_total = max(1, int(np.ceil(max_frames / max(1, frame_stride))))

    tracks_acc: dict[int, list[dict]] = {}
    best_crops: dict[int, dict] = {}

    try:
        for evt in detector.track_video(
            src_path,
            str(raw_out),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            frame_stride=frame_stride,
            max_frames=max_frames,
            weights=weights,
        ):
            if evt["type"] == "meta":
                processed_total = evt.get("processed_total", processed_total)
                status.info(
                    f"Video: {evt['width']}x{evt['height']} @ {evt['fps']:.1f} fps. "
                    f"Processing {processed_total} inference frames at imgsz={imgsz}."
                )
            elif evt["type"] == "progress":
                pct = min(1.0, evt["frame"] / max(1, processed_total))
                progress_bar.progress(
                    pct, text=f"Tracking frame {evt['frame']} / {processed_total}"
                )
            elif evt["type"] == "done":
                tracks_acc = evt["tracks"]
                best_crops = evt.get("best_crops", {})
    except Exception as exc:
        st.error(f"Tracking failed: {exc}")
        return

    progress_bar.progress(1.0, text="Preparing browser video...")
    if videolib.has_ffmpeg() and videolib.transcode_to_h264(str(raw_out), str(final_out)):
        st.session_state.processed_path = str(final_out)
        try:
            raw_out.unlink()
        except OSError:
            pass
    else:
        st.session_state.processed_path = str(raw_out)
    st.session_state.raw_processed_path = str(raw_out)

    track_meta: dict[int, dict] = {}
    for tid, observations in tracks_acc.items():
        if len(observations) < 4:
            continue
        best = max(observations, key=lambda obs: obs["conf"])
        crop_info = best_crops.get(tid)
        crop = crop_info["crop"] if crop_info is not None else None
        if crop is None:
            crop = detector.crop_from_video(src_path, best["frame"], best["bbox"])
        if crop is None:
            continue
        fp = reid.fingerprint(crop)
        track_meta[tid] = {
            "crop": crop,
            "fingerprint": fp,
            "best_obs": best,
            "n_obs": len(observations),
        }

    st.session_state.tracks = tracks_acc
    st.session_state.track_meta = track_meta
    st.session_state.track_decisions = {}

    herd = store.list_cattle()
    auto_matched = 0
    for tid, meta in track_meta.items():
        match = reid.best_match(meta["fingerprint"], herd, threshold=0.55)
        if match is not None:
            st.session_state.track_decisions[tid] = {
                "cow_id": match["cow"]["id"],
                "skipped": False,
                "auto": True,
                "score": match["score"],
            }
            auto_matched += 1

    progress_bar.empty()
    status.success(
        f"Tracked {len(track_meta)} cattle in this clip. "
        f"Auto-recognized {auto_matched} from your herd."
    )
    st.rerun()


def render_results() -> None:
    proc = st.session_state.processed_path
    track_meta = st.session_state.track_meta or {}

    if not proc:
        return

    st.subheader("Tracked video")
    if Path(proc).exists():
        st.video(proc)
        if proc.endswith(".mp4") and proc == st.session_state.raw_processed_path:
            st.caption(
                "Note: ffmpeg was not available, so the video uses mp4v codec. "
                "It may not preview in all browsers - download to view."
            )
            with open(proc, "rb") as video_file:
                st.download_button(
                    "Download annotated video",
                    video_file.read(),
                    file_name="cattle_tracked.mp4",
                    mime="video/mp4",
                )
    else:
        st.warning("Processed video file is missing.")

    if not track_meta:
        st.info(
            "No cattle were tracked in this clip. Try lowering confidence, "
            "using a larger input size, or selecting a different video."
        )
        return

    st.subheader(f"Cattle in this clip - {len(track_meta)} tracked")
    st.caption(
        "For each animal, confirm who they are. Recognized matches are pre-filled "
        "with an AI suggestion you can override."
    )

    herd = store.list_cattle()
    name_to_cow = {cow["name"]: cow for cow in herd}

    for tid in sorted(track_meta.keys()):
        meta = track_meta[tid]
        decision = st.session_state.track_decisions.get(tid, {})
        render_track_card(tid, meta, decision, herd, name_to_cow)

    st.divider()
    if st.button("Save all to herd", type="primary", use_container_width=True):
        save_session()


def render_track_card(
    tid: int,
    meta: dict,
    decision: dict,
    herd: list[dict],
    name_to_cow: dict[str, dict],
) -> None:
    with st.container(border=True):
        cols = st.columns([1, 3])
        with cols[0]:
            crop_rgb = cv2.cvtColor(meta["crop"], cv2.COLOR_BGR2RGB)
            st.image(crop_rgb, caption=f"Track #{tid}", use_container_width=True)
            st.caption(
                f"Seen in {meta['n_obs']} frames - "
                f"best conf {meta['best_obs']['conf']:.2f}"
            )

        with cols[1]:
            saved_cow_id = decision.get("cow_id")
            saved_cow = next((cow for cow in herd if cow["id"] == saved_cow_id), None)
            if decision.get("auto") and saved_cow is not None:
                st.success(
                    f"AI suggestion: **{saved_cow['name']}** "
                    f"(match score {decision.get('score', 0):.2f})"
                )

            skip_label = "- Skip for now -"
            new_label = "+ Add as a NEW cow"
            options = [skip_label, new_label] + [cow["name"] for cow in herd]
            default_index = 0
            if decision.get("skipped"):
                default_index = 0
            elif saved_cow is not None:
                try:
                    default_index = 2 + [cow["id"] for cow in herd].index(saved_cow["id"])
                except ValueError:
                    default_index = 0

            choice = st.selectbox(
                "Who is this?",
                options,
                index=default_index,
                key=f"choice_{tid}",
            )

            new_name_value = ""
            if choice == new_label:
                new_name_value = st.text_input(
                    "New name",
                    key=f"newname_{tid}",
                    placeholder="e.g. Bessie",
                )

            c1, c2 = st.columns([1, 1])
            if c1.button("Confirm", key=f"confirm_{tid}", use_container_width=True):
                if choice == skip_label:
                    st.session_state.track_decisions[tid] = {
                        "cow_id": None,
                        "skipped": True,
                        "auto": False,
                    }
                elif choice == new_label:
                    if not new_name_value.strip():
                        st.warning("Please enter a name first.")
                    else:
                        cow = store.add_cow(
                            new_name_value.strip(),
                            meta["crop"],
                            meta["fingerprint"],
                        )
                        st.session_state.track_decisions[tid] = {
                            "cow_id": cow["id"],
                            "skipped": False,
                            "auto": False,
                            "created": True,
                        }
                        st.success(f"Added {cow['name']} to the herd.")
                        st.rerun()
                else:
                    cow = name_to_cow.get(choice)
                    if cow is not None:
                        st.session_state.track_decisions[tid] = {
                            "cow_id": cow["id"],
                            "skipped": False,
                            "auto": False,
                        }
                        st.success(f"This will be saved as {cow['name']}.")

            if c2.button("Skip", key=f"skip_{tid}", use_container_width=True):
                st.session_state.track_decisions[tid] = {
                    "cow_id": None,
                    "skipped": True,
                    "auto": False,
                }
                st.rerun()


def save_session() -> None:
    """Apply all confirmed/auto decisions: append snapshots and bump lastSeen."""
    track_meta = st.session_state.track_meta or {}
    saved = 0
    for tid, decision in st.session_state.track_decisions.items():
        cow_id = decision.get("cow_id")
        if not cow_id or decision.get("skipped") or decision.get("created"):
            continue

        cow = store.get_cow(cow_id)
        if cow is None:
            continue

        meta = track_meta.get(tid)
        if meta is None:
            continue

        store.add_observation(cow_id, meta["crop"], meta["fingerprint"])
        saved += 1

    st.success(f"Saved {saved} sightings to the herd.")
    reset_video_state()
    st.rerun()


def main() -> None:
    render_sidebar()

    st.title("Cattle Tracker")
    st.caption(
        "Upload a video, let YOLO find every animal, and tag who's who. "
        "Cattle you've named will be auto-recognized in future videos."
    )

    if st.session_state.processed_path is None:
        render_uploader()
    else:
        c1, _ = st.columns([1, 1])
        if c1.button("Process another video", use_container_width=True):
            reset_video_state()
            st.rerun()
        render_results()


if __name__ == "__main__":
    main()
