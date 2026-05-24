"""
pipeline.py
===========
End-to-end inference pipeline:
    Video → Detect (YOLOv8) → Track (ByteTrack) → Classify (MobileNetV3) → JSON + annotated video

Usage:
    python inference/pipeline.py \\
        --video path/to/video.mp4 \\
        --detector path/to/best.pt \\
        --classifier path/to/best_classifier.pth \\
        --output_video path/to/output.mp4
"""

import cv2
import torch
import json
import numpy as np
import supervision as sv
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
from PIL import Image
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking.bytetrack_wrapper import CattleTracker          # type: ignore
from train.train_classifier import (                          # type: ignore
    BehaviorClassifier, BEHAVIORS, INFERENCE_TRANSFORM
)

BEHAVIOR_COLORS = {
    "standing":   (  0, 220,   0),
    "walking":    (  0, 220, 220),
    "eating":     (255, 165,   0),
    "drinking":   (255,  50,  50),
    "lying_down": (180,   0, 180),
    "sleeping":   ( 50,  50, 255),
    "rumination": (255, 180, 200),
    "unknown":    (128, 128, 128),
}

INFER_SIZE = (1280, 720)


class CattlePipeline:
    """
    Full Detect → Track → Classify pipeline.

    Parameters
    ----------
    detector_path   : Path to YOLOv8 .pt weights
    classifier_path : Path to MobileNetV3 .pth state dict
    device          : 'cuda' or 'cpu'
    conf_thresh     : Detection confidence threshold
    """

    def __init__(
        self,
        detector_path: str,
        classifier_path: str,
        device: str = "cuda",
        conf_thresh: float = 0.35,
    ):
        self.device      = device
        self.conf_thresh = conf_thresh

        # ── Detector ────────────────────────────────────────────────────────
        self.detector = YOLO(detector_path)
        print(f"✅ Detector loaded: {detector_path}")

        # ── Classifier ──────────────────────────────────────────────────────
        self.classifier = BehaviorClassifier().to(device)
        state = torch.load(classifier_path, map_location=device)
        self.classifier.load_state_dict(state)
        self.classifier.eval()
        print(f"✅ Classifier loaded: {classifier_path}")

        # ── Tracker ─────────────────────────────────────────────────────────
        self.tracker = CattleTracker(
            track_thresh=conf_thresh,
            track_buffer=30,
        )

    # ── Behavior classification ──────────────────────────────────────────────
    def classify_from_buffer(self, crop_buffer: list) -> tuple[str, float]:
        """Average class probabilities over the crop buffer (temporal fusion)."""
        if not crop_buffer:
            return "unknown", 0.0

        probs_list = []
        with torch.no_grad():
            for crop_bgr in crop_buffer:
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                tensor   = INFERENCE_TRANSFORM(
                    Image.fromarray(crop_rgb)
                ).unsqueeze(0).to(self.device)
                logits = self.classifier(tensor)
                probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                probs_list.append(probs)

        avg_probs  = np.mean(probs_list, axis=0)
        pred_idx   = int(np.argmax(avg_probs))
        confidence = float(avg_probs[pred_idx])
        return BEHAVIORS[pred_idx], confidence

    # ── Video processing ─────────────────────────────────────────────────────
    def process_video(
        self,
        video_path: str,
        output_path: str = None,
        save_json: bool = True,
        classify_every: int = 4,    # run classifier every N frames (speed/accuracy trade-off)
    ) -> dict:
        """
        Process a video end-to-end.

        Parameters
        ----------
        video_path    : Input .mp4 path
        output_path   : Output annotated .mp4 path (optional)
        save_json     : Save per-frame JSON results alongside the input video
        classify_every: Run behavior classifier every N frames (1 = every frame)
        """
        self.tracker.reset()

        cap          = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, INFER_SIZE)

        results = {
            "video_path"   : str(video_path),
            "processed_at" : datetime.now().isoformat(),
            "fps"          : fps,
            "resolution"   : f"{INFER_SIZE[0]}x{INFER_SIZE[1]}",
            "total_frames" : total_frames,
            "frames"       : {},
        }

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_inf = cv2.resize(frame, INFER_SIZE)

            # Stage 1: Detect
            yolo_res   = self.detector(frame_inf, conf=self.conf_thresh, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(yolo_res)

            # Stage 2: Track
            tracked = self.tracker.update(detections, frame_inf)

            # Stage 3: Classify (throttled for speed)
            frame_results = []
            if tracked.tracker_id is not None:
                for i, track_id in enumerate(tracked.tracker_id):
                    # Classify or re-use last prediction
                    if frame_idx % classify_every == 0:
                        buf      = self.tracker.get_crop_buffer(track_id)
                        beh, conf = self.classify_from_buffer(buf)
                        self.tracker.set_behavior(track_id, beh, conf)
                    else:
                        state = self.tracker.track_states.get(int(track_id))
                        beh   = state.last_behavior  if state else "unknown"
                        conf  = state.last_confidence if state else 0.0

                    box = tracked.xyxy[i].tolist()
                    frame_results.append({
                        "track_id"            : int(track_id),
                        "bbox_xyxy"           : [round(v, 1) for v in box],
                        "behavior"            : beh,
                        "behavior_confidence" : round(conf, 4),
                        "detection_confidence": round(float(tracked.confidence[i]), 4),
                    })

                    # Annotate
                    if writer:
                        color       = BEHAVIOR_COLORS.get(beh, BEHAVIOR_COLORS["unknown"])
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame_inf, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            frame_inf,
                            f"ID:{track_id} {beh} {conf:.2f}",
                            (x1, max(y1 - 8, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                        )

            results["frames"][frame_idx] = frame_results

            if writer:
                writer.write(frame_inf)

            if frame_idx % 100 == 0:
                print(f"  Frame {frame_idx}/{total_frames}", end="\r")

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        # Save JSON
        if save_json:
            json_path = str(video_path).replace(".mp4", "_results.json")
            Path(json_path).write_text(json.dumps(results, indent=2))
            print(f"\n💾 Results saved: {json_path}")

        print(f"\n✅ Processed {frame_idx} frames from {video_path}")
        return results


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True, help="Input mp4 path")
    parser.add_argument("--detector",   required=True, help="YOLO best.pt path")
    parser.add_argument("--classifier", required=True, help="MobileNetV3 best_classifier.pth")
    parser.add_argument("--output_video", default=None, help="Output annotated mp4")
    parser.add_argument("--conf",       type=float, default=0.35)
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--classify_every", type=int, default=4)
    args = parser.parse_args()

    pipeline = CattlePipeline(
        detector_path  = args.detector,
        classifier_path= args.classifier,
        device         = args.device,
        conf_thresh    = args.conf,
    )
    pipeline.process_video(
        video_path   = args.video,
        output_path  = args.output_video,
        classify_every = args.classify_every,
    )


if __name__ == "__main__":
    main()
