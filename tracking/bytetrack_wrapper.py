"""
bytetrack_wrapper.py
====================
ByteTrack wrapper for calf multi-object tracking.
Maintains a rolling crop buffer per track_id for the behavior classifier.
"""

import numpy as np
import supervision as sv
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import cv2

BEHAVIOR_BUFFER_SIZE = 8   # frames of temporal history fed to classifier


@dataclass
class TrackState:
    track_id: int
    crop_buffer: deque = field(
        default_factory=lambda: deque(maxlen=BEHAVIOR_BUFFER_SIZE)
    )
    last_behavior: str = "unknown"
    last_confidence: float = 0.0
    frame_count: int = 0


class CattleTracker:
    """
    Wraps supervision.ByteTrack.

    Parameters
    ----------
    track_thresh : float
        Min detection confidence to initiate a new track.
    track_buffer : int
        Frames to keep a lost track alive (set high for occlusion-heavy scenes).
    match_thresh : float
        IoU threshold for track-detection association.
    frame_rate : int
        Source video frame rate.
    """

    def __init__(
        self,
        track_thresh: float = 0.35,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 25,
    ):
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate,
        )
        self.track_states: dict[int, TrackState] = {}

    def update(self, detections: sv.Detections, frame: np.ndarray) -> sv.Detections:
        """
        Update tracker with new detections.

        Parameters
        ----------
        detections : sv.Detections
            Raw YOLO detections for this frame.
        frame : np.ndarray
            BGR frame at inference resolution (e.g. 1280×720).

        Returns
        -------
        sv.Detections
            Detections with tracker_id assigned.
        """
        tracked = self.tracker.update_with_detections(detections)

        if tracked.tracker_id is None:
            return tracked

        h, w = frame.shape[:2]

        for i, track_id in enumerate(tracked.tracker_id):
            if track_id not in self.track_states:
                self.track_states[track_id] = TrackState(track_id=int(track_id))

            state = self.track_states[int(track_id)]
            state.frame_count += 1

            # Extract crop and add to rolling buffer
            x1, y1, x2, y2 = map(int, tracked.xyxy[i])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                state.crop_buffer.append(cv2.resize(crop, (224, 224)))

        return tracked

    def get_crop_buffer(self, track_id: int) -> Optional[list[np.ndarray]]:
        """Return the crop history list for a given track_id."""
        state = self.track_states.get(int(track_id))
        if state and len(state.crop_buffer) > 0:
            return list(state.crop_buffer)
        return None

    def set_behavior(self, track_id: int, behavior: str, confidence: float):
        """Store the latest behavior prediction for a track."""
        state = self.track_states.get(int(track_id))
        if state:
            state.last_behavior = behavior
            state.last_confidence = confidence

    def reset(self):
        """Reset tracker state (call between videos)."""
        self.tracker.reset()
        self.track_states.clear()
