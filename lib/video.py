"""Helpers for rewriting tracker output to a browser-playable codec."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def transcode_to_h264(src: str, dst: str) -> bool:
    """Re-encode an mp4v video to H.264 so browsers can play it.
    Returns True on success.
    """
    if not has_ffmpeg():
        return False
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", src,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-an",
                dst,
            ],
            capture_output=True,
            timeout=600,
        )
        return result.returncode == 0 and Path(dst).exists()
    except Exception:
        return False
