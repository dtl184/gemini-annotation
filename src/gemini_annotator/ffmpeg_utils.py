"""Local video probing/trimming. Only ever touches temp files this pipeline
downloaded itself via the Annotator API - never the dataset in place."""

from __future__ import annotations

import subprocess
from pathlib import Path


class FfmpegError(RuntimeError):
    pass


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return round(float(out.stdout.strip()), 3)
    except ValueError as err:
        raise FfmpegError(f"ffprobe could not read duration of {path}: {out.stderr.strip()}") from err


def trim_clip(src: Path, start: float, end: float, dest: Path, *, keep_audio: bool = False) -> Path:
    """Cut [start, end) out of `src` into `dest`, frame-accurate.

    -ss before -i is both fast (input-side seek) and, when transcoding rather
    than stream-copying, still decodes from the exact requested time - the
    keyframe-snapping imprecision only applies to `-c copy`. Frame accuracy
    matters here: an off-by-a-few-seconds trim would shift every Gemini
    timestamp in this clip away from true episode-relative time, and those
    timestamps get added straight to the episode's global_start.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)
    if duration <= 0:
        raise FfmpegError(f"Non-positive clip duration: start={start} end={end}")

    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
    ]
    cmd += ["-c:a", "aac"] if keep_audio else ["-an"]
    cmd += [str(dest)]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise FfmpegError(f"ffmpeg failed trimming {src} [{start}, {end}]: {result.stderr[-2000:]}")
    return dest
