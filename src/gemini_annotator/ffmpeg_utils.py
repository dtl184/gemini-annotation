"""Local video probing/trimming. Only ever touches temp files this pipeline
downloaded itself via the Annotator API - never the dataset in place."""

from __future__ import annotations

import subprocess
import tempfile
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


def concat_videos(paths: list[Path], dest: Path) -> Path:
    """Losslessly stitch several mp4s (in order) into one, via ffmpeg's concat
    demuxer (`-c copy` - no re-encode). Only valid when the inputs share
    codec/parameters, which LeRobot v3's per-view file splitting guarantees:
    a view's files are the same recording cut by size, not by re-encoding.
    """
    if not paths:
        raise FfmpegError("concat_videos called with no inputs")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(paths) == 1:
        import shutil
        shutil.copy2(paths[0], dest)
        return dest

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in paths:
            f.write(f"file '{p.resolve()}'\n")
        list_path = Path(f.name)
    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(dest)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            raise FfmpegError(f"ffmpeg failed concatenating {len(paths)} files: {result.stderr[-2000:]}")
    finally:
        list_path.unlink(missing_ok=True)
    return dest
