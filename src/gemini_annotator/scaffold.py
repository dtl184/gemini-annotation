"""Wrap a standalone mp4 (not part of any LeRobot dataset) in the minimal
directory shape Annotator's scanner (dataset.py) recognizes, so it becomes
openable in the GUI - and reachable through the exact same live-API
annotation path as a real dataset - without inventing a second Annotator
integration.

Annotator only requires `meta/info.json` to exist (`dataset.is_dataset`) and
videos to sit under `videos/<view>/chunk-NNN/file-NNN.mp4`
(`dataset.scan_video_files`); everything else (episode metadata, robot
fields) is optional and build_timeline tolerates its absence. So the scaffold
is deliberately tiny: one video, one view, no episode metadata - the pipeline
then treats the whole file as a single synthetic episode (see
episode_source.list_episodes).

What this does NOT produce is a trainable LeRobot dataset (no
data/chunk-*/file-*.parquet with per-frame state/action). `--export lerobot`
only makes sense for real LeRobot datasets; scaffolded ones support the GUI
and JSON/CSV export only.
"""

from __future__ import annotations

import json
import os
import subprocess
from fractions import Fraction
from pathlib import Path

from .ffmpeg_utils import probe_duration

DEFAULT_VIEW_NAME = "observation.images.main"


class ScaffoldError(RuntimeError):
    pass


def _probe_fps(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return float(Fraction(out.stdout.strip()))
    except (ValueError, ZeroDivisionError) as err:
        raise ScaffoldError(f"ffprobe could not read the frame rate of {path}: {out.stderr.strip()}") from err


def scaffold_dataset(
    video_path: Path,
    out_dir: Path,
    *,
    view_name: str = DEFAULT_VIEW_NAME,
    fps: float | None = None,
) -> Path:
    if not video_path.is_file():
        raise ScaffoldError(f"No such video file: {video_path}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ScaffoldError(f"{out_dir} already exists and is not empty")

    video_dir = out_dir / "videos" / view_name / "chunk-000"
    video_dir.mkdir(parents=True, exist_ok=True)
    dest_video = video_dir / "file-000.mp4"
    try:
        os.link(video_path, dest_video)  # same filesystem: instant, no extra disk use
    except OSError:
        import shutil
        shutil.copy2(video_path, dest_video)

    resolved_fps = fps or _probe_fps(dest_video)
    duration = probe_duration(dest_video)

    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "codebase_version": "v3.0",
        "robot_type": "unknown",
        "fps": resolved_fps,
        "total_episodes": 1,
        "total_frames": round(duration * resolved_fps),
        "video_files_size_in_mb": 200,
        "chunks_size": 1000,
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    return out_dir
