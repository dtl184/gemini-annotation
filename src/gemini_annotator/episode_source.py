"""Resolve an episode (as reported by GET /api/session) to a local, trimmed
mp4 clip, going through the Annotator API for every byte - we never read the
dataset directory directly. Source files are cached locally so annotating
several episodes that share one mp4 (common - LeRobot v3 files hold many
episodes) only downloads it once.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .annotator_client import AnnotatorClient
from .ffmpeg_utils import concat_videos, probe_duration, trim_clip


class EpisodeSourceError(RuntimeError):
    pass


@dataclass
class Episode:
    episode_index: int
    global_start: float
    global_end: float
    tasks: list[str] | None = None


def pick_view(timeline: dict[str, Any], requested: str | None) -> str:
    view_keys = list(timeline.get("view_keys") or timeline.get("views", {}).keys())
    if not view_keys:
        raise EpisodeSourceError("This dataset has no video views (nothing under videos/).")
    if requested:
        if requested not in view_keys:
            raise EpisodeSourceError(f"View {requested!r} not found. Available views: {view_keys}")
        return requested
    return view_keys[0]


def list_episodes(timeline: dict[str, Any], view_key: str) -> list[Episode]:
    """Real episodes if the dataset has meta/episodes/; otherwise (e.g. a
    scaffolded standalone video) one synthetic episode covering the whole
    view's duration, so the rest of the pipeline doesn't need two code paths.
    """
    raw = timeline.get("episodes") or []
    if raw:
        return [
            Episode(
                episode_index=e["episode_index"],
                global_start=float(e["global_start"]),
                global_end=float(e["global_end"]) if e.get("global_end") is not None else float(timeline["duration"]),
                tasks=e.get("tasks"),
            )
            for e in raw
        ]

    total = timeline.get("views", {}).get(view_key, {}).get("total")
    if not total:
        raise EpisodeSourceError("No episode metadata and no video duration to fall back on.")
    return [Episode(episode_index=0, global_start=0.0, global_end=float(total))]


def _cache_path(cache_dir: Path, root: str, rel_path: str) -> Path:
    digest = hashlib.sha1(f"{root}::{rel_path}".encode()).hexdigest()[:12]
    return cache_dir / "sources" / f"{digest}-{Path(rel_path).name}"


def resolve_episode_clip(
    client: AnnotatorClient,
    root: str,
    timeline: dict[str, Any],
    episode: Episode,
    view_key: str,
    cache_dir: Path,
) -> Path:
    """Download (if needed) the source file covering `episode`, trim it to the
    episode's span, and return the path to the trimmed local clip.
    """
    segments = timeline.get("views", {}).get(view_key, {}).get("segments", [])
    segment = next(
        (s for s in segments if s["start"] <= episode.global_start < s["end"]),
        None,
    )
    if segment is None:
        raise EpisodeSourceError(
            f"Episode {episode.episode_index} start ({episode.global_start:.2f}s) doesn't fall "
            f"inside any {view_key!r} video segment."
        )
    if episode.global_end > segment["end"] + 1e-3:
        raise EpisodeSourceError(
            f"Episode {episode.episode_index} spans past the end of {segment['path']} "
            f"({episode.global_end:.2f}s > {segment['end']:.2f}s) - it straddles two files, "
            "which this pipeline doesn't stitch across. Try a different --view."
        )

    src = _cache_path(cache_dir, root, segment["path"])
    if not src.is_file():
        client.download_media(root, segment["path"], src)

    local_start = episode.global_start - segment["start"]
    local_end = episode.global_end - segment["start"]
    dest = cache_dir / "clips" / f"{src.stem}_ep{episode.episode_index:04d}.mp4"
    if dest.is_file():
        dest.unlink()
    trim_clip(src, local_start, local_end, dest)
    return dest


def clip_duration(path: Path) -> float:
    return probe_duration(path)


def resolve_whole_video(
    client: AnnotatorClient,
    root: str,
    timeline: dict[str, Any],
    view_key: str,
    cache_dir: Path,
) -> tuple[Path, float]:
    """Download every source file for `view_key` (cached) and, if there's more
    than one, stitch them into a single local mp4 spanning the view's whole
    global timeline - so windowing (see pipeline.run_annotate_whole_video)
    never has to reason about file boundaries, only about time.
    """
    segments = timeline.get("views", {}).get(view_key, {}).get("segments", [])
    if not segments:
        raise EpisodeSourceError(f"No video segments for view {view_key!r}.")

    local_files = []
    for seg in segments:
        src = _cache_path(cache_dir, root, seg["path"])
        if not src.is_file():
            client.download_media(root, seg["path"], src)
        local_files.append(src)

    digest = hashlib.sha1(f"{root}::{view_key}::{[s['path'] for s in segments]}".encode()).hexdigest()[:12]
    dest = cache_dir / "wholevideo" / f"{digest}.mp4"
    if not dest.is_file():
        concat_videos(local_files, dest)

    total = timeline.get("views", {}).get(view_key, {}).get("total")
    return dest, float(total)
