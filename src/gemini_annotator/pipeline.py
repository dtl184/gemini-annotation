"""Orchestrates one `annotate` run: session -> per-episode clip -> Gemini ->
merge into the project -> save back through the API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .annotator_client import AnnotatorClient
from .config import Config
from .episode_source import (
    Episode,
    list_episodes,
    pick_view,
    resolve_episode_clip,
    resolve_whole_video,
)
from .ffmpeg_utils import trim_clip
from .gemini_client import annotate_video, annotate_window
from .project_builder import (
    DEFAULT_LEROBOT_STYLE_MAP,
    apply_episode_annotation,
    apply_whole_video_annotation,
)
from .schema import DetectedEpisode, EpisodeAnnotation

log = logging.getLogger("gemini_annotator")

DEFAULT_WINDOW_SECONDS = 100.0
# If a window's last detected episode ends within this many seconds of the
# window's own edge, treat its end as unreliable (possibly window-truncated)
# rather than a real boundary - see run_annotate_whole_video.
WINDOW_EDGE_SLACK_S = 2.0
MAX_WINDOWS = 1000  # guards against a runaway loop, not a real-world limit


@dataclass
class EpisodeResult:
    episode: Episode
    annotation: EpisodeAnnotation
    clip_counts: dict[str, int]


def run_annotate(
    config: Config,
    root: str,
    *,
    view: str | None = None,
    episode_indices: list[int] | None = None,
    known_task: str | None = None,
    known_objects: list[str] | None = None,
    cache_dir: Path,
    save: bool = True,
    export_lerobot: bool = False,
    export_dry_run: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> list[EpisodeResult]:
    notify = on_progress or (lambda msg: None)
    client = AnnotatorClient(config.annotator_url)

    notify(f"Fetching session for {root} from {config.annotator_url} ...")
    session = client.get_session(root, scope="dataset")
    timeline = session["timeline"]
    project = session["project"]

    view_key = pick_view(timeline, view)
    notify(f"Using view {view_key!r}")

    episodes = list_episodes(timeline, view_key)
    if episode_indices is not None:
        wanted = set(episode_indices)
        episodes = [e for e in episodes if e.episode_index in wanted]
        missing = wanted - {e.episode_index for e in episodes}
        if missing:
            raise ValueError(f"Episode indices not found in this dataset: {sorted(missing)}")

    results: list[EpisodeResult] = []
    for episode in episodes:
        notify(
            f"Episode {episode.episode_index}: "
            f"[{episode.global_start:.1f}s - {episode.global_end:.1f}s]"
        )
        clip_path = resolve_episode_clip(client, root, timeline, episode, view_key, cache_dir)
        duration = episode.global_end - episode.global_start

        task_hint = known_task or (episode.tasks[0] if episode.tasks else None)
        notify(f"  -> uploading {clip_path.name} to Gemini ({config.gemini_model}) ...")
        annotation = annotate_video(
            config, clip_path, duration=duration, known_task=task_hint, known_objects=known_objects,
        )
        notify(
            f"  -> overall_task={annotation.overall_task!r}, "
            f"{len(annotation.chunks)} chunk(s), "
            f"{sum(c.is_recovery for c in annotation.chunks)} recovery"
        )

        counts = apply_episode_annotation(
            project, annotation,
            episode_global_start=episode.global_start,
            episode_global_end=episode.global_end,
        )
        results.append(EpisodeResult(episode=episode, annotation=annotation, clip_counts=counts))

    if save and results:
        notify("Saving project back to Annotator ...")
        client.save_project(project)

    if export_lerobot and results:
        notify(f"Exporting to LeRobot language_persistent (dry_run={export_dry_run}) ...")
        report = client.export_lerobot(
            root, style_map=DEFAULT_LEROBOT_STYLE_MAP, dry_run=export_dry_run,
        )
        notify(f"  -> {report}")

    return results


@dataclass
class WholeVideoResult:
    episodes: list[DetectedEpisode]
    clip_counts: dict[str, int]
    video_duration: float


def run_annotate_whole_video(
    config: Config,
    root: str,
    *,
    view: str | None = None,
    known_objects: list[str] | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    cache_dir: Path,
    save: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> WholeVideoResult:
    """Annotate a whole (possibly many-episode, stitched) video in one pass,
    with Gemini detecting episode boundaries itself rather than trusting the
    dataset's own episode metadata (or working at all when there isn't any -
    e.g. a raw scaffolded video).

    Gemini's boundary detection degrades sharply with how much video it has
    to hold in view at once - accurate to a fraction of a second over a ~100s/
    3-4-episode window, unreliable (even hallucinating timestamps past the
    clip's real length) once a window holds 5+ episodes. So the video is
    walked in fixed-size windows; when a window's last detected episode ends
    flush with the window's own edge, its boundary is untrusted (may be an
    artifact of the window cutting it off, not a real reset) and it is
    re-processed as part of the next window instead of committed.
    """
    notify = on_progress or (lambda msg: None)
    client = AnnotatorClient(config.annotator_url)

    notify(f"Fetching session for {root} from {config.annotator_url} ...")
    session = client.get_session(root, scope="dataset")
    timeline = session["timeline"]
    project = session["project"]

    view_key = pick_view(timeline, view)
    notify(f"Using view {view_key!r}")

    notify("Resolving the whole video (downloading/stitching source files) ...")
    video_path, duration = resolve_whole_video(client, root, timeline, view_key, cache_dir)
    notify(f"  -> {duration:.1f}s total")

    all_episodes: list[DetectedEpisode] = []
    cursor = 0.0
    windows_dir = cache_dir / "windows"
    for i in range(MAX_WINDOWS):
        if cursor >= duration - 0.5:
            break
        window_end = duration if duration - cursor <= window_seconds * 1.3 else cursor + window_seconds
        notify(f"Window {i}: [{cursor:.1f}s - {window_end:.1f}s] ...")

        window_clip = windows_dir / f"w{i:04d}_{cursor:.0f}-{window_end:.0f}.mp4"
        trim_clip(video_path, cursor, window_end, window_clip)
        episodes = annotate_window(
            config, window_clip, window_start=cursor, window_end=window_end, known_objects=known_objects,
        )

        committed = episodes
        next_cursor = window_end
        if window_end < duration - 1e-6 and episodes:
            last = episodes[-1]
            if last.end_sec >= window_end - WINDOW_EDGE_SLACK_S and last.start_sec > cursor + 1.0:
                committed = episodes[:-1]
                next_cursor = last.start_sec  # re-process this one with a fresh window ahead of it

        notify(f"  -> {len(committed)} episode(s) committed" + ("" if committed == episodes else " (1 deferred to next window)"))
        all_episodes.extend(committed)
        cursor = next_cursor
    else:
        raise RuntimeError(f"Whole-video annotation did not finish within {MAX_WINDOWS} windows - likely a bug.")

    all_episodes.sort(key=lambda e: e.start_sec)
    for i, ep in enumerate(all_episodes):
        ep.index = i

    counts = apply_whole_video_annotation(project, all_episodes, video_start=0.0, video_end=duration)
    notify(
        f"Detected {counts['episodes']} episode(s), {counts['subtask']} subtask chunk(s), "
        f"{counts['recovery']} recovery chunk(s)."
    )

    if save:
        notify("Saving project back to Annotator ...")
        client.save_project(project)

    return WholeVideoResult(episodes=all_episodes, clip_counts=counts, video_duration=duration)
