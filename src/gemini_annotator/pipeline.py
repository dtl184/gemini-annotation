"""Orchestrates one `annotate` run: session -> per-episode clip -> Gemini ->
merge into the project -> save back through the API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .annotator_client import AnnotatorClient
from .config import Config
from .episode_source import Episode, list_episodes, pick_view, resolve_episode_clip
from .gemini_client import annotate_video
from .project_builder import DEFAULT_LEROBOT_STYLE_MAP, apply_episode_annotation
from .schema import EpisodeAnnotation

log = logging.getLogger("gemini_annotator")


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
        annotation = annotate_video(config, clip_path, duration=duration, known_task=task_hint)
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
