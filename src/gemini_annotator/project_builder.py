"""Turn a Gemini EpisodeAnnotation into clips inside an Annotator project.

Annotator's project schema (store.py, format "segment-annotations" v2) is:

    project["styles"] = [{"id", "name", "color"}, ...]
    project["layers"] = [{"id", "style_id", "clips": [{"id", "start", "end", "text"}]}]

Clip times are seconds on Annotator's GLOBAL time axis (the whole recording,
each view's files concatenated in order) - the same axis project["episodes"]
uses for global_start/global_end, so an episode's Gemini chunk times (which
are local to that episode's trimmed clip) just need the episode's
global_start added.

We write into four styles:
  main     - one clip per episode, the overall task (reuses Annotator's
             built-in "main" style if the project already has one).
  subtask  - one clip per chunk.
  atomic   - one clip per chunk, text = its atomic actions joined with "; ".
  recovery - one clip per chunk where is_recovery is true (reuses
             Annotator's built-in "recovery" style - it ships as a default
             style precisely for this).

Re-running annotation for an episode is idempotent: existing clips in these
four layers that start inside the episode's [global_start, global_end) are
replaced, not duplicated, so nothing has to be done by hand between runs.
Layers/clips outside that range (other episodes, human edits) are untouched.
"""

from __future__ import annotations

import uuid
from typing import Any

from .schema import EpisodeAnnotation

STYLE_MAIN = "main"
STYLE_SUBTASK = "subtask"
STYLE_ATOMIC = "atomic"
STYLE_RECOVERY = "recovery"

# Matches store.py's DEFAULT_COLORS; new styles we introduce pick the next
# unused color from the same palette so they look native in the timeline.
COLOR_PALETTE = [
    "#6EA8FF", "#7ED9A7", "#E2A0FF", "#F2B45C",
    "#7FD6E8", "#FF9BA8", "#B9CE6A", "#C3A1F0",
]

# Default style_map for POST /api/export/lerobot: our timeline style name ->
# LeRobot's canonical persistent style. "subtask" and "motion" are canonical;
# "recovery" is not, so lerobot_io registers it as a custom EXTENDED_STYLE
# automatically. "main" (the restated overall task) maps onto "task_aug".
DEFAULT_LEROBOT_STYLE_MAP = {
    STYLE_MAIN: "task_aug",
    STYLE_SUBTASK: "subtask",
    STYLE_ATOMIC: "motion",
    STYLE_RECOVERY: "recovery",
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _ensure_style(project: dict[str, Any], name: str) -> str:
    for style in project.setdefault("styles", []):
        if style["name"] == name:
            return style["id"]
    used_colors = {s["color"] for s in project["styles"]}
    color = next((c for c in COLOR_PALETTE if c not in used_colors), COLOR_PALETTE[0])
    style_id = _new_id("st")
    project["styles"].append({"id": style_id, "name": name, "color": color})
    return style_id


def _ensure_layer(project: dict[str, Any], style_id: str) -> dict[str, Any]:
    for layer in project.setdefault("layers", []):
        if layer["style_id"] == style_id:
            return layer
    layer = {"id": _new_id("ly"), "style_id": style_id, "clips": []}
    project["layers"].append(layer)
    return layer


def _replace_range(layer: dict[str, Any], start: float, end: float, new_clips: list[dict[str, Any]]) -> None:
    kept = [c for c in layer["clips"] if not (start <= float(c["start"]) < end)]
    layer["clips"] = kept + new_clips


def apply_episode_annotation(
    project: dict[str, Any],
    annotation: EpisodeAnnotation,
    *,
    episode_global_start: float,
    episode_global_end: float,
) -> dict[str, int]:
    """Mutate `project` in place. Returns a summary of clips written per layer."""
    main_id = _ensure_style(project, STYLE_MAIN)
    subtask_id = _ensure_style(project, STYLE_SUBTASK)
    atomic_id = _ensure_style(project, STYLE_ATOMIC)
    recovery_id = _ensure_style(project, STYLE_RECOVERY)

    main_layer = _ensure_layer(project, main_id)
    subtask_layer = _ensure_layer(project, subtask_id)
    atomic_layer = _ensure_layer(project, atomic_id)
    recovery_layer = _ensure_layer(project, recovery_id)

    _replace_range(main_layer, episode_global_start, episode_global_end, [{
        "id": _new_id("cl"),
        "start": round(episode_global_start, 3),
        "end": round(episode_global_end, 3),
        "text": annotation.overall_task,
    }])

    subtask_clips, atomic_clips, recovery_clips = [], [], []
    for chunk in annotation.chunks:
        g_start = round(episode_global_start + chunk.start_sec, 3)
        g_end = round(episode_global_start + chunk.end_sec, 3)

        subtask_clips.append({"id": _new_id("cl"), "start": g_start, "end": g_end, "text": chunk.subtask})

        if chunk.atomic_actions:
            atomic_clips.append({
                "id": _new_id("cl"), "start": g_start, "end": g_end,
                "text": "; ".join(chunk.atomic_actions),
            })

        if chunk.is_recovery:
            text = chunk.recovery_action or chunk.recovery_from or chunk.subtask
            if chunk.recovery_from and chunk.recovery_action:
                text = f"{chunk.recovery_action} (recovers from: {chunk.recovery_from})"
            recovery_clips.append({"id": _new_id("cl"), "start": g_start, "end": g_end, "text": text})

    _replace_range(subtask_layer, episode_global_start, episode_global_end, subtask_clips)
    _replace_range(atomic_layer, episode_global_start, episode_global_end, atomic_clips)
    _replace_range(recovery_layer, episode_global_start, episode_global_end, recovery_clips)

    return {
        "main": 1,
        "subtask": len(subtask_clips),
        "atomic": len(atomic_clips),
        "recovery": len(recovery_clips),
    }
