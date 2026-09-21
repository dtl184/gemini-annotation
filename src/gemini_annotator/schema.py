"""In-memory shape of a Gemini annotation result, independent of Annotator's
on-disk project schema (see project_builder.py for the conversion).

Two Gemini calls produce this, mirroring the steerable-policies pipeline's
"decompose, then restate" structure but skipping its grounded feature
extraction stage (Molmo/SAM2/DETR) - we hand Gemini the raw video instead:

  1. segmentation: watch the episode, name the overall task, and split it
     into contiguous (start_sec, end_sec, subtask) chunks.
  2. elaboration: given those chunk boundaries plus the video again, name the
     atomic actions inside each chunk and flag chunks whose action is a
     plausible *recovery* for a common VLA/manipulation failure - judged by
     what the action itself is, not by whether that failure is actually
     shown earlier in the video. A chunk showing "pick up the fork off the
     table" gets flagged as a recovery action (it's what you'd want a policy
     to do after dropping the fork) even in a clean demo where the fork was
     never dropped on screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# --------------------------------------------------------------------------
# stage 1: segmentation
# --------------------------------------------------------------------------

SEGMENTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_task": {
            "type": "string",
            "description": "One sentence describing the overall task accomplished in the video.",
        },
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "subtask": {
                        "type": "string",
                        "description": "Short imperative phrase, e.g. 'pick up the fork'.",
                    },
                },
                "required": ["start_sec", "end_sec", "subtask"],
            },
        },
    },
    "required": ["overall_task", "chunks"],
}

# --------------------------------------------------------------------------
# stage 2: elaboration (atomic actions + recovery labeling)
# --------------------------------------------------------------------------

ELABORATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "0-based index matching the input chunk list."},
                    "atomic_actions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Ordered short atomic motions within this chunk, e.g. "
                                       "['reach toward fork', 'close gripper', 'lift'].",
                    },
                    "is_recovery": {
                        "type": "boolean",
                        "description": "True if this chunk's action is the kind of thing that "
                                       "would serve as the right recovery for a common, plausible "
                                       "VLA/manipulation failure (dropped grasp, missed grasp, "
                                       "knocked-over object, overshoot, wrong placement, etc) - "
                                       "judged from the action itself, not from whether that "
                                       "failure is actually visible anywhere in this video. A "
                                       "clean, successful demo can still contain recovery-eligible "
                                       "chunks.",
                    },
                    "recovery_from": {
                        "type": ["string", "null"],
                        "description": "If is_recovery, the common failure mode this action would "
                                       "recover from, e.g. 'dropped the fork while lifting it'. "
                                       "Hypothetical - it does not need to appear in the video. "
                                       "Otherwise null.",
                    },
                    "recovery_action": {
                        "type": ["string", "null"],
                        "description": "If is_recovery, a short imperative phrase for the recovery "
                                       "behavior itself, suitable as a steering command, e.g. "
                                       "'pick the fork back up off the table'. Otherwise null.",
                    },
                },
                "required": ["index", "atomic_actions", "is_recovery"],
            },
        },
    },
    "required": ["chunks"],
}


# --------------------------------------------------------------------------
# whole-video mode: per-window episode-boundary + subtask detection
# --------------------------------------------------------------------------
#
# For "annotate the whole video" (many LeRobot episodes stitched back-to-back
# with no metadata trusted, or none available at all - e.g. a raw video),
# Gemini has to find the episode boundaries itself: an abrupt reset in arm/
# object position where one demo ends and the next begins. Empirically, this
# degrades fast with window length - accurate to a fraction of a second over
# ~90s/3 episodes, but badly wrong (even hallucinating timestamps past the
# clip's real length) over 150s/5 episodes. So the whole video is processed
# in short windows (see pipeline.run_annotate_whole_video for the sliding-
# window/carry-forward logic), and this schema handles ONE window: episode
# boundaries plus, in the same call, that episode's subtask chunks.

WINDOW_SEGMENTATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "episodes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sec": {"type": "number"},
                    "end_sec": {"type": "number"},
                    "overall_task": {
                        "type": "string",
                        "description": "One sentence describing what this episode accomplishes.",
                    },
                    "chunks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_sec": {"type": "number"},
                                "end_sec": {"type": "number"},
                                "subtask": {
                                    "type": "string",
                                    "description": "Short imperative phrase, e.g. 'pick up the fork'.",
                                },
                            },
                            "required": ["start_sec", "end_sec", "subtask"],
                        },
                    },
                },
                "required": ["start_sec", "end_sec", "overall_task", "chunks"],
            },
        },
    },
    "required": ["episodes"],
}


# --------------------------------------------------------------------------
# result types
# --------------------------------------------------------------------------

@dataclass
class SubtaskChunk:
    index: int
    start_sec: float
    end_sec: float
    subtask: str
    atomic_actions: list[str] = field(default_factory=list)
    is_recovery: bool = False
    recovery_from: str | None = None
    recovery_action: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "subtask": self.subtask,
            "atomic_actions": list(self.atomic_actions),
            "is_recovery": self.is_recovery,
            "recovery_from": self.recovery_from,
            "recovery_action": self.recovery_action,
        }


@dataclass
class EpisodeAnnotation:
    overall_task: str
    chunks: list[SubtaskChunk]

    def as_dict(self) -> dict[str, Any]:
        return {"overall_task": self.overall_task, "chunks": [c.as_dict() for c in self.chunks]}


@dataclass
class DetectedEpisode:
    index: int
    start_sec: float  # global (whole-video) seconds
    end_sec: float
    overall_task: str
    chunks: list[SubtaskChunk]


def parse_window_segmentation(
    data: dict[str, Any], *, window_start: float, window_end: float
) -> list[DetectedEpisode]:
    """Validate one window's output and shift its (window-local) times onto
    the whole video's global axis. Chunk indices are unique within this
    window (0..N-1 across all its episodes combined), matching what
    apply_elaboration expects - the elaboration call for a window sees the
    same flattened numbering.
    """
    raw_episodes = data.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError("Gemini returned no episodes for this window")

    episodes: list[DetectedEpisode] = []
    chunk_counter = 0
    for i, raw in enumerate(raw_episodes):
        try:
            ep_start = window_start + max(0.0, float(raw["start_sec"]))
            ep_end = window_start + float(raw["end_sec"])
            ep_end = min(ep_end, window_end)
            overall_task = str(raw["overall_task"]).strip()
            raw_chunks = raw.get("chunks")
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"Malformed episode at position {i}: {raw!r}") from err
        if not overall_task or ep_end <= ep_start or not isinstance(raw_chunks, list) or not raw_chunks:
            continue  # degenerate episode from the model; drop rather than fail the whole window

        chunks: list[SubtaskChunk] = []
        for c in raw_chunks:
            try:
                c_start = window_start + max(0.0, float(c["start_sec"]))
                c_end = min(window_start + float(c["end_sec"]), window_end)
                subtask = str(c["subtask"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
            if not subtask or c_end <= c_start:
                continue
            chunks.append(SubtaskChunk(
                index=chunk_counter, start_sec=round(c_start, 3), end_sec=round(c_end, 3), subtask=subtask,
            ))
            chunk_counter += 1
        if not chunks:
            continue

        episodes.append(DetectedEpisode(
            index=len(episodes), start_sec=round(ep_start, 3), end_sec=round(ep_end, 3),
            overall_task=overall_task, chunks=chunks,
        ))

    if not episodes:
        raise ValueError("Every episode Gemini returned for this window was degenerate")
    episodes.sort(key=lambda e: e.start_sec)
    return episodes


def apply_window_elaboration(episodes: list[DetectedEpisode], data: dict[str, Any]) -> None:
    """Same merge as apply_elaboration, just over every chunk across every
    episode in this window (they share one flat 0..N-1 index space)."""
    all_chunks = [c for ep in episodes for c in ep.chunks]
    apply_elaboration(all_chunks, data)


def parse_segmentation(data: dict[str, Any], *, duration: float) -> tuple[str, list[SubtaskChunk]]:
    """Validate and clamp stage-1 output. Raises ValueError on unusable data."""
    overall_task = str(data.get("overall_task") or "").strip()
    if not overall_task:
        raise ValueError("Gemini returned an empty overall_task")

    raw_chunks = data.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("Gemini returned no chunks")

    chunks: list[SubtaskChunk] = []
    for i, raw in enumerate(raw_chunks):
        try:
            start = max(0.0, float(raw["start_sec"]))
            end = min(duration, float(raw["end_sec"])) if duration else float(raw["end_sec"])
            subtask = str(raw["subtask"]).strip()
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"Malformed chunk at position {i}: {raw!r}") from err
        if not subtask:
            raise ValueError(f"Chunk at position {i} has an empty subtask")
        if end <= start:
            continue  # degenerate chunk from the model; drop it rather than fail the whole episode
        chunks.append(SubtaskChunk(index=len(chunks), start_sec=round(start, 3), end_sec=round(end, 3), subtask=subtask))

    if not chunks:
        raise ValueError("Every chunk Gemini returned was degenerate (end <= start)")

    chunks.sort(key=lambda c: c.start_sec)
    for i, c in enumerate(chunks):
        c.index = i
    return overall_task, chunks


def apply_elaboration(chunks: list[SubtaskChunk], data: dict[str, Any]) -> None:
    """Merge stage-2 output into the stage-1 chunks in place, by index."""
    raw_chunks = data.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ValueError("Gemini elaboration response had no chunks list")

    by_index = {c.index: c for c in chunks}
    for raw in raw_chunks:
        try:
            idx = int(raw["index"])
        except (KeyError, TypeError, ValueError):
            continue
        chunk = by_index.get(idx)
        if chunk is None:
            continue
        actions = raw.get("atomic_actions")
        if isinstance(actions, list):
            chunk.atomic_actions = [str(a).strip() for a in actions if str(a).strip()]
        chunk.is_recovery = bool(raw.get("is_recovery", False))
        if chunk.is_recovery:
            chunk.recovery_from = (str(raw.get("recovery_from")).strip() or None) if raw.get("recovery_from") else None
            chunk.recovery_action = (str(raw.get("recovery_action")).strip() or None) if raw.get("recovery_action") else None
        else:
            chunk.recovery_from = None
            chunk.recovery_action = None
