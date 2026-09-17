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
