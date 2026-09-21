"""Prompt templates for the two Gemini stages. Kept as plain functions rather
than a templating engine - there are only two of them and they are short."""

from __future__ import annotations


def _objects_hint(known_objects: list[str] | None) -> str:
    if not known_objects:
        return ""
    names = ", ".join(known_objects)
    return (
        f"\n\nThe only objects that may appear in this scene are: {names}. Identify objects by "
        f"these exact names - the video may make them easy to mistake for something else (e.g. "
        f"a similarly-shaped or colored object not in this list), so when in doubt pick the "
        f"closest match from this list rather than inventing a different object name."
    )


def segmentation_prompt(*, known_task: str | None, duration: float, known_objects: list[str] | None = None) -> str:
    task_hint = (
        f'The dataset labels this episode\'s overall task as: "{known_task}". '
        "Treat that as context for what session/scene this is, but overall_task should still "
        "be a specific, concrete sentence describing what actually happens in THIS video (which "
        "objects are manipulated, and what is done with them) - not a copy of the dataset label, "
        "which may be generic (e.g. a session or scene name rather than a real instruction)."
        if known_task
        else "Infer the overall task from what the robot arm actually accomplishes."
    )
    return f"""You are watching a {duration:.1f}-second video of a robot arm performing a manipulation
task. {task_hint}{_objects_hint(known_objects)}

Break the ENTIRE video into a sequence of contiguous time chunks, covering the whole
duration with no gaps and no overlaps (chunk[i].end_sec == chunk[i+1].start_sec), from
0.0 to {duration:.2f}. Each chunk is one semantic subtask: a short, human-readable,
imperative phrase describing what the arm is doing in that chunk, e.g. "reach for the
carrot", "grasp the container", "pick up the fork", "place the fork on the plate".

A new chunk should start whenever the subtask changes - including if the robot fails,
drops something, overshoots, retries, or otherwise deviates from a clean execution. If
that happens, don't smooth it away: a drop followed by picking the object back up is two
chunks, not one. Use timestamps you can actually see on screen; do not guess round
numbers.

Respond with JSON matching the provided schema only."""


def _elaboration_fields_instructions() -> str:
    return """- atomic_actions: the ordered low-level atomic motions that make up the chunk, as short
  phrases (2-5 words each), e.g. ["move left", "lower gripper", "close gripper", "lift"].
  These should be finer-grained than the subtask - the individual movements a policy
  would need to execute, not a restatement of the subtask.

- is_recovery: true if this chunk's action is the kind of thing that would serve as the
  right recovery for a COMMON, PLAUSIBLE VLA/manipulation failure - regardless of
  whether this video actually shows that failure happening. Judge the action on its own
  merits, not by looking for evidence of a mistake earlier in the video: this video may
  well be a clean, successful demonstration with no failure in it at all, and a chunk can
  still be a recovery action. For example, a chunk where the arm "picks up the fork off
  the table" IS a recovery-eligible action, because it is exactly what a policy should do
  after dropping the fork - even if this particular video never shows the fork being
  dropped. Common failure modes to consider: dropping a grasped object, missing/failing a
  grasp attempt, knocking an object over or out of position, overshooting or colliding
  with something, placing an object in the wrong spot, or losing grip and needing to
  re-grasp. A chunk that is just an unremarkable forward step of the task with no
  plausible framing as fixing one of these (e.g. "move the empty gripper toward the next
  object") is NOT a recovery.

- recovery_from: if is_recovery, a short phrase naming the common failure this action
  would recover from (e.g. "dropped the fork while lifting it"). This describes a
  hypothetical failure mode, not something that necessarily happened in this video.
  Otherwise null.

- recovery_action: if is_recovery, a short imperative phrase for the recovery behavior
  itself, phrased so it could be used as a steering command to a policy (e.g. "pick the
  fork back up off the table"). Otherwise null."""


def elaboration_prompt(*, overall_task: str, chunks: list, known_objects: list[str] | None = None) -> str:
    chunk_lines = "\n".join(
        f'{c.index}. [{c.start_sec:.2f}s - {c.end_sec:.2f}s] "{c.subtask}"' for c in chunks
    )
    return f"""You are watching the same robot-arm video again. The overall task is:
"{overall_task}"{_objects_hint(known_objects)}

It has already been segmented into these subtask chunks:
{chunk_lines}

For EVERY chunk listed above, return:

{_elaboration_fields_instructions()}

Return one entry per chunk index, 0 through {len(chunks) - 1}, matching the schema."""


def window_segmentation_prompt(*, window_duration: float, known_objects: list[str] | None = None) -> str:
    return f"""This {window_duration:.1f}-second video is one or more separate short robot
manipulation demonstration episodes concatenated back-to-back with NO transition - raw
footage cut together, not a single continuous recording. At each episode boundary there
is an abrupt jump: the arm and any objects instantly reset to a new starting position or
state, unlike anything a continuous motion would produce. There may be only one episode
in this window, or several.{_objects_hint(known_objects)}

Watch the whole window and, for each episode in it, in order, return:

- start_sec, end_sec: contiguous across episodes (episode[i].end_sec ==
  episode[i+1].start_sec), covering the full window from 0.0 to {window_duration:.2f} with
  no gaps or overlaps. Use timestamps you can actually see on screen; do not guess round
  numbers, and do not report a time beyond {window_duration:.2f}.
- overall_task: one sentence describing what that episode accomplishes.
- chunks: break that episode into subtask chunks (contiguous within the episode, same
  start/end rule as above), each a short imperative phrase, e.g. "reach for the carrot",
  "grasp the container", "pick up the fork", "place the fork on the plate". A new chunk
  should start whenever the subtask changes, including failures/retries within an
  episode - don't smooth those away.

If the window ends mid-episode (the last episode doesn't get a clean reset before the
window runs out), still report it with end_sec at {window_duration:.2f} - it will be
reconciled against the next window.

Respond with JSON matching the provided schema only."""


def window_elaboration_prompt(*, episodes: list, known_objects: list[str] | None = None) -> str:
    lines = []
    for ep in episodes:
        lines.append(f'Episode: "{ep.overall_task}"')
        for c in ep.chunks:
            lines.append(f'  {c.index}. [{c.start_sec:.2f}s - {c.end_sec:.2f}s] "{c.subtask}"')
    chunk_lines = "\n".join(lines)
    return f"""You are watching the same window of robot-arm video again. It has already been
segmented into these episodes and, within each, subtask chunks:{_objects_hint(known_objects)}

{chunk_lines}

For EVERY chunk listed above (referenced by its number, regardless of which episode it's
under), return:

{_elaboration_fields_instructions()}

Return one entry per chunk index shown above, matching the schema."""
