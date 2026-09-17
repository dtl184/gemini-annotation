"""Prompt templates for the two Gemini stages. Kept as plain functions rather
than a templating engine - there are only two of them and they are short."""

from __future__ import annotations


def segmentation_prompt(*, known_task: str | None, duration: float) -> str:
    task_hint = (
        f'The dataset labels this episode\'s overall task as: "{known_task}". '
        "Use that as the overall_task unless the video clearly shows something else."
        if known_task
        else "Infer the overall task from what the robot arm actually accomplishes."
    )
    return f"""You are watching a {duration:.1f}-second video of a robot arm performing a manipulation
task. {task_hint}

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


def elaboration_prompt(*, overall_task: str, chunks: list) -> str:
    chunk_lines = "\n".join(
        f'{c.index}. [{c.start_sec:.2f}s - {c.end_sec:.2f}s] "{c.subtask}"' for c in chunks
    )
    return f"""You are watching the same robot-arm video again. The overall task is:
"{overall_task}"

It has already been segmented into these subtask chunks:
{chunk_lines}

For EVERY chunk listed above, return:

- atomic_actions: the ordered low-level atomic motions that make up the chunk, as short
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
  fork back up off the table"). Otherwise null.

Return one entry per chunk index, 0 through {len(chunks) - 1}, matching the schema."""
