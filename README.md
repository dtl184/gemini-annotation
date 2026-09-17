# gemini-annotator-pipeline

A Gemini-based annotation pipeline for robot demonstration videos, in the
spirit of the [Steerable Vision-Language-Action Policies](https://steerable-policies.github.io/)
pipeline: it watches a video and labels it at multiple levels of
abstraction - overall task, per-chunk subtasks, per-chunk atomic actions -
so it can be used to train a policy that is steerable at any of those
levels.

Two differences from that paper's pipeline:

1. **No scene feature extraction.** The paper's pipeline runs Molmo (object
   grounding) + SAM2 (mask tracking) + DETR (gripper traces) before ever
   calling Gemini. This pipeline hands Gemini the raw video directly and
   relies on its native video understanding instead - simpler, at the cost
   of not producing pointing/gripper-trace-style commands.
2. **Recovery actions.** Most demos are clean, successful executions with no
   failure in them at all - but some of their chunks are still, by their
   nature, exactly what you'd want a policy to do if it *had* failed. "Pick
   up the fork off the table" is a plausible recovery for "dropped the
   fork" whether or not this particular video ever shows a drop. This
   pipeline asks Gemini to judge each chunk on that basis - would this
   action be the right recovery for some common VLA/manipulation failure
   (dropped grasp, missed grasp, knocked-over object, overshoot, wrong
   placement)? - and labels the ones that qualify with the recovery action,
   independent of whether a failure is actually visible earlier in the
   video. The idea: this supervision is already latent in ordinary
   demonstration footage, free for the taking, rather than something that
   has to be collected by staging real failures.

It writes directly into [ogoudey/Annotator](https://github.com/ogoudey/Annotator),
a Flask GUI for annotating LeRobot v3 datasets with time-stamped clips, over
Annotator's HTTP API - so the result is immediately visible and editable in
the GUI, and from there exportable into a LeRobot dataset's
`language_persistent` column for training.

## How it fits together

```
                      HTTP (Annotator's own API)
  ┌─────────────────┐  GET  /api/session   ┌───────────────────┐
  │ gemini-annotator │  GET  /media         │  Annotator (Flask)│
  │     pipeline     │◄─────────────────────┤   ogoudey/Annotator│
  │                   │  POST /api/project   │                    │
  │  1. fetch session │─────────────────────►│  reads/writes      │
  │  2. download +     │  POST /api/export/  │  annotations/*.json│
  │     trim episode   │       lerobot        │  and, on export,   │
  │     clip (ffmpeg)  │                      │  the dataset's     │
  │  3. ask Gemini      │                      │  parquet shards    │
  │     (2 calls)       │                      └───────────────────┘
  │  4. merge clips into                              ▲
  │     the project JSON                               │ open in browser
  └──────────────────────┘                     http://127.0.0.1:5111
```

The pipeline never reads the dataset directory itself - every fact about it
(views, file layout, episode boundaries, fps, the current annotation state)
comes from Annotator's API, and every video byte comes from `GET /media`.
Annotator's own scanner (`dataset.py`) stays the single source of truth for
"what does this dataset look like," which is also why this pipeline doesn't
need to vendor any of Annotator's LeRobot-parsing code.

## Setup

```bash
git clone https://github.com/ogoudey/Annotator
cd Annotator
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py --data-root ~/datasets   # http://127.0.0.1:5111
```

```bash
git clone <this repo>
cd gemini-annotator-pipeline
uv venv && uv pip install -e .
export GEMINI_API_KEY=...   # already the case if it's in your shell rc
```

## Usage

Annotate every episode of a LeRobot v3 dataset that's open (or openable) in
Annotator, and push the result straight into the running GUI:

```bash
gemini-annotator annotate --root ~/datasets/fold_towel
```

Just a few episodes, a specific camera view, and a known task hint (skips
Gemini having to guess the overall task):

```bash
gemini-annotator annotate \
  --root ~/datasets/fold_towel \
  --view observation.images.top \
  --episodes 0,1,2 \
  --task "fold the towel in half"
```

Preview only - runs Gemini but doesn't save into Annotator:

```bash
gemini-annotator annotate --root ~/datasets/fold_towel --no-save
```

Also bake the result into the dataset's `language_persistent` column
(dry-run by default; add `--write` to actually rewrite the parquet shards -
same safeguard Annotator's own UI uses):

```bash
gemini-annotator annotate --root ~/datasets/fold_towel --export-lerobot
gemini-annotator annotate --root ~/datasets/fold_towel --export-lerobot --write
```

### Standalone videos

Annotator only opens LeRobot dataset directories (it looks for
`meta/info.json`). To annotate a plain mp4 that isn't part of one, wrap it
first:

```bash
gemini-annotator scaffold --video ~/clips/demo.mp4 --out ~/datasets/demo_wrapped
gemini-annotator annotate --root ~/datasets/demo_wrapped
```

`scaffold` hardlinks (falls back to copying) the video into the minimal
directory shape Annotator's scanner recognizes. It does **not** produce a
trainable LeRobot dataset (no per-frame state/action parquet), so
`--export-lerobot` doesn't apply to a scaffolded video - only the GUI view
and JSON/CSV export do.

## What gets written

Four styles/layers in Annotator's project schema, per episode:

| style      | one clip per...     | text                                                              |
|------------|----------------------|--------------------------------------------------------------------|
| `main`     | episode              | the overall task                                                   |
| `subtask`  | chunk                | e.g. "pick up the fork"                                             |
| `atomic`   | chunk                | that chunk's atomic actions, e.g. "reach; close gripper; lift"      |
| `recovery` | chunk, only if flagged | the recovery action, e.g. "pick the fork back up off the table" |

`main` and `recovery` reuse Annotator's own default styles (`recovery`
ships as one of Annotator's three default styles for exactly this purpose);
`subtask` and `atomic` are added if the project doesn't already have them.
Re-running `annotate` for an episode replaces that episode's clips in these
four layers rather than duplicating them, so it's safe to re-run.

On `--export-lerobot`, styles map onto LeRobot's canonical persistent styles
as: `main` → `task_aug`, `subtask` → `subtask`, `atomic` → `motion`,
`recovery` stays `recovery` (registered as a custom style automatically -
see `lerobot_io.py` in Annotator).

## How annotation works

Two Gemini calls per episode, both against the same uploaded clip:

1. **Segmentation** - watch the whole episode, name the overall task, and
   split it into contiguous, non-overlapping time chunks, each a short
   subtask phrase. The model is told explicitly not to smooth over
   failures: a drop-and-recover is two chunks, not one.
2. **Elaboration** - given those chunk boundaries (and the video again),
   name each chunk's atomic actions, and judge whether the chunk's action is
   a plausible recovery for a common VLA/manipulation failure. This is
   judged from what the action *is* (e.g. "pick the object back up off the
   table"), not from whether a failure actually appears earlier in the
   video - most source videos will be clean, successful demos with nothing
   to visibly recover from, and that's fine.

See `src/gemini_annotator/prompts.py` for the exact prompts and
`src/gemini_annotator/schema.py` for the JSON schemas Gemini is constrained
to.

## Development

```bash
uv pip install -e '.[dev]'  # or: uv pip install -e . pytest
pytest
```

Tests cover the schema parsing and project-merging logic (no network calls,
no Gemini API key needed). There's no offline test for the Gemini or
Annotator HTTP calls themselves - those need a live server/API key to
exercise meaningfully.
