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

Just a few episodes, a specific camera view, a known task hint (skips Gemini
having to guess the overall task), and a closed object vocabulary (useful
when Gemini tends to misidentify props from pixels alone - e.g. calling a
leek "broccoli"):

```bash
gemini-annotator annotate \
  --root ~/datasets/fold_towel \
  --view observation.images.top \
  --episodes 0,1,2 \
  --task "fold the towel in half" \
  --objects "carrot, leek, green pepper"
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

### Whole-video mode

`annotate` (above) processes one episode at a time, using the dataset's own
episode metadata for boundaries - accurate, but one Gemini call pair per
episode (100 calls for a 50-episode dataset). `annotate-whole` instead makes
Gemini find the episode boundaries itself by watching the video - useful
when you don't trust/want the metadata, or there isn't any (a scaffolded
standalone video with several demos stitched into one file):

```bash
gemini-annotator annotate-whole --root ~/datasets/fold_towel
```

It processes the whole recording in a handful of short, overlapping-aware
windows rather than one call over the whole thing (accuracy at finding
boundaries degrades sharply with how much video is in view at once - see
`pipeline.run_annotate_whole_video`'s docstring for the measurements behind
`--window-seconds`'s default). This is also what backs the GUI's "Annotate
whole video" button, below.

## GUI integration

Annotator's own toolbar has an **Annotate whole video** button (next to its
existing, unfinished `Auto-annotate` stub) that runs the same whole-video
pipeline as `annotate-whole` above, from inside the browser, on demand -
**only** when clicked, never on opening a dataset or automatically in the
background.

This needs two small changes to your local Annotator checkout - **not**
upstream in ogoudey/Annotator, so pull latest there and re-apply if you
update it:

- `templates/index.html`: one new `<button data-action="annotate-whole-video">`
- `static/app.js`: the `annotate-whole-video` action, wired to `fetch()` a
  separate local server (below) and refresh the session on success

Annotator's own Flask process and dependencies are untouched - no Gemini
import, no new pip requirement in its venv. The button instead calls a
second, separate server that this pipeline runs on its own port:

```bash
gemini-annotator serve --annotator-url http://127.0.0.1:5111 --port 5114
```

`serve` is a thin HTTP wrapper around `run_annotate_whole_video`: the button
POSTs `{root}` to `http://<host>:5114/annotate-whole-video`, it runs Gemini
and saves via Annotator's own `/api/project` (exactly like the CLI does),
and the button's `fetch()` refreshes the open session once it returns. CORS
is wide open (`Access-Control-Allow-Origin: *`) since this is a local,
single-user tool.

**Over SSH / VSCode Remote:** forward both ports (`5111` for Annotator,
`5114` for this server) - the Ports tab, or `ssh -L 5111:127.0.0.1:5111 -L
5114:127.0.0.1:5114 <host>`. The button computes the pipeline server's URL
from `location.hostname`, so it resolves correctly through the forward
without editing `app.js`.

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

## Testing

There are three layers, from cheapest to most realistic. None of them need
your real robot data - `scripts/setup_test_dataset.sh` generates a tiny
synthetic video and scaffolds it into an openable dataset.

**1. Unit tests** - schema parsing and project-merging logic, no network, no
API key, no server:

```bash
uv pip install -e . pytest
pytest
```

**2. API wiring, against a real Annotator server, no Gemini calls (free)** -
this is what exercises `/api/session`, `/media`, and `/api/project` for
real: session fetch, downloading + `ffmpeg`-trimming an episode clip, and a
full save → fresh-fetch → verify-it-persisted round trip.

```bash
# terminal 1: a real Annotator server
git clone https://github.com/ogoudey/Annotator && cd Annotator
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python app.py --data-root /tmp/gemini-annotator-smoketest --port 5111

# terminal 2: build the synthetic dataset, then point the integration tests at that server
cd gemini-annotator-pipeline
scripts/setup_test_dataset.sh /tmp/gemini-annotator-smoketest
ANNOTATOR_TEST_URL=http://127.0.0.1:5111 \
  ANNOTATOR_TEST_ROOT=/tmp/gemini-annotator-smoketest/dataset \
  pytest tests/test_annotator_integration.py -v
```

These are skipped (not failed) when the env vars aren't set, so a plain
`pytest` stays fast and offline. Open `http://127.0.0.1:5111` in a browser
afterward to see the round-tripped clips in the actual GUI.

**3. A real Gemini call, on the same tiny clip** - confirms the prompts,
JSON schema, and response parsing actually work against the live model.
Costs a few seconds of a 6s synthetic clip, not your dataset:

```bash
gemini-annotator annotate --root /tmp/gemini-annotator-smoketest/dataset --no-save
```

`--no-save` runs Gemini and prints the result without writing anything back
to Annotator. Drop it once you trust the output, to actually save.
