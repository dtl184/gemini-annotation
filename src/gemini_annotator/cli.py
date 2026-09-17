from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_ANNOTATOR_URL, DEFAULT_MODEL, load_config
from .pipeline import run_annotate
from .scaffold import DEFAULT_VIEW_NAME, ScaffoldError, scaffold_dataset

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "gemini-annotator-pipeline"


def _cmd_annotate(args: argparse.Namespace) -> int:
    config = load_config(model=args.model, annotator_url=args.annotator_url)
    episode_indices = None
    if args.episodes:
        episode_indices = [int(x) for x in args.episodes.split(",") if x.strip()]

    try:
        results = run_annotate(
            config,
            args.root,
            view=args.view,
            episode_indices=episode_indices,
            known_task=args.task,
            cache_dir=Path(args.cache_dir),
            save=not args.no_save,
            export_lerobot=args.export_lerobot,
            export_dry_run=not args.write,
            on_progress=lambda msg: print(msg, file=sys.stderr),
        )
    except Exception as err:  # surfaced as a clean CLI error, not a traceback
        print(f"error: {err}", file=sys.stderr)
        return 1

    print(f"Annotated {len(results)} episode(s).")
    for r in results:
        print(
            f"  episode {r.episode.episode_index}: "
            f"main=1 subtask={r.clip_counts['subtask']} "
            f"atomic={r.clip_counts['atomic']} recovery={r.clip_counts['recovery']}"
        )
    return 0


def _cmd_scaffold(args: argparse.Namespace) -> int:
    try:
        out = scaffold_dataset(
            Path(args.video), Path(args.out),
            view_name=args.view_name, fps=args.fps,
        )
    except ScaffoldError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(f"Scaffolded a minimal LeRobot-shaped dataset at {out}")
    print("Open it from Annotator's Open dialog, or run:")
    print(f"  gemini-annotator annotate --root {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemini-annotator",
        description="Gemini video annotation pipeline for ogoudey/Annotator: labels robot demo "
                    "videos with an overall task, per-chunk subtasks and atomic actions, and "
                    "recovery-action chunks, then pushes them into a running Annotator server.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ann = sub.add_parser("annotate", help="Annotate episodes in a LeRobot dataset (or scaffolded video).")
    ann.add_argument("--root", required=True, help="Dataset root directory, as Annotator would open it.")
    ann.add_argument("--annotator-url", default=None, help=f"Default: {DEFAULT_ANNOTATOR_URL} or $ANNOTATOR_URL")
    ann.add_argument("--model", default=None, help=f"Gemini model. Default: {DEFAULT_MODEL} or $GEMINI_MODEL")
    ann.add_argument("--view", default=None, help="Video view/camera key to use. Default: first available.")
    ann.add_argument("--episodes", default=None, help="Comma-separated episode indices. Default: all.")
    ann.add_argument("--task", default=None, help="Known overall task text, used as a hint for every episode.")
    ann.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Where downloaded/trimmed clips are cached.")
    ann.add_argument("--no-save", action="store_true", help="Don't POST the result back to Annotator (dry run).")
    ann.add_argument("--export-lerobot", action="store_true", help="Also call /api/export/lerobot after saving.")
    ann.add_argument("--write", action="store_true", help="With --export-lerobot: actually write (default is dry-run).")
    ann.set_defaults(func=_cmd_annotate)

    scaf = sub.add_parser("scaffold", help="Wrap a standalone mp4 into a minimal dataset Annotator can open.")
    scaf.add_argument("--video", required=True, help="Path to the source mp4.")
    scaf.add_argument("--out", required=True, help="Output dataset directory to create (must not exist/be empty).")
    scaf.add_argument("--view-name", default=DEFAULT_VIEW_NAME, help=f"Default: {DEFAULT_VIEW_NAME}")
    scaf.add_argument("--fps", type=float, default=None, help="Default: probed from the video.")
    scaf.set_defaults(func=_cmd_scaffold)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
