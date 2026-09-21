"""Local HTTP server backing the "Annotate whole video" button patched into
Annotator's GUI (see README's "GUI integration" section for the patch itself
and where it lives).

Annotator's frontend is a separate repo with no Gemini dependency and no
knowledge of this pipeline. Rather than importing gemini_annotator into
Annotator's own Flask process, the button in its browser page POSTs here -
on a different port, with CORS enabled - and this server does the actual
work: it's exactly pipeline.run_annotate_whole_video, reachable over HTTP.
Annotator's Python process and dependencies stay untouched.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import Flask, jsonify, request

from .config import Config
from .pipeline import DEFAULT_WINDOW_SECONDS, run_annotate_whole_video

log = logging.getLogger("gemini_annotator.server")


def create_app(config: Config, *, cache_dir: Path) -> Flask:
    app = Flask(__name__)

    @app.after_request
    def _cors(resp):
        # A local single-user tool; the browser tab calling this may be
        # reaching it through an SSH port-forward under an arbitrary
        # forwarded origin, so reflecting a fixed "*" is simplest.
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    @app.route("/annotate-whole-video", methods=["POST", "OPTIONS"])
    def annotate_whole_video():
        if request.method == "OPTIONS":
            return "", 204

        body = request.get_json(silent=True) or {}
        root = body.get("root")
        if not root:
            return jsonify({"ok": False, "error": "Missing 'root'"}), 400

        objects = body.get("objects")
        known_objects = None
        if objects:
            items = objects if isinstance(objects, list) else str(objects).split(",")
            known_objects = [o.strip() for o in items if o.strip()]

        try:
            result = run_annotate_whole_video(
                config, root,
                view=body.get("view") or None,
                known_objects=known_objects,
                window_seconds=float(body.get("window_seconds") or DEFAULT_WINDOW_SECONDS),
                cache_dir=cache_dir,
                save=True,
                on_progress=lambda msg: print(msg, file=sys.stderr, flush=True),
            )
        except Exception as err:
            log.exception("annotate-whole-video failed for root=%r", root)
            return jsonify({"ok": False, "error": str(err)}), 500

        return jsonify({
            "ok": True,
            "episodes_detected": result.clip_counts["episodes"],
            "subtask_clips": result.clip_counts["subtask"],
            "atomic_clips": result.clip_counts["atomic"],
            "recovery_clips": result.clip_counts["recovery"],
            "video_duration": result.video_duration,
        })

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"ok": True, "annotator_url": config.annotator_url, "model": config.gemini_model})

    return app


def run_server(config: Config, *, host: str, port: int, cache_dir: Path) -> None:
    app = create_app(config, cache_dir=cache_dir)
    print(f"  gemini-annotator server -> http://{host}:{port}")
    print(f"  will save into Annotator at {config.annotator_url}")
    app.run(host=host, port=port, threaded=True)
