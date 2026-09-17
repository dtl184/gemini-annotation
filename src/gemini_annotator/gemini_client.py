"""Thin wrapper around google-genai for the two annotation calls.

We upload the (already trimmed, see ffmpeg_utils.py) episode clip as a File,
wait for it to finish processing, then run segmentation and elaboration as
two separate generate_content calls against the same uploaded file so both
stages see the identical video.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from google import genai
from google.genai import types

from .config import Config
from .prompts import elaboration_prompt, segmentation_prompt
from .schema import (
    ELABORATION_SCHEMA,
    SEGMENTATION_SCHEMA,
    EpisodeAnnotation,
    SubtaskChunk,
    apply_elaboration,
    parse_segmentation,
)

UPLOAD_POLL_INTERVAL_S = 2.0
UPLOAD_TIMEOUT_S = 300.0


class GeminiAnnotationError(RuntimeError):
    pass


def _client(config: Config) -> genai.Client:
    return genai.Client(api_key=config.gemini_api_key)


def _upload_and_wait(client: genai.Client, video_path: Path) -> types.File:
    uploaded = client.files.upload(
        file=str(video_path),
        config=types.UploadFileConfig(mime_type="video/mp4", display_name=video_path.name),
    )
    deadline = time.monotonic() + UPLOAD_TIMEOUT_S
    while uploaded.state == types.FileState.PROCESSING:
        if time.monotonic() > deadline:
            raise GeminiAnnotationError(f"Timed out waiting for {video_path} to finish processing")
        time.sleep(UPLOAD_POLL_INTERVAL_S)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state != types.FileState.ACTIVE:
        raise GeminiAnnotationError(f"Gemini file upload for {video_path} ended in state {uploaded.state!r}")
    return uploaded


def _generate_json(
    client: genai.Client, model: str, contents: list, schema: dict
) -> dict:
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0.2,
        ),
    )
    text = response.text
    if not text:
        raise GeminiAnnotationError("Gemini returned an empty response")
    try:
        return json.loads(text)
    except json.JSONDecodeError as err:
        raise GeminiAnnotationError(f"Gemini response was not valid JSON: {err}\n---\n{text[:2000]}") from err


def annotate_video(
    config: Config,
    video_path: Path,
    *,
    duration: float,
    known_task: str | None = None,
) -> EpisodeAnnotation:
    """Run the segmentation + elaboration stages against one video clip.

    `duration` is the clip's length in seconds as measured locally (ffprobe),
    used to validate/clamp what Gemini reports rather than trusting it blindly.
    """
    client = _client(config)
    uploaded = _upload_and_wait(client, video_path)
    try:
        seg_data = _generate_json(
            client,
            config.gemini_model,
            [uploaded, segmentation_prompt(known_task=known_task, duration=duration)],
            SEGMENTATION_SCHEMA,
        )
        overall_task, chunks = parse_segmentation(seg_data, duration=duration)

        elab_data = _generate_json(
            client,
            config.gemini_model,
            [uploaded, elaboration_prompt(overall_task=overall_task, chunks=chunks)],
            ELABORATION_SCHEMA,
        )
        apply_elaboration(chunks, elab_data)
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass  # best-effort cleanup; uploaded files also expire on their own

    return EpisodeAnnotation(overall_task=overall_task, chunks=chunks)
