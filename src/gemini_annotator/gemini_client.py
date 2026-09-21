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
from .prompts import (
    elaboration_prompt,
    segmentation_prompt,
    window_elaboration_prompt,
    window_segmentation_prompt,
)
from .schema import (
    ELABORATION_SCHEMA,
    SEGMENTATION_SCHEMA,
    WINDOW_SEGMENTATION_SCHEMA,
    DetectedEpisode,
    EpisodeAnnotation,
    SubtaskChunk,
    apply_elaboration,
    apply_window_elaboration,
    parse_segmentation,
    parse_window_segmentation,
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
    known_objects: list[str] | None = None,
) -> EpisodeAnnotation:
    """Run the segmentation + elaboration stages against one video clip.

    `duration` is the clip's length in seconds as measured locally (ffprobe),
    used to validate/clamp what Gemini reports rather than trusting it blindly.
    `known_objects`, if given, is a closed list of object names Gemini is told
    to use verbatim - useful when the model tends to misidentify props (e.g.
    calling a leek "broccoli") and you'd rather pin the vocabulary than hope
    it infers it correctly from pixels alone.
    """
    client = _client(config)
    uploaded = _upload_and_wait(client, video_path)
    try:
        seg_data = _generate_json(
            client,
            config.gemini_model,
            [uploaded, segmentation_prompt(known_task=known_task, duration=duration, known_objects=known_objects)],
            SEGMENTATION_SCHEMA,
        )
        overall_task, chunks = parse_segmentation(seg_data, duration=duration)

        elab_data = _generate_json(
            client,
            config.gemini_model,
            [uploaded, elaboration_prompt(overall_task=overall_task, chunks=chunks, known_objects=known_objects)],
            ELABORATION_SCHEMA,
        )
        apply_elaboration(chunks, elab_data)
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass  # best-effort cleanup; uploaded files also expire on their own

    return EpisodeAnnotation(overall_task=overall_task, chunks=chunks)


def annotate_window(
    config: Config,
    video_path: Path,
    *,
    window_start: float,
    window_end: float,
    known_objects: list[str] | None = None,
) -> list[DetectedEpisode]:
    """Whole-video mode: run episode-boundary + subtask detection, then
    elaboration, against one WINDOW of a longer, multi-episode video.

    `video_path` is already trimmed to this window (window_end - window_start
    seconds) - see episode_source.resolve_whole_video for how the window is
    cut out of a (possibly multi-file) view. `window_start`/`window_end` are
    this window's offsets on the whole video's global axis, used to shift the
    window-local times Gemini reports (0..window duration) onto that axis.
    """
    window_duration = window_end - window_start
    client = _client(config)
    uploaded = _upload_and_wait(client, video_path)
    try:
        seg_data = _generate_json(
            client,
            config.gemini_model,
            [uploaded, window_segmentation_prompt(window_duration=window_duration, known_objects=known_objects)],
            WINDOW_SEGMENTATION_SCHEMA,
        )
        episodes = parse_window_segmentation(seg_data, window_start=window_start, window_end=window_end)

        elab_data = _generate_json(
            client,
            config.gemini_model,
            [uploaded, window_elaboration_prompt(episodes=episodes, known_objects=known_objects)],
            ELABORATION_SCHEMA,
        )
        apply_window_elaboration(episodes, elab_data)
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass

    return episodes
