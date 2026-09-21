"""Round-trips against a REAL running Annotator server - no Gemini calls, no
API key needed. Skipped by default; opt in with two env vars pointing at a
server and a dataset it can open (see scripts/setup_test_dataset.sh and the
README's Testing section):

    ANNOTATOR_TEST_URL=http://127.0.0.1:5111 \\
    ANNOTATOR_TEST_ROOT=/tmp/gemini-annotator-smoketest/dataset \\
    pytest tests/test_annotator_integration.py -v
"""

from __future__ import annotations

import os

import pytest

from gemini_annotator.annotator_client import AnnotatorClient
from gemini_annotator.episode_source import list_episodes, pick_view, resolve_episode_clip
from gemini_annotator.project_builder import apply_episode_annotation
from gemini_annotator.schema import EpisodeAnnotation, SubtaskChunk

ANNOTATOR_URL = os.environ.get("ANNOTATOR_TEST_URL")
ANNOTATOR_ROOT = os.environ.get("ANNOTATOR_TEST_ROOT")

pytestmark = pytest.mark.skipif(
    not (ANNOTATOR_URL and ANNOTATOR_ROOT),
    reason="set ANNOTATOR_TEST_URL and ANNOTATOR_TEST_ROOT to test against a live Annotator "
           "server (see scripts/setup_test_dataset.sh and the README's Testing section)",
)


@pytest.fixture
def client() -> AnnotatorClient:
    return AnnotatorClient(ANNOTATOR_URL)


def test_get_session_returns_a_usable_timeline(client: AnnotatorClient):
    session = client.get_session(ANNOTATOR_ROOT, scope="dataset")
    assert session["api"] >= 2
    assert session["timeline"]["view_keys"], "dataset has no video views - wrong ANNOTATOR_TEST_ROOT?"


def test_media_download_and_trim(client: AnnotatorClient, tmp_path):
    session = client.get_session(ANNOTATOR_ROOT, scope="dataset")
    timeline = session["timeline"]
    view = pick_view(timeline, None)
    episode = list_episodes(timeline, view)[0]

    clip = resolve_episode_clip(client, ANNOTATOR_ROOT, timeline, episode, view, tmp_path)

    assert clip.is_file()
    assert clip.stat().st_size > 0


def test_save_project_round_trips_through_the_real_server(client: AnnotatorClient, tmp_path):
    session = client.get_session(ANNOTATOR_ROOT, scope="dataset")
    timeline = session["timeline"]
    project = session["project"]
    view = pick_view(timeline, None)
    episode = list_episodes(timeline, view)[0]

    annotation = EpisodeAnnotation(
        overall_task="integration-test overall task",
        chunks=[
            SubtaskChunk(
                0, 0.0, episode.global_end - episode.global_start,
                "integration-test chunk", atomic_actions=["step a", "step b"],
                is_recovery=True, recovery_from="integration-test failure",
                recovery_action="integration-test recovery action",
            ),
        ],
    )
    apply_episode_annotation(
        project, annotation,
        episode_global_start=episode.global_start, episode_global_end=episode.global_end,
    )
    result = client.save_project(project)
    assert result["ok"] is True

    # Fetch a FRESH session - proves the server, not just our in-memory object, has it.
    reloaded = client.get_session(ANNOTATOR_ROOT, scope="dataset")["project"]

    def clips_for(style_name: str) -> list[dict]:
        style_id = next(s["id"] for s in reloaded["styles"] if s["name"] == style_name)
        layer = next(l for l in reloaded["layers"] if l["style_id"] == style_id)
        return layer["clips"]

    assert any(c["text"] == "integration-test overall task" for c in clips_for("main"))
    assert any(c["text"] == "integration-test chunk" for c in clips_for("subtask"))
    assert any(c["text"] == "step a; step b" for c in clips_for("atomic"))
    assert any("integration-test recovery action" in c["text"] for c in clips_for("recovery"))
