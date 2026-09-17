from __future__ import annotations

from gemini_annotator.project_builder import apply_episode_annotation
from gemini_annotator.schema import EpisodeAnnotation, SubtaskChunk


def _empty_project() -> dict:
    return {
        "format": "segment-annotations",
        "version": 2,
        "scope": "dataset",
        "styles": [
            {"id": "st_main", "name": "main", "color": "#6EA8FF"},
            {"id": "st_recovery", "name": "recovery", "color": "#7ED9A7"},
            {"id": "st_gripper", "name": "gripper", "color": "#E2A0FF"},
        ],
        "layers": [
            {"id": "ly_1", "style_id": "st_main", "clips": []},
            {"id": "ly_2", "style_id": "st_recovery", "clips": []},
            {"id": "ly_3", "style_id": "st_gripper", "clips": []},
        ],
    }


def _layer(project: dict, style_name: str) -> dict:
    style_id = next(s["id"] for s in project["styles"] if s["name"] == style_name)
    return next(l for l in project["layers"] if l["style_id"] == style_id)


def _annotation() -> EpisodeAnnotation:
    return EpisodeAnnotation(
        overall_task="pick up the fork and place it on the plate",
        chunks=[
            SubtaskChunk(0, 0.0, 2.0, "reach for the fork", atomic_actions=["reach", "open gripper"]),
            SubtaskChunk(1, 2.0, 3.0, "dropped the fork", atomic_actions=["close gripper", "lift"]),
            SubtaskChunk(
                2, 3.0, 5.0, "pick the fork back up", atomic_actions=["reach", "close gripper"],
                is_recovery=True, recovery_from="dropped the fork while lifting it",
                recovery_action="pick the fork back up off the table",
            ),
        ],
    )


def test_apply_episode_annotation_reuses_default_styles_and_creates_new_ones():
    project = _empty_project()
    counts = apply_episode_annotation(
        project, _annotation(), episode_global_start=100.0, episode_global_end=105.0,
    )
    assert counts == {"main": 1, "subtask": 3, "atomic": 3, "recovery": 1}

    style_names = {s["name"] for s in project["styles"]}
    assert style_names == {"main", "recovery", "gripper", "subtask", "atomic"}
    # existing default style ids must be reused, not duplicated
    assert sum(1 for s in project["styles"] if s["name"] == "main") == 1
    assert next(s["id"] for s in project["styles"] if s["name"] == "main") == "st_main"
    assert next(s["id"] for s in project["styles"] if s["name"] == "recovery") == "st_recovery"

    main_layer = _layer(project, "main")
    assert len(main_layer["clips"]) == 1
    assert main_layer["clips"][0]["start"] == 100.0
    assert main_layer["clips"][0]["end"] == 105.0
    assert main_layer["clips"][0]["text"] == "pick up the fork and place it on the plate"

    recovery_layer = _layer(project, "recovery")
    assert len(recovery_layer["clips"]) == 1
    assert recovery_layer["clips"][0]["start"] == 103.0
    assert "pick the fork back up off the table" in recovery_layer["clips"][0]["text"]
    assert "dropped the fork while lifting it" in recovery_layer["clips"][0]["text"]

    subtask_layer = _layer(project, "subtask")
    assert [c["text"] for c in subtask_layer["clips"]] == [
        "reach for the fork", "dropped the fork", "pick the fork back up",
    ]

    atomic_layer = _layer(project, "atomic")
    assert atomic_layer["clips"][0]["text"] == "reach; open gripper"


def test_apply_episode_annotation_is_idempotent_and_scoped_to_its_range():
    project = _empty_project()
    apply_episode_annotation(project, _annotation(), episode_global_start=0.0, episode_global_end=5.0)
    apply_episode_annotation(project, _annotation(), episode_global_start=10.0, episode_global_end=15.0)

    # re-running episode 1 must not touch episode 2's clips
    apply_episode_annotation(project, _annotation(), episode_global_start=0.0, episode_global_end=5.0)

    subtask_layer = _layer(project, "subtask")
    assert len(subtask_layer["clips"]) == 6  # 3 per episode, no duplicates
    starts = sorted(c["start"] for c in subtask_layer["clips"])
    assert starts == [0.0, 2.0, 3.0, 10.0, 12.0, 13.0]


def test_apply_episode_annotation_skips_recovery_clip_when_none_flagged():
    project = _empty_project()
    annotation = EpisodeAnnotation(
        overall_task="fold the towel",
        chunks=[SubtaskChunk(0, 0.0, 3.0, "fold the towel", atomic_actions=["grasp", "fold"])],
    )
    apply_episode_annotation(project, annotation, episode_global_start=0.0, episode_global_end=3.0)
    assert _layer(project, "recovery")["clips"] == []
