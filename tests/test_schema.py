from __future__ import annotations

import pytest

from gemini_annotator.schema import (
    apply_elaboration,
    apply_window_elaboration,
    parse_segmentation,
    parse_window_segmentation,
)


def test_parse_segmentation_happy_path():
    data = {
        "overall_task": "Pick up the fork and place it on the plate",
        "chunks": [
            {"start_sec": 0.0, "end_sec": 2.0, "subtask": "reach for the fork"},
            {"start_sec": 2.0, "end_sec": 4.0, "subtask": "pick up the fork"},
        ],
    }
    task, chunks = parse_segmentation(data, duration=4.0)
    assert task == "Pick up the fork and place it on the plate"
    assert [c.subtask for c in chunks] == ["reach for the fork", "pick up the fork"]
    assert [c.index for c in chunks] == [0, 1]


def test_parse_segmentation_drops_degenerate_chunks():
    data = {
        "overall_task": "task",
        "chunks": [
            {"start_sec": 0.0, "end_sec": 2.0, "subtask": "a"},
            {"start_sec": 2.0, "end_sec": 2.0, "subtask": "degenerate"},
            {"start_sec": 2.0, "end_sec": 4.0, "subtask": "b"},
        ],
    }
    task, chunks = parse_segmentation(data, duration=4.0)
    assert len(chunks) == 2
    assert [c.index for c in chunks] == [0, 1]


def test_parse_segmentation_clamps_to_duration():
    data = {
        "overall_task": "task",
        "chunks": [{"start_sec": 0.0, "end_sec": 999.0, "subtask": "a"}],
    }
    _, chunks = parse_segmentation(data, duration=5.0)
    assert chunks[0].end_sec == 5.0


def test_parse_segmentation_rejects_empty_task():
    with pytest.raises(ValueError):
        parse_segmentation({"overall_task": "", "chunks": [{"start_sec": 0, "end_sec": 1, "subtask": "a"}]}, duration=1.0)


def test_parse_segmentation_rejects_no_chunks():
    with pytest.raises(ValueError):
        parse_segmentation({"overall_task": "task", "chunks": []}, duration=1.0)


def test_apply_elaboration_merges_by_index():
    _, chunks = parse_segmentation(
        {
            "overall_task": "task",
            "chunks": [
                {"start_sec": 0, "end_sec": 2, "subtask": "pick up the fork"},
                {"start_sec": 2, "end_sec": 4, "subtask": "pick the fork back up"},
            ],
        },
        duration=4.0,
    )
    apply_elaboration(chunks, {
        "chunks": [
            {"index": 0, "atomic_actions": ["reach", "close gripper"], "is_recovery": False},
            {
                "index": 1,
                "atomic_actions": ["reach", "close gripper", "lift"],
                "is_recovery": True,
                "recovery_from": "dropped the fork",
                "recovery_action": "pick the fork back up off the table",
            },
        ]
    })
    assert chunks[0].atomic_actions == ["reach", "close gripper"]
    assert chunks[0].is_recovery is False
    assert chunks[0].recovery_action is None
    assert chunks[1].is_recovery is True
    assert chunks[1].recovery_from == "dropped the fork"
    assert chunks[1].recovery_action == "pick the fork back up off the table"


def test_apply_elaboration_clears_recovery_fields_when_not_recovery():
    _, chunks = parse_segmentation(
        {"overall_task": "task", "chunks": [{"start_sec": 0, "end_sec": 2, "subtask": "a"}]},
        duration=2.0,
    )
    chunks[0].recovery_from = "stale"
    chunks[0].recovery_action = "stale"
    apply_elaboration(chunks, {"chunks": [{"index": 0, "atomic_actions": [], "is_recovery": False,
                                            "recovery_from": "should be ignored", "recovery_action": "should be ignored"}]})
    assert chunks[0].recovery_from is None
    assert chunks[0].recovery_action is None


def test_parse_window_segmentation_shifts_times_onto_global_axis():
    data = {
        "episodes": [
            {
                "start_sec": 0.0, "end_sec": 30.0, "overall_task": "pick up the fork",
                "chunks": [{"start_sec": 0.0, "end_sec": 30.0, "subtask": "pick up the fork"}],
            },
            {
                "start_sec": 30.0, "end_sec": 60.0, "overall_task": "pick up the carrot",
                "chunks": [
                    {"start_sec": 30.0, "end_sec": 45.0, "subtask": "reach for the carrot"},
                    {"start_sec": 45.0, "end_sec": 60.0, "subtask": "grasp the carrot"},
                ],
            },
        ]
    }
    episodes = parse_window_segmentation(data, window_start=100.0, window_end=160.0)
    assert [e.start_sec for e in episodes] == [100.0, 130.0]
    assert [e.end_sec for e in episodes] == [130.0, 160.0]
    assert episodes[1].chunks[0].start_sec == 130.0
    assert episodes[1].chunks[1].end_sec == 160.0
    # chunk indices are globally unique within the window (0..N-1 across all episodes)
    assert [c.index for ep in episodes for c in ep.chunks] == [0, 1, 2]


def test_parse_window_segmentation_clamps_to_window_end():
    data = {"episodes": [{
        "start_sec": 0.0, "end_sec": 999.0, "overall_task": "task",
        "chunks": [{"start_sec": 0.0, "end_sec": 999.0, "subtask": "a"}],
    }]}
    episodes = parse_window_segmentation(data, window_start=0.0, window_end=50.0)
    assert episodes[0].end_sec == 50.0
    assert episodes[0].chunks[0].end_sec == 50.0


def test_parse_window_segmentation_drops_degenerate_episodes():
    data = {"episodes": [
        {"start_sec": 0.0, "end_sec": 0.0, "overall_task": "degenerate", "chunks": []},
        {"start_sec": 0.0, "end_sec": 10.0, "overall_task": "real one",
         "chunks": [{"start_sec": 0.0, "end_sec": 10.0, "subtask": "a"}]},
    ]}
    episodes = parse_window_segmentation(data, window_start=0.0, window_end=10.0)
    assert len(episodes) == 1
    assert episodes[0].overall_task == "real one"


def test_parse_window_segmentation_rejects_no_episodes():
    with pytest.raises(ValueError):
        parse_window_segmentation({"episodes": []}, window_start=0.0, window_end=10.0)


def test_apply_window_elaboration_merges_across_episodes_by_global_index():
    data = {
        "episodes": [
            {"start_sec": 0.0, "end_sec": 10.0, "overall_task": "task a",
             "chunks": [{"start_sec": 0.0, "end_sec": 10.0, "subtask": "chunk a"}]},
            {"start_sec": 10.0, "end_sec": 20.0, "overall_task": "task b",
             "chunks": [{"start_sec": 10.0, "end_sec": 20.0, "subtask": "chunk b"}]},
        ]
    }
    episodes = parse_window_segmentation(data, window_start=0.0, window_end=20.0)
    apply_window_elaboration(episodes, {
        "chunks": [
            {"index": 0, "atomic_actions": ["x"], "is_recovery": False},
            {"index": 1, "atomic_actions": ["y"], "is_recovery": True,
             "recovery_from": "dropped it", "recovery_action": "pick it back up"},
        ]
    })
    assert episodes[0].chunks[0].atomic_actions == ["x"]
    assert episodes[0].chunks[0].is_recovery is False
    assert episodes[1].chunks[0].atomic_actions == ["y"]
    assert episodes[1].chunks[0].is_recovery is True
    assert episodes[1].chunks[0].recovery_action == "pick it back up"
