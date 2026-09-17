from __future__ import annotations

import pytest

from gemini_annotator.schema import apply_elaboration, parse_segmentation


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
