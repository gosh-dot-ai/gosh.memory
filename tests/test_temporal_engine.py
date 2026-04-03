"""Focused tests for the temporal engine MVP."""

from src.memory import build_episode_hybrid_context
from src.temporal import lookup_events_for_fact, lookup_ordinal_anchor, lookup_ordinal_range
from src.temporal_normalizer import normalize_temporal_index
from src.temporal_planner import (
    _question_action_intent,
    _question_objective_tokens,
    _select_reducer,
    classify_temporal_query,
    execute_calendar_query,
    execute_ordinal_query,
    extract_calendar_query,
    resolve_calendar_query_interval,
)


def test_temporal_normalizer_extracts_explicit_ordinal_events():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": '[Step 5] Action: execute_bash: {"command":"pytest -q"}',
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 57]},
                "support_fact_ids": ["f_01"],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Turn 6: user asks for the exact command.",
                "timestamp": "2026-03-02",
                "provenance": {"raw_span": [58, 99]},
                "support_fact_ids": ["f_02"],
                "payload": {"episode_id": "SRC_e02"},
            },
        ]
    )

    step_events = lookup_ordinal_anchor(index, kind="step", value=5)
    turn_events = lookup_ordinal_anchor(index, kind="turn", value=6)

    assert len(step_events) == 1
    assert step_events[0]["payload"]["episode_id"] == "SRC_e01"
    assert step_events[0]["payload"]["tool_name"] == "execute_bash"
    assert step_events[0]["payload"]["tool_args"]["command"] == "pytest -q"
    assert len(turn_events) == 1
    assert turn_events[0]["payload"]["episode_id"] == "SRC_e02"


def test_execute_ordinal_query_returns_deterministic_exact_command():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 8]\nAction: execute_snowflake_sql: SELECT * FROM wholesale WHERE year BETWEEN 2020 AND 2023\nObservation: ok",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 113]},
                "support_fact_ids": ["f_08"],
                "payload": {"episode_id": "SRC_e01"},
            }
        ]
    )

    hit = execute_ordinal_query("At step 8, what SQL did the agent run?", index)

    assert hit["matched"] is True
    assert hit["deterministic_answer"] == "SELECT * FROM wholesale WHERE year BETWEEN 2020 AND 2023"
    assert hit["reducer"] == "exact_sql"


def test_execute_ordinal_query_matches_what_was_the_exact_command_form():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 6]\nAction: execute_bash: {\"command\":\"find . -name \\\"*.py\\\" -exec grep -l \\\"MatrixSymbol\\\\|Sum\\\\|refine\\\" {} \\\\;\"}\nObservation: files found",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 154]},
                "support_fact_ids": ["f_06"],
                "payload": {"episode_id": "SRC_e01"},
            }
        ]
    )

    hit = execute_ordinal_query(
        "At step 6, what was the exact command executed by the agent to begin their targeted search?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "exact_command"
    assert hit["deterministic_answer"] == 'find . -name "*.py" -exec grep -l "MatrixSymbol\\|Sum\\|refine" {} \\;'


def test_exact_command_strips_terminal_head_from_search_pipeline():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": (
                    "[Step 6]\n"
                    "Action: execute_bash: {\"command\":\"find . -name \\\"*.py\\\" -exec grep -l \\\"MatrixSymbol\\\\|Sum\\\\|refine\\\" {} \\\\; | head -20\"}\n"
                    "Observation: files found"
                ),
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 166]},
                "support_fact_ids": ["f_06"],
                "payload": {"episode_id": "SRC_e01"},
            }
        ]
    )

    hit = execute_ordinal_query(
        "At step 6, what was the exact command executed by the agent to begin their targeted search?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "exact_command"
    assert hit["deterministic_answer"] == 'find . -name "*.py" -exec grep -l "MatrixSymbol\\|Sum\\|refine" {} \\;'


def test_execute_ordinal_query_can_answer_adjacent_after_step_action():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 1]\nAction: open merge requests\nObservation: list opened",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 61]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 2]\nAction: type color utility into filter box\nObservation: filtered list",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [62, 140]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e02"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "After opening Merge requests at Step 1, which action filtered the list for the target MR?",
        index,
    )

    assert hit["matched"] is True
    assert hit["deterministic_answer"] == "type color utility into filter box"
    assert hit["reducer"] == "exact_action"


def test_temporal_normalizer_imputes_same_step_companion_after_suffix_marker():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e07",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": (
                    "Action: execute_bash: {\"command\":\"find . -name '*.py'\"}\n"
                    "Observation: many files\n"
                    "[Step 7]"
                ),
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 92]},
                "support_fact_ids": ["f_07"],
                "payload": {"episode_id": "SRC_e07"},
            },
            {
                "span_id": "SRC_e08",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": (
                    "Action: execute_bash: {\"command\":\"grep -n \\\"def changelist_view\\\" options.py\"}\n"
                    "Observation: 1914:    def changelist_view"
                ),
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [93, 216]},
                "support_fact_ids": ["f_08"],
                "payload": {"episode_id": "SRC_e08"},
            },
            {
                "span_id": "SRC_e09",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 8]\nAction: execute_bash: {\"command\":\"next\"}",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [217, 268]},
                "support_fact_ids": ["f_09"],
                "payload": {"episode_id": "SRC_e09"},
            },
        ]
    )

    step7_events = lookup_ordinal_anchor(index, kind="step", value=7)
    assert any((event.get("payload") or {}).get("ordinal_imputed") for event in step7_events)

    hit = execute_ordinal_query(
        "At step 7, what exact command did the agent run?",
        index,
    )
    assert hit["deterministic_answer"] == 'grep -n "def changelist_view" options.py'


def test_temporal_normalizer_imputes_missing_steps_between_sparse_markers():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 0]\nAction: open dashboard\nObservation: dashboard",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 54]},
                "support_fact_ids": ["f_01"],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: click merge requests\nObservation: merge requests page",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [55, 118]},
                "support_fact_ids": ["f_02"],
                "payload": {"episode_id": "SRC_e02"},
            },
            {
                "span_id": "SRC_e03",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: type color utility into filter box\nObservation: filtered",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [119, 186]},
                "support_fact_ids": ["f_03"],
                "payload": {"episode_id": "SRC_e03"},
            },
            {
                "span_id": "SRC_e04",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 3]\nAction: open merge request\nObservation: detail page",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [187, 248]},
                "support_fact_ids": ["f_04"],
                "payload": {"episode_id": "SRC_e04"},
            },
        ]
    )

    step1_events = lookup_ordinal_anchor(index, kind="step", value=1)
    step2_events = lookup_ordinal_anchor(index, kind="step", value=2)
    assert any((event.get("payload") or {}).get("ordinal_imputed") for event in step1_events)
    assert any((event.get("payload") or {}).get("ordinal_imputed") for event in step2_events)

    hit = execute_ordinal_query(
        "After opening Merge requests at Step 1, which action filtered the list for the target MR?",
        index,
    )
    assert hit["deterministic_answer"] == "type color utility into filter box"


def test_temporal_normalizer_imputes_same_step_action_after_marker_only_span():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e11",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 5]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 8]},
                "support_fact_ids": ["f_11"],
                "payload": {"episode_id": "SRC_e11"},
            },
            {
                "span_id": "SRC_e12",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: execute_bash: \nObservation: {'error': 'API error: 500 Server Error: Internal Server Error'}",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [9, 110]},
                "support_fact_ids": ["f_12"],
                "payload": {"episode_id": "SRC_e12"},
            },
            {
                "span_id": "SRC_e13",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 6]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [111, 119]},
                "support_fact_ids": ["f_13"],
                "payload": {"episode_id": "SRC_e13"},
            },
            {
                "span_id": "SRC_e14",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: execute_snowflake_sql: \nObservation: {'error': 'API error: 500 Server Error: Internal Server Error'}",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [120, 233]},
                "support_fact_ids": ["f_14"],
                "payload": {"episode_id": "SRC_e14"},
            },
            {
                "span_id": "SRC_e15",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 7]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [234, 242]},
                "support_fact_ids": ["f_15"],
                "payload": {"episode_id": "SRC_e15"},
            },
        ]
    )

    step6_events = lookup_ordinal_anchor(index, kind="step", value=6)
    assert any(
        (event.get("payload") or {}).get("ordinal_imputed")
        and str((event.get("payload") or {}).get("tool_name") or "") == "execute_snowflake_sql"
        for event in step6_events
    )

    hit = execute_ordinal_query(
        "At turn 6, what tool did the agent switch to, and what key input was missing from the call based on the trajectory record?",
        index,
    )

    assert hit["matched"] is True
    assert hit["resolved_kind"] == "step"
    assert hit["reducer"] == "tool_missing_input"
    assert hit["deterministic_answer"] == (
        "The agent switched to execute_snowflake_sql; the required sql input was missing."
    )


def test_temporal_normalizer_imputes_pre_marker_action_to_previous_step():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e14",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 12]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 9]},
                "support_fact_ids": ["f_14"],
                "payload": {"episode_id": "SRC_e14"},
            },
            {
                "span_id": "SRC_e15",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": (
                    "Action: execute_bash: {\"command\":\"grep -n \\\"def cla\\\" lib/matplotlib/axes/_base.py\"}\n"
                    "Observation: 1182:    def cla(self):\n"
                    "[Step 13]"
                ),
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [10, 146]},
                "support_fact_ids": ["f_15"],
                "payload": {"episode_id": "SRC_e15"},
            },
        ]
    )

    step12_events = lookup_ordinal_anchor(index, kind="step", value=12)
    assert any(
        (event.get("payload") or {}).get("ordinal_imputed") and "grep -n" in str((event.get("payload") or {}).get("action_raw") or "")
        for event in step12_events
    )

    hit = execute_ordinal_query(
        "At step 12, what exact grep command did the agent execute to find the location of the cla() method definition?",
        index,
    )
    assert hit["matched"] is True
    assert hit["reducer"] == "exact_command"
    assert hit["deterministic_answer"] == 'grep -n "def cla" lib/matplotlib/axes/_base.py'


def test_temporal_normalizer_imputes_pre_marker_result_to_previous_step():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e20",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 10]\nAction: execute_snowflake_sql: SELECT project_short_name, COUNT(*) AS cnt FROM segment WHERE project_short_name = 'TCGA-KIRC'\nObservation: Query executed successfully",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 175]},
                "support_fact_ids": ["f_20"],
                "payload": {"episode_id": "SRC_e20"},
            },
            {
                "span_id": "SRC_e21",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "```csv\nproject_short_name,CNT\nTCGA-KIRC,85084\n```\n\n[Step 11]\nAction: execute_snowflake_sql: SELECT COUNT(*) FROM segment\nObservation: Query executed successfully",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [176, 338]},
                "support_fact_ids": ["f_21"],
                "payload": {"episode_id": "SRC_e21"},
            },
        ]
    )

    step10_events = lookup_ordinal_anchor(index, kind="step", value=10)
    assert any((event.get("payload") or {}).get("result_companion") for event in step10_events)

    hit = execute_ordinal_query(
        "At step 10, what exact SQL action did the agent execute to validate that TCGA-KIRC records exist in the segment table, and what numeric result did it observe?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "exact_sql_result"
    assert "SELECT project_short_name, COUNT(*) AS cnt FROM segment WHERE project_short_name = 'TCGA-KIRC'" in hit["deterministic_answer"]
    assert "Observed result: TCGA-KIRC, 85084." in hit["deterministic_answer"]


def test_after_step_uses_best_event_of_next_ordinal_step_not_empty_marker():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Observation: dashboard [Step 1]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 29]},
                "support_fact_ids": ["f_01"],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: click merge requests\nObservation: merge requests page",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [30, 93]},
                "support_fact_ids": ["f_02"],
                "payload": {"episode_id": "SRC_e02"},
            },
            {
                "span_id": "SRC_e03",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Observation: merge requests list [Step 2]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [94, 134]},
                "support_fact_ids": ["f_03"],
                "payload": {"episode_id": "SRC_e03"},
            },
            {
                "span_id": "SRC_e04",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: type color utility into filter box\nObservation: filtered",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [135, 202]},
                "support_fact_ids": ["f_04"],
                "payload": {"episode_id": "SRC_e04"},
            },
            {
                "span_id": "SRC_e05",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 3]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [203, 211]},
                "support_fact_ids": ["f_05"],
                "payload": {"episode_id": "SRC_e05"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "After opening Merge requests at Step 1, which action filtered the list for the target MR?",
        index,
    )
    assert hit["matched"] is True
    assert hit["reducer"] == "exact_action"
    assert hit["deterministic_answer"] == "type color utility into filter box"


def test_exact_status_prefers_observation_error_over_later_step_status():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e30",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 3]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 8]},
                "support_fact_ids": ["f_30"],
                "payload": {"episode_id": "SRC_e30"},
            },
            {
                "span_id": "SRC_e31",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: execute_code: {\"code\":\"print(1)\"}\nObservation: requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://example.com/api",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [9, 160]},
                "support_fact_ids": ["f_31"],
                "payload": {"episode_id": "SRC_e31"},
            },
            {
                "span_id": "SRC_e32",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 4]\nAction: mark_step: {\"step_status\":\"completed\"}",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [161, 215]},
                "support_fact_ids": ["f_32"],
                "payload": {"episode_id": "SRC_e32"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "What was the specific result status of the action taken at turn 3?",
        index,
    )

    assert hit["matched"] is True
    assert hit["resolved_kind"] == "step"
    assert hit["reducer"] == "exact_status"
    assert hit["deterministic_answer"] == "Execution failed with a 403 Client Error (Forbidden)."


def test_first_occurrence_reducer_returns_earliest_matching_step_and_action():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 1]\nAction: scroll [down]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 30]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 2]\nAction: click [482] where [482] is link 'Find directions between two points'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [31, 114]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e02"},
            },
        ]
    )

    hit = execute_ordinal_query("Which step first opened the directions interface?", index)

    assert hit["matched"] is True
    assert hit["reducer"] == "first_occurrence"
    assert hit["deterministic_answer"] == "Step 2, click link 'Find directions between two points'."


def test_first_occurrence_respects_explicit_state_constraint():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e00",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 0]\nAction: click [394] where [394] is button '' hasPopup: menu expanded: False",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 88]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e00"},
            },
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 1]\nAction: scroll [down]\nObservation: button '' hasPopup: menu expanded: True",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [89, 177]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 2]\nAction: click [394] where [394] is button '' hasPopup: menu expanded: True",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [178, 265]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e02"},
            },
        ]
    )

    hit = execute_ordinal_query("Which step first opened the profile menu (expanded: True)?", index)

    assert hit["matched"] is True
    assert hit["reducer"] == "first_occurrence"
    assert hit["deterministic_answer"] == "Step 2, click [394] where [394] is button '' hasPopup: menu expanded: True."


def test_consecutive_before_scroll_reducer_counts_matching_click_run():
    index = normalize_temporal_index(
        [
            {
                "span_id": f"SRC_e0{i}",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": text,
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [i * 10, i * 10 + len(text)]},
                "support_fact_ids": [],
                "payload": {"episode_id": f"SRC_e0{i}"},
            }
            for i, text in enumerate(
                [
                    "[Step 1]\nAction: goto [http://example.test]",
                    "[Step 2]\nAction: click [1960] where [1960] is link 'Submissions'",
                    "[Step 3]\nAction: click [2195] where [2195] is link 'Submissions'",
                    "[Step 4]\nAction: click [2430] where [2430] is link 'Submissions'",
                    "[Step 5]\nAction: click [2665] where [2665] is link 'Submissions'",
                    "[Step 6]\nAction: scroll [down]",
                ],
                start=1,
            )
        ]
    )

    hit = execute_ordinal_query(
        "How many consecutive 'Submissions' link clicks occurred before the first scroll action?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "consecutive_before_scroll"
    assert hit["deterministic_answer"] == "4 consecutive clicks (Steps 2-5)."


def test_range_environment_changes_returns_transfer_deltas_only():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 1]\nAction: go to sofa 1",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 29]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 2]\nAction: take keychain 4 from sofa 1",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [30, 76]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e02"},
            },
            {
                "span_id": "SRC_e03",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 3]\nAction: go to safe 1",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [77, 106]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e03"},
            },
            {
                "span_id": "SRC_e04",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 4]\nAction: open safe 1",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [107, 135]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e04"},
            },
            {
                "span_id": "SRC_e05",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 5]\nAction: move keychain 4 to safe 1",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [136, 179]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e05"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "From step 1 to step 5, what actions made the environment changes and what were the environment changes?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "environment_changes"
    assert hit["deterministic_answer"] == (
        "Action-environment changes: "
        "step 2: 'take keychain 4 from sofa 1' caused [keychain 4 added to inventory; keychain 4 moved from sofa 1 to inventory] | "
        "step 5: 'move keychain 4 to safe 1' caused [keychain 4 removed from inventory; keychain 4 moved from inventory to safe 1]"
    )


def test_range_actions_between_boundary_steps_uses_interior_actions_only():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e03",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 3]\nAction: click [7026] where [7026] is link 'Submit'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 57]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e03"},
            },
            {
                "span_id": "SRC_e04",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 4]\nAction: type [7043] [Title text] where [7043] is textbox 'Title *' required: True",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [58, 151]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e04"},
            },
            {
                "span_id": "SRC_e05",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 5]\nAction: type [7053] [Body text] where [7053] is textbox 'Body' required: False",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [152, 240]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e05"},
            },
            {
                "span_id": "SRC_e06",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 6]\nAction: click [7026] where [7026] is link 'Submit'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [241, 298]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e06"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "Between the first Submit click (Step 3) and the second Submit click (Step 6), which two actions occurred?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "range_actions"
    assert hit["deterministic_answer"] == (
        "Between step 3 and step 6, the agent performed the following actions: "
        "at step 4, type 'Title text' into textbox 'Title *'; "
        "at step 5, type 'Body text' into textbox 'Body'."
    )


def test_exact_grid_swap_match_explains_moved_token_and_match_coords():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e12",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": (
                    "[Step 12]\n"
                    "Action: ((-1, -1), (-1, -1))\n"
                    "Observation: Board:\n\n"
                    "0| G R R P G P R C\n"
                    "1| R P C P C C R G\n"
                    "2| R C G R C C P P\n"
                    "3| C R G R G P C C\n"
                    "4| R G P G C P G G\n"
                    "5| C P G R R G P C"
                ),
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 190]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e12"},
            },
            {
                "span_id": "SRC_e13",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 13]\nAction: ((4,3),(5,3))",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [191, 223]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e13"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "In step 13, by examining the board state at step 12, which specific candy was moved and what coordinates formed the match?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "grid_swap_match"
    assert hit["deterministic_answer"] == (
        "The 'R' candy from (5,3) was moved into (4,3), swapping with the 'G' candy that had been there. "
        "This created a match at coordinates (2,3), (3,3), (4,3)."
    )


def test_range_checkbox_order_returns_clicked_ids_in_order():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e08",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 8]\nAction: click [3742] where [3742] is button 'Add Attribute'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 70]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e08"},
            },
            {
                "span_id": "SRC_e09",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 9]\nAction: click [6986] where [6986] is checkbox '' checked: false",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [71, 146]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e09"},
            },
            {
                "span_id": "SRC_e10",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 10]\nAction: click [7030] where [7030] is checkbox '' checked: false",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [147, 223]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e10"},
            },
            {
                "span_id": "SRC_e11",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 11]\nAction: scroll [down]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [224, 255]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e11"},
            },
            {
                "span_id": "SRC_e12",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 12]\nAction: click [3655] where [3655] is button 'Cancel'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [256, 322]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e12"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "Between the first Add Attribute click (Step 8) and Cancel (Step 12), which two attribute checkboxes were selected and in what order?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "checkbox_order"
    assert hit["deterministic_answer"] == "Checkbox [6986] was clicked first, then checkbox [7030]."


def test_invalid_action_range_reducer_finds_first_malformed_step_and_restored_navigation():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 1]\nAction: click [2040] where [2040] is link 'Most stars'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 67]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 2]\nAction: The previous prediction you issued was \"Let's think step-by-step...\"",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [68, 155]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e02"},
            },
            {
                "span_id": "SRC_e03",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 3]\nAction: click [3947] where [3947] is link 'Most stars'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [156, 223]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e03"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "Between Step 1 and Step 3, the agent issued an invalid action. Identify the step and the formatting error, and state which step restored a valid navigation.",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "invalid_action_window"
    assert hit["deterministic_answer"] == (
        "Step 2 was the invalid action: it used the wrong format by placing long reasoning text in the action field instead of a valid command/action, so no state change occurred. "
        "Step 3 restored valid navigation by clicking 'Most stars' [3947]."
    )


def test_first_malformed_before_anchor_finds_earliest_bad_step():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e10",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 10]\nAction: click [29383] where [29383] is link 'Target product'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 75]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e10"},
            },
            {
                "span_id": "SRC_e11",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 11]\nAction: The previous prediction you issued was \"Let's think step-by-step. ...\"",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [76, 168]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e11"},
            },
            {
                "span_id": "SRC_e12",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 12]\nAction: The previous prediction you issued was \"Let's think step by step. ...\"",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [169, 260]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e12"},
            },
            {
                "span_id": "SRC_e14",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 14]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [261, 269]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e14"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "The agent's trajectory ends at Step 14 due to 'Failed to parse actions.' Trace the trajectory back: Identify the specific Step Index where the agent first made the formatting choice that led to this failure, and explain how it violated the system's input specifications.",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "first_malformed_before_anchor"
    assert hit["deterministic_answer"] == (
        "Step 11. The agent put long reasoning text into the action field instead of a valid command/action."
    )


def test_at_step_immediately_after_phrase_does_not_force_next_step_shift():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e35",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 35]\nAction: think: {\"thought\":\"Issue still exists\"}\nObservation: logged",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 79]},
                "support_fact_ids": ["f_35"],
                "payload": {"episode_id": "SRC_e35"},
            },
            {
                "span_id": "SRC_e36",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 36]\nAction: str_replace_editor: {\"command\":\"str_replace\",\"path\":\"file.py\"}\nObservation: edited",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [80, 184]},
                "support_fact_ids": ["f_36"],
                "payload": {"episode_id": "SRC_e36"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "At step 35, what exact action was taken immediately after confirming that the original issue still existed and before implementing any fix?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "exact_action"
    assert hit["deterministic_answer"] == 'think: {"thought":"Issue still exists"}'


def test_exact_action_can_select_better_objective_match_from_next_local_step():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e07",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: execute_bash: {\"command\":\"find . -name \\\"*.py\\\" | head\"}\nObservation: options.py\n[Step 7]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 97]},
                "support_fact_ids": ["f_07"],
                "payload": {"episode_id": "SRC_e07"},
            },
            {
                "span_id": "SRC_e08",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: str_replace_editor: {\"command\":\"view\",\"path\":\"options.py\"}\nObservation: file contents",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [98, 189]},
                "support_fact_ids": ["f_08"],
                "payload": {"episode_id": "SRC_e08"},
            },
            {
                "span_id": "SRC_e09",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 8]\nAction: execute_bash: {\"command\":\"grep -n \\\"target_symbol\\\" options.py\"}\nObservation: 1914: def target_symbol(self)\n[Step 9]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [190, 332]},
                "support_fact_ids": ["f_09"],
                "payload": {"episode_id": "SRC_e09"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "At step 7, what exact action did the agent execute to identify the target_symbol method location?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "exact_action"
    assert hit["deterministic_answer"] == 'grep -n "target_symbol" options.py'


def test_exact_action_local_repair_prefers_search_over_later_view_for_location_queries():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e07",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Observation: options.py [Step 7]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 30]},
                "support_fact_ids": ["f_07"],
                "payload": {"episode_id": "SRC_e07"},
            },
            {
                "span_id": "SRC_e08",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": (
                    "[Step 8]\n"
                    "Action: execute_bash: {\"command\":\"grep -n \\\"def changelist_view\\\" options.py\"}\n"
                    "Observation: 1914: def changelist_view"
                ),
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [31, 154]},
                "support_fact_ids": ["f_08"],
                "payload": {"episode_id": "SRC_e08"},
            },
            {
                "span_id": "SRC_e09",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": (
                    "[Step 9]\n"
                    "Action: str_replace_editor: {\"command\":\"view\",\"path\":\"options.py\",\"view_range\":[1910,1970]}\n"
                    "Observation: def changelist_view(self, request)"
                ),
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [155, 310]},
                "support_fact_ids": ["f_09"],
                "payload": {"episode_id": "SRC_e09"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "At step 7, what exact action did the agent execute to identify the changelist_view method location?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "exact_action"
    assert hit["deterministic_answer"] == 'grep -n "def changelist_view" options.py'


def test_exact_action_does_not_jump_to_distant_same_source_event_without_local_match():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e28",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 28]\nAction: str_replace_editor: {\"command\":\"str_replace\",\"path\":\"test_reproduction.py\"}\nObservation: edited",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 110]},
                "support_fact_ids": ["f_28"],
                "payload": {"episode_id": "SRC_e28"},
            },
            {
                "span_id": "SRC_e34",
                "source_id": "SRC",
                "timeline_id": "conversation:SRC",
                "text": "[Step 34]\nAction: str_replace_editor: {\"command\":\"str_replace\",\"path\":\"django/core/management/base.py\",\"old_str\":\"parser.add_argument()\",\"new_str\":\"parser.add_argument('--skip-checks')\"}",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [111, 315]},
                "support_fact_ids": ["f_34"],
                "payload": {"episode_id": "SRC_e34"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "At step 28, what exact action did the agent execute to modify the BaseCommand class?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "exact_action"
    assert hit["primary_event"]["event_id"] == "SRC_e28:step:28:0"
    assert hit["deterministic_answer"] == "Used str_replace_editor to modify test_reproduction.py"


def test_exact_action_canonicalizes_ui_click_and_type_patterns():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 8]\nAction: type [6157] [color utility ] where [6157] is textbox 'Search or filter results...' required: False",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 112]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 9]\nAction: click [14278] where [14278] is link 'Edit'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [113, 175]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e02"},
            },
        ]
    )

    hit_type = execute_ordinal_query("At step 8, what exact action was taken?", index)
    hit_click = execute_ordinal_query("At step 9, what exact action was taken?", index)

    assert hit_type["deterministic_answer"] == "type 'color utility' into textbox 'Search or filter results...'"
    assert hit_click["deterministic_answer"] == "click link 'Edit'"


def test_exact_action_objective_rescue_prefers_local_view_match_and_formats_editor_action():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e58",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 58]\nAction: str_replace_editor: {\"command\":\"view\",\"path\":\"django/db/models/sql/query.py\",\"view_range\":[1885,1920]}",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 121]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e58"},
            },
            {
                "span_id": "SRC_e59",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 59]\nAction: execute_bash: {\"command\":\"grep -n _check_ordering django/db/models/base.py\"}",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [122, 218]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e59"},
            },
            {
                "span_id": "SRC_e60",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 60]\nAction: str_replace_editor: {\"command\":\"view\",\"path\":\"django/db/models/options.py\",\"view_range\":[1900,1950]}\nObservation: def _check_ordering(self)",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [219, 381]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e60"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "At step 58, what exact action did the agent execute to examine the _check_ordering method?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "exact_action"
    assert hit["deterministic_answer"] == (
        "Used str_replace_editor to view lines 1900-1950 of django/db/models/options.py"
    )


def test_exact_action_formats_str_replace_delta_as_added_literal():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e33",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 33]\nAction: str_replace_editor: {\"command\":\"str_replace\",\"path\":\"/workspace/django/core/management/base.py\",\"old_str\":\"show_last = {'--force-color'}\",\"new_str\":\"show_last = {'--force-color', '--skip-checks'}\"}",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 227]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e33"},
            }
        ]
    )

    hit = execute_ordinal_query(
        "At step 33, what exact action did the agent execute to modify the BaseCommand class?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "exact_action"
    assert hit["deterministic_answer"] == "Added '--skip-checks' to /workspace/django/core/management/base.py"


def test_question_objective_tokens_split_camel_case_symbols():
    tokens = _question_objective_tokens(
        "At step 33, what exact action did the agent execute to modify the BaseCommand class?"
    )

    assert "basecommand" in tokens
    assert "base" in tokens


def test_question_action_intent_prefers_view_over_edit_when_query_explicitly_requests_view():
    reducer = _select_reducer("At step 73, what exact action did the agent execute to view the modified code?")

    assert reducer == "exact_action"
    assert _question_action_intent(
        "At step 73, what exact action did the agent execute to view the modified code?",
        reducer,
    ) == "view"


def test_select_reducer_treats_specific_action_as_exact_action():
    assert _select_reducer(
        "At step 15, what specific action did the agent execute to locate the problematic code mentioned in the issue description?"
    ) == "exact_action"


def test_question_action_intent_does_not_force_search_for_generic_locate_wording():
    reducer = _select_reducer(
        "At step 15, what specific action did the agent execute to locate the problematic code mentioned in the issue description?"
    )

    assert reducer == "exact_action"
    assert _question_action_intent(
        "At step 15, what specific action did the agent execute to locate the problematic code mentioned in the issue description?",
        reducer,
    ) is None


def test_relative_distance_counts_first_matching_future_step():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Observation: review list [Step 3]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 32]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: click [2409] where [2409] is link 'Edit'\nObservation: edit page",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [33, 107]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e02"},
            },
            {
                "span_id": "SRC_e03",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Observation: status form [Step 4]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [108, 141]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e03"},
            },
            {
                "span_id": "SRC_e04",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: click [3429] where [3429] is combobox 'Status *'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [142, 201]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e04"},
            },
            {
                "span_id": "SRC_e05",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Observation: status open [Step 5]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [202, 234]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e05"},
            },
            {
                "span_id": "SRC_e06",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Action: press [ArrowDown]\nObservation: focus moved",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [235, 286]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e06"},
            },
            {
                "span_id": "SRC_e07",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "Observation: moved [Step 6]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [287, 313]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e07"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "After clicking Edit at Step 3, how many steps later did the first ArrowDown occur?",
        index,
    )
    assert hit["reducer"] == "relative_distance"
    assert hit["deterministic_answer"] == "2 steps later (Step 5)."


def test_semantic_relative_distance_without_numeric_anchor():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e01",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 1]\nAction: type [801] [technology] where [801] is textbox 'Search'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 72]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e01"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 2]\nAction: click link 'GitHub'\nObservation: GitHub opened from Postmill",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [73, 152]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e02"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "How many steps after typing 'technology' did the agent leave Postmill for GitHub?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "relative_distance"
    assert hit["deterministic_answer"] == "1 step later."


def test_semantic_relative_distance_prefers_action_match_over_observation_mentions():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e00",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 0]\nObservation: filters shown include $0.00-$99.99",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 56]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e00"},
            },
            {
                "span_id": "SRC_e03",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 3]\nAction: click '$0.00-$99.99' filter\nObservation: filtered results",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [57, 134]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e03"},
            },
            {
                "span_id": "SRC_e04",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 4]\nAction: scroll down\nObservation: more results",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [135, 190]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e04"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "How many steps after applying the $0.00-$99.99 filter did the first scroll occur?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "relative_distance"
    assert hit["deterministic_answer"] == "1 step later."


def test_semantic_relative_distance_can_use_support_facts_and_substantive_steps():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e00",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 0]\nAction: type [64] [earthporn] where [64] is searchbox 'Search query'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 76]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e00"},
            },
            {
                "span_id": "SRC_e02",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 2]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [77, 85]},
                "support_fact_ids": ["f_02"],
                "payload": {
                    "episode_id": "SRC_e02",
                    "support_texts": ["Agent typed search query 'author:CameronKelsey subreddit:EarthPorn' into the Search field"],
                },
            },
            {
                "span_id": "SRC_e03",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 3]\nAction: scroll [down]\nObservation: search",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [86, 137]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e03"},
            },
            {
                "span_id": "SRC_e04",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 4]\nAction: attempt to perfom \"type\" on element \"[164]\" but no matching element found.",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [138, 231]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e04"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "How many steps after the advanced query was typed did the first missing-element error occur?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "relative_distance"
    assert hit["deterministic_answer"] == "2 steps later."


def test_loop_window_reducer_expands_contiguous_module_window_and_returns_ids():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e13",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 7]\nObservation: Products / Inventory / Catalog / Magento Admin\n[Step 8]",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 80]},
                "support_fact_ids": [],
                "payload": {
                    "episode_id": "SRC_e13",
                    "support_texts": ["Current page title is Products / Inventory / Catalog / Magento Admin"],
                },
            },
            {
                "span_id": "SRC_e21",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 10]\nObservation: Customers / Customers / Magento Admin",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [81, 143]},
                "support_fact_ids": [],
                "payload": {
                    "episode_id": "SRC_e21",
                    "support_texts": ["There is a button labeled Add New Customer"],
                },
            },
            {
                "span_id": "SRC_e27",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 12]\nAction: click [29575] where [29575] is link 'All Customers'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [144, 214]},
                "support_fact_ids": [],
                "payload": {
                    "episode_id": "SRC_e27",
                    "support_texts": ["Current page is Products / Inventory / Catalog / Magento Admin"],
                },
            },
            {
                "span_id": "SRC_e35",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 14]\nAction: click [34203] where [34203] is link 'Products'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [215, 278]},
                "support_fact_ids": [],
                "payload": {
                    "episode_id": "SRC_e35",
                    "support_texts": ["Current page is Customers / Customers / Magento Admin"],
                },
            },
            {
                "span_id": "SRC_e51",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 19]\nAction: click [42021] where [42021] is link 'CUSTOMERS'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [279, 346]},
                "support_fact_ids": [],
                "payload": {
                    "episode_id": "SRC_e51",
                    "support_texts": ["Current page is Products / Inventory / Catalog / Magento Admin"],
                },
            },
            {
                "span_id": "SRC_e52",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 20]\nAction: click [44056] where [44056] is link 'MARKETING'",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [347, 414]},
                "support_fact_ids": [],
                "payload": {
                    "episode_id": "SRC_e52",
                    "support_texts": ["Current page is Marketing"],
                },
            },
        ]
    )

    hit = execute_ordinal_query(
        "Identify the precise start and end step indices of the redundant navigation loop where the agent oscillated between the 'Customers' and 'Catalog' modules. Furthermore, cite the specific element ID of the 'All Customers' link clicked in Step 12 and the 'Products' link clicked in Step 14.",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "loop_window"
    assert hit["deterministic_answer"] == (
        "The loop occurs between Step 8 and Step 19. "
        "The 'All Customers' link clicked in Step 12 has the element ID [29575]. "
        "The 'Products' link clicked in Step 14 has the element ID [34203]."
    )


def test_temporal_planner_requires_numeric_ordinal_anchor():
    assert classify_temporal_query("At step 27, what exact command was run?") == "ordinal"
    assert classify_temporal_query("Between step 7 and 9, what changed?") == "ordinal"
    assert classify_temporal_query("At the step where it first opened the file, what happened?") == "semantic"
    assert classify_temporal_query("Who did I meet last week?") == "calendar"
    assert classify_temporal_query("Who did Maria have dinner with on May 3, 2023?") == "calendar"


def test_temporal_planner_calendar_seeking_is_strict():
    assert classify_temporal_query("When did Joanna first watch that movie?") == "calendar"
    assert classify_temporal_query("Which year did Audrey adopt her dogs?") == "calendar"
    assert classify_temporal_query("How old was James when he started?") == "semantic"
    plan = extract_calendar_query("Which year did Audrey adopt her dogs?")
    assert plan["mode"] == "seeking"
    assert plan["granularity"] == "year"
    assert plan["content_query"] == "Audrey adopt her dogs"


def test_temporal_normalizer_resolves_relative_years_from_human_session_date():
    index = normalize_temporal_index(
        [
            {
                "span_id": "fact:s1_f_23",
                "source_id": "SRC",
                "timeline_id": "timeline:SRC:main",
                "text": "I first watched the movie around 3 years ago.",
                "timestamp": "7:31 pm on 21 January, 2022",
                "provenance": {"start_char": 0, "end_char": 44, "source_field": "raw_text", "episode_id": "SRC_e01"},
                "support_fact_ids": ["s1_f_23"],
                "payload": {"episode_id": "SRC_e01", "fact_id": "s1_f_23"},
            }
        ]
    )

    events = lookup_events_for_fact(index, fact_id="s1_f_23")
    assert len(events) == 1
    assert events[0]["time_start"] == "2019-01-22"
    assert events[0]["time_granularity"] == "year"
    assert isinstance(events[0]["time_sort_start"], int)
    assert events[0]["time_sort_start"] == events[0]["time_sort_end"]


def test_temporal_normalizer_keeps_deep_history_relative_years_without_crashing():
    index = normalize_temporal_index(
        [
            {
                "span_id": "fact:s1_f_99",
                "source_id": "SRC",
                "timeline_id": "timeline:SRC:main",
                "text": "The museum traces human settlement here to some 9000 years ago.",
                "timestamp": "2026-01-01",
                "provenance": {"start_char": 0, "end_char": 66, "source_field": "raw_text", "episode_id": "SRC_e99"},
                "support_fact_ids": ["s1_f_99"],
                "payload": {"episode_id": "SRC_e99", "fact_id": "s1_f_99"},
            }
        ]
    )

    events = lookup_events_for_fact(index, fact_id="s1_f_99")
    assert len(events) == 1
    assert events[0]["time_start"] == "-6970-12-24"
    assert events[0]["time_granularity"] == "year"
    assert isinstance(events[0]["time_sort_start"], int)
    assert events[0]["time_sort_start"] == events[0]["time_sort_end"]


def test_temporal_runtime_pins_exact_step_episode_inside_selected_source():
    corpus = {
        "documents": [
            {
                "doc_id": "document:SRC",
                "episodes": [
                    {
                        "episode_id": "SRC_e01",
                        "source_type": "document",
                        "source_id": "SRC",
                        "source_date": "2026-03-01",
                        "topic_key": "step 5 execution",
                        "state_label": "trace",
                        "currentness": "current",
                        "raw_text": '[Step 5] Action: execute_bash: {"command":"pytest -q tests/test_issue.py"}',
                        "provenance": {"raw_span": [0, 74]},
                    },
                    {
                        "episode_id": "SRC_e02",
                        "source_type": "document",
                        "source_id": "SRC",
                        "source_date": "2026-03-01",
                        "topic_key": "later validation",
                        "state_label": "trace",
                        "currentness": "current",
                        "raw_text": '[Step 51] Action: execute_bash: {"command":"pytest -q tests/test_comprehensive.py"}',
                        "provenance": {"raw_span": [75, 160]},
                    },
                ],
            }
        ]
    }
    facts = [
        {
            "id": "f_late",
            "session": 51,
            "fact": "At step 51 the agent executed pytest -q tests/test_comprehensive.py.",
            "metadata": {"episode_id": "SRC_e02", "episode_source_id": "SRC"},
        }
    ]
    index = normalize_temporal_index(
        [
            {
                "span_id": ep["episode_id"],
                "source_id": ep["source_id"],
                "timeline_id": "document:SRC",
                "text": ep["raw_text"],
                "timestamp": ep["source_date"],
                "provenance": ep["provenance"],
                "support_fact_ids": [],
                "payload": {"episode_id": ep["episode_id"]},
            }
            for ep in corpus["documents"][0]["episodes"]
        ]
    )

    packet = build_episode_hybrid_context(
        "At step 5, what exact command was executed?",
        corpus,
        facts,
        search_family="document",
        temporal_index=index,
    )

    assert packet["retrieved_episode_ids"][0] == "SRC_e01"
    assert packet["temporal_trace"]["pinned_episode_ids"] == ["SRC_e01"]
    assert "pytest -q tests/test_issue.py" in packet["context"]


def test_temporal_runtime_narrows_shared_exact_step_query_to_single_source():
    corpus = {
        "documents": [
            {
                "doc_id": "document:SRC_A",
                "episodes": [
                    {
                        "episode_id": "SRC_A_e01",
                        "source_type": "document",
                        "source_id": "SRC_A",
                        "source_date": "2026-03-01",
                        "topic_key": "step 5 execution",
                        "state_label": "trace",
                        "currentness": "current",
                        "raw_text": '[Step 5] Action: execute_bash: {"command":"pytest -q tests/test_issue.py"}',
                        "provenance": {"raw_span": [0, 74]},
                    }
                ],
            },
            {
                "doc_id": "document:SRC_B",
                "episodes": [
                    {
                        "episode_id": "SRC_B_e01",
                        "source_type": "document",
                        "source_id": "SRC_B",
                        "source_date": "2026-03-01",
                        "topic_key": "step 5 profile",
                        "state_label": "trace",
                        "currentness": "current",
                        "raw_text": "[Step 5] Action: open profile menu",
                        "provenance": {"raw_span": [0, 35]},
                    }
                ],
            },
        ]
    }
    facts = [
        {
            "id": "f_a",
            "session": 5,
            "fact": "At step 5 the agent executed pytest -q tests/test_issue.py.",
            "metadata": {"episode_id": "SRC_A_e01", "episode_source_id": "SRC_A"},
        },
        {
            "id": "f_b",
            "session": 5,
            "fact": "At step 5 the agent opened the profile menu.",
            "metadata": {"episode_id": "SRC_B_e01", "episode_source_id": "SRC_B"},
        },
    ]
    index = normalize_temporal_index(
        [
            {
                "span_id": ep["episode_id"],
                "source_id": ep["source_id"],
                "timeline_id": doc["doc_id"],
                "text": ep["raw_text"],
                "timestamp": ep["source_date"],
                "provenance": ep["provenance"],
                "support_fact_ids": [],
                "payload": {"episode_id": ep["episode_id"]},
            }
            for doc in corpus["documents"]
            for ep in doc["episodes"]
        ]
    )

    packet = build_episode_hybrid_context(
        "At step 5, what exact command was executed?",
        corpus,
        facts,
        search_family="document",
        temporal_index=index,
    )

    assert packet["temporal_trace"]["fallback"] is False
    assert packet["temporal_trace"]["scope_trace"]["selected_source_ids"] == ["SRC_A"]
    assert packet["retrieved_episode_ids"][0] == "SRC_A_e01"
    assert "pytest -q tests/test_issue.py" in packet["context"]


def test_temporal_range_lookup_returns_contiguous_events():
    index = normalize_temporal_index(
        [
            {
                "span_id": f"SRC_e0{i}",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": f"[Step {i}] Action: noop",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [i * 10, i * 10 + 9]},
                "support_fact_ids": [],
                "payload": {"episode_id": f"SRC_e0{i}"},
            }
            for i in range(3, 7)
        ]
    )

    events = lookup_ordinal_range(index, kind="step", start=4, end=6)
    assert [event["ordinal_index"] for event in events] == [4, 5, 6]


def test_execute_ordinal_query_can_answer_range_action_listing():
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e45",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 45]\nAction: go to drawer 14",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 34]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e45"},
            },
            {
                "span_id": "SRC_e46",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 46]\nAction: open drawer 14",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [35, 70]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e46"},
            },
            {
                "span_id": "SRC_e47",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": "[Step 47]\nAction: close drawer 14",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [71, 107]},
                "support_fact_ids": [],
                "payload": {"episode_id": "SRC_e47"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "What actions were performed between step 45 and step 47?",
        index,
    )

    assert hit["matched"] is True
    assert hit["reducer"] == "range_actions"
    assert hit["deterministic_answer"] == (
        "Between step 45 and step 47, the agent performed the following actions: "
        "at step 45, go to drawer 14; at step 46, open drawer 14; at step 47, close drawer 14."
    )


def test_execute_ordinal_query_does_not_truncate_long_ranges_before_reduction():
    index = normalize_temporal_index(
        [
            {
                "span_id": f"SRC_e{i:02d}",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": f"[Step {i}]\nAction: action {i}",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [i * 20, i * 20 + 19]},
                "support_fact_ids": [],
                "payload": {"episode_id": f"SRC_e{i:02d}"},
            }
            for i in range(1, 21)
        ]
    )

    hit = execute_ordinal_query(
        "Between step 1 and step 20, what actions were performed?",
        index,
        limit=8,
    )

    assert hit["matched"] is True
    assert hit["resolved"] is True
    assert len(hit["events"]) == 20
    assert "at step 20, action 20" in (hit["deterministic_answer"] or "").lower()


def test_execute_ordinal_query_scopes_first_occurrence_to_requested_source():
    index = normalize_temporal_index(
        [
            {
                "span_id": "A_e05",
                "source_id": "conv-a",
                "timeline_id": "conversation:conv-a",
                "text": "[Step 5]\nAction: open profile menu",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 32]},
                "support_fact_ids": [],
                "payload": {"episode_id": "A_e05"},
            },
            {
                "span_id": "B_e02",
                "source_id": "conv-b",
                "timeline_id": "conversation:conv-b",
                "text": "[Step 2]\nAction: open profile menu",
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [33, 65]},
                "support_fact_ids": [],
                "payload": {"episode_id": "B_e02"},
            },
        ]
    )

    hit = execute_ordinal_query(
        "Which step first opened the profile menu?",
        index,
        source_ids={"conv-a"},
    )

    assert hit["matched"] is True
    assert hit["resolved"] is True
    assert hit["primary_event"]["source_id"] == "conv-a"
    assert hit["primary_event"]["ordinal_index"] == 5
    assert hit["deterministic_answer"] == "Step 5, open profile menu."


def test_temporal_runtime_falls_back_when_ordinal_executor_cannot_reduce_question():
    corpus = {
        "documents": [
            {
                "doc_id": "document:SRC",
                "episodes": [
                    {
                        "episode_id": "SRC_e08",
                        "source_type": "document",
                        "source_id": "SRC",
                        "source_date": "2026-03-01",
                        "topic_key": "step 8 failure",
                        "state_label": "trace",
                        "currentness": "current",
                        "raw_text": "[Step 8] Action: execute_bash: {\"command\":\"pytest -q\"}\nObservation: command failed",
                        "provenance": {"raw_span": [0, 84]},
                    }
                ],
            }
        ]
    }
    facts = [
        {
            "id": "f_step_8",
            "session": 8,
            "fact": "At step 8, the agent ran pytest -q and the command failed.",
            "metadata": {"episode_id": "SRC_e08", "episode_source_id": "SRC"},
        }
    ]
    index = normalize_temporal_index(
        [
            {
                "span_id": "SRC_e08",
                "source_id": "SRC",
                "timeline_id": "document:SRC",
                "text": corpus["documents"][0]["episodes"][0]["raw_text"],
                "timestamp": "2026-03-01",
                "provenance": {"raw_span": [0, 84]},
                "support_fact_ids": ["f_step_8"],
                "payload": {"episode_id": "SRC_e08"},
            }
        ]
    )

    hit = execute_ordinal_query("Why did step 8 fail?", index)
    assert hit["matched"] is True
    assert hit["resolved"] is False
    assert hit["primary_event"] is None
    assert hit["deterministic_answer"] is None

    packet = build_episode_hybrid_context(
        "Why did step 8 fail?",
        corpus,
        facts,
        search_family="document",
        temporal_index=index,
    )

    assert packet["retrieved_episode_ids"] == ["SRC_e08"]
    assert packet["temporal_trace"]["fallback"] is True
    assert packet["temporal_trace"]["fallback_reason"] == "unresolved"
    assert packet["family_first_pass_trace"].get("mode") != "skipped_by_ordinal_executor"


def test_calendar_answer_query_resolves_overlap_from_explicit_year():
    index = normalize_temporal_index(
        [
            {
                "span_id": "fact:s1_f_23",
                "source_id": "SRC",
                "timeline_id": "timeline:SRC:main",
                "text": "I first watched the movie around 3 years ago.",
                "timestamp": "7:31 pm on 21 January, 2022",
                "provenance": {"start_char": 0, "end_char": 44, "source_field": "raw_text", "episode_id": "SRC_e01"},
                "support_fact_ids": ["s1_f_23"],
                "payload": {"episode_id": "SRC_e01", "fact_id": "s1_f_23"},
            }
        ]
    )

    plan = resolve_calendar_query_interval("What happened in 2019?")
    assert plan["time_start"] == "2019-01-01"
    assert plan["time_end"] == "2019-12-31"

    hit = execute_calendar_query("What happened in 2019?", index)
    assert hit["matched"] is True
    assert [event["event_id"] for event in hit["events"]] == ["fact:s1_f_23:time"]


def test_temporal_runtime_narrows_shared_calendar_answer_query_to_single_source():
    corpus = {
        "documents": [
            {
                "doc_id": "document:SRC_A",
                "episodes": [
                    {
                        "episode_id": "SRC_A_e01",
                        "source_type": "document",
                        "source_id": "SRC_A",
                        "source_date": "2024-05-03",
                        "topic_key": "prius day",
                        "state_label": "trace",
                        "currentness": "current",
                        "raw_text": "On 2024-05-03, Evan drove his Prius hybrid car to work.",
                        "provenance": {"raw_span": [0, 58]},
                    }
                ],
            },
            {
                "doc_id": "document:SRC_B",
                "episodes": [
                    {
                        "episode_id": "SRC_B_e01",
                        "source_type": "document",
                        "source_id": "SRC_B",
                        "source_date": "2024-05-03",
                        "topic_key": "jogging day",
                        "state_label": "trace",
                        "currentness": "current",
                        "raw_text": "On 2024-05-03, Maria went jogging in the park.",
                        "provenance": {"raw_span": [0, 49]},
                    }
                ],
            },
        ]
    }
    facts = [
        {
            "id": "f_a",
            "session": 1,
            "fact": "Evan drove his Prius hybrid car to work on 2024-05-03.",
            "metadata": {"episode_id": "SRC_A_e01", "episode_source_id": "SRC_A"},
        },
        {
            "id": "f_b",
            "session": 1,
            "fact": "Maria went jogging in the park on 2024-05-03.",
            "metadata": {"episode_id": "SRC_B_e01", "episode_source_id": "SRC_B"},
        },
    ]
    index = normalize_temporal_index(
        [
            {
                "span_id": ep["episode_id"],
                "source_id": ep["source_id"],
                "timeline_id": doc["doc_id"],
                "text": ep["raw_text"],
                "timestamp": ep["source_date"],
                "provenance": ep["provenance"],
                "support_fact_ids": [],
                "payload": {"episode_id": ep["episode_id"]},
            }
            for doc in corpus["documents"]
            for ep in doc["episodes"]
        ]
    )

    packet = build_episode_hybrid_context(
        "What happened on 2024-05-03 with the Prius?",
        corpus,
        facts,
        search_family="document",
        temporal_index=index,
    )

    assert packet["temporal_trace"]["fallback"] is False
    assert packet["temporal_trace"]["scope_trace"]["selected_source_ids"] == ["SRC_A"]
    assert packet["retrieved_episode_ids"][0] == "SRC_A_e01"
    assert "prius hybrid car" in packet["context"].lower()


def test_calendar_answer_query_resolves_relative_intervals():
    ago = resolve_calendar_query_interval(
        "What happened 3 years ago?",
        anchor_timestamp="2024-01-21",
    )
    assert ago["time_kind"] == "point"
    assert ago["time_start"] == "2021-01-21"
    assert ago["time_end"] == "2021-01-21"
    assert ago["time_granularity"] == "year"

    last = resolve_calendar_query_interval(
        "What happened last week?",
        anchor_timestamp="2024-01-21",
    )
    assert last["time_kind"] == "interval"
    assert last["time_start"] == "2024-01-14"
    assert last["time_end"] == "2024-01-20"
    assert last["time_granularity"] == "day"

    duration = resolve_calendar_query_interval(
        "What happened for 3 years?",
        anchor_timestamp="2024-01-21",
    )
    assert duration["time_kind"] == "duration"
    assert duration["time_start"] == "2021-01-21"
    assert duration["time_end"] == "2024-01-21"
    assert duration["time_granularity"] == "year"


def test_calendar_answer_query_resolves_deep_history_relative_years():
    ago = resolve_calendar_query_interval(
        "What happened 9000 years ago?",
        anchor_timestamp="2026-01-01",
    )
    assert ago["time_kind"] == "point"
    assert ago["time_start"] == "-6970-12-24"
    assert ago["time_end"] == "-6970-12-24"
    assert ago["time_granularity"] == "year"
    assert isinstance(ago["time_sort_start"], int)
    assert ago["time_sort_start"] == ago["time_sort_end"]


def test_temporal_runtime_pins_calendar_episode_inside_selected_source():
    corpus = {
        "documents": [
            {
                "doc_id": "document:SRC",
                "episodes": [
                    {
                        "episode_id": "SRC_e01",
                        "source_type": "document",
                        "source_id": "SRC",
                        "source_date": "2024-06-01",
                        "topic_key": "launch history",
                        "state_label": "trace",
                        "currentness": "current",
                        "raw_text": "Last year the team launched the product.",
                        "provenance": {"raw_span": [0, 40]},
                    },
                    {
                        "episode_id": "SRC_e02",
                        "source_type": "document",
                        "source_id": "SRC",
                        "source_date": "2024-06-01",
                        "topic_key": "hiring update",
                        "state_label": "trace",
                        "currentness": "current",
                        "raw_text": "This year the team hired two engineers.",
                        "provenance": {"raw_span": [41, 83]},
                    },
                ],
            }
        ]
    }
    facts = [
        {
            "id": "f_hiring",
            "session": 2,
            "fact": "The team hired two engineers this year.",
            "metadata": {"episode_id": "SRC_e02", "episode_source_id": "SRC"},
        }
    ]
    index = normalize_temporal_index(
        [
            {
                "span_id": ep["episode_id"],
                "source_id": ep["source_id"],
                "timeline_id": "document:SRC",
                "text": ep["raw_text"],
                "timestamp": ep["source_date"],
                "provenance": ep["provenance"],
                "support_fact_ids": [],
                "payload": {"episode_id": ep["episode_id"]},
            }
            for ep in corpus["documents"][0]["episodes"]
        ]
    )

    packet = build_episode_hybrid_context(
        "What happened in 2023?",
        corpus,
        facts,
        search_family="document",
        temporal_index=index,
    )

    assert packet["retrieved_episode_ids"][0] == "SRC_e01"
    assert packet["temporal_trace"]["pinned_episode_ids"] == ["SRC_e01"]
    assert "Last year the team launched the product." in packet["context"]
