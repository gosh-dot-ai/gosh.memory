"""Tests for detect_format() — deterministic format classification."""

from src.librarian import detect_format


def test_conversation_format():
    assert detect_format("user: hello\nassistant: hi") == "CONVERSATION"


def test_conversation_default():
    """Plain text without format markers defaults to CONVERSATION."""
    assert detect_format("Just some text without any special markers.") == "CONVERSATION"


def test_fact_list_format():
    assert detect_format(
        "1. The author wrote a book.\n"
        "2. The book sold 10K copies.\n"
        "3. The author lives in Berlin.\n"
        "4. The author teaches at TU Berlin."
    ) == "FACT_LIST"


def test_document_not_stolen_by_fact_list():
    text = "# Requirements\n\n1. Use SQLite\n2. Preserve ACL semantics\n3. Keep API stable"
    assert detect_format(text) == "DOCUMENT"


def test_raw_chunks_labeled_falls_back_to_conversation():
    text = "question1\nlabel: 5\nquestion2\nlabel: 3\nquestion3\nlabel: 7"
    assert detect_format(text) == "CONVERSATION"


def test_raw_chunks_input_output_falls_back_to_conversation():
    text = "Input: What is 2+2?\nOutput: 4\nInput: What is 3+3?\nOutput: 6"
    assert detect_format(text) == "CONVERSATION"


def test_json_conv():
    assert detect_format("['Chat Time: 2023', [{'role': 'user'}]]") == "JSON_CONV"


def test_json_conv_role():
    text = '[{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]'
    assert detect_format(text) == "JSON_CONV"


def test_agent_trace():
    assert detect_format("[Step 1]\nAction: click\nObservation: done") == "AGENT_TRACE"


def test_agent_trace_multi_step():
    text = "[Step 1]\nAction: navigate\nObservation: page loaded\n[Step 2]\nAction: click\nObservation: done"
    assert detect_format(text) == "AGENT_TRACE"


def test_document():
    assert detect_format("# Chapter 1\n---\n**Title:** Introduction") == "DOCUMENT"


def test_document_markdown_heading():
    text = "# Overview\n\nThis document describes the pipeline architecture."
    assert detect_format(text) == "DOCUMENT"


def test_document_bold_key():
    # Regex matches **Word**: (colon after closing bold)
    text = "**Author**: John Smith\n**Date**: 2024-01-15\n\nContent here."
    assert detect_format(text) == "DOCUMENT"


def test_narrative():
    text = (
        "He stood on the quay and watched the ferry leave.\n\n"
        "\"We are too late,\" she said. After a minute, they turned back toward the market. "
        "Later that evening, his brother found them near the old warehouse."
    )
    assert detect_format(text) == "NARRATIVE"


def test_plain_text_prefers_conversation_over_narrative():
    text = "The user mentioned they like tea and prefer quiet cafes on weekends."
    assert detect_format(text) == "CONVERSATION"


def test_web_dom():
    text = "RootWebArea focused: true\n  navigation 'Main menu'\n  heading 'Welcome'"
    assert detect_format(text) == "WEB_DOM"


def test_game_board():
    text = "A| B C D E\nB| C D E F\nC| D E F G\nD| E F G H"
    assert detect_format(text) == "GAME_BOARD"


def test_code_trace():
    text = "$ ls -la\ntotal 42\nexecute_bash: echo hello\nEXECUTION RESULT: hello"
    assert detect_format(text) == "CODE_TRACE"


def test_code_trace_dollar_prompt():
    text = "$ python script.py\nRunning...\nDone."
    assert detect_format(text) == "CODE_TRACE"
