"""Tests for inference prompts."""


def test_synthesis_prompt_format():
    from src.inference import INF_PROMPT_SYNTHESIS
    prompt = INF_PROMPT_SYNTHESIS.format(context="ctx", question="q")
    assert "The user would prefer" in prompt
    assert "Do NOT output bullet points" in prompt


def test_synthesis_prompt_no_markdown_tables():
    from src.inference import INF_PROMPT_SYNTHESIS
    prompt = INF_PROMPT_SYNTHESIS.format(context="ctx", question="q")
    assert "markdown tables" in prompt.lower() or "markdown" in prompt
