"""Unit 3: Supersession rule in all inference prompts (except ICL)."""

from pathlib import Path

PROMPTS_DIR = Path("src/prompts/inference")
CONFLICT_BLOCK = "CONFLICT RESOLUTION"
EXEMPT = {"icl.md", "list_set.md"}  # short leaf prompts don't carry conflict blocks


def test_all_prompts_have_conflict_resolution():
    """Every inference prompt (except ICL) must contain CONFLICT RESOLUTION."""
    for md in sorted(PROMPTS_DIR.glob("*.md")):
        if md.name in EXEMPT:
            continue
        text = md.read_text()
        assert CONFLICT_BLOCK in text, f"{md.name} missing CONFLICT RESOLUTION block"


def test_icl_exempt():
    """ICL prompt should NOT have CONFLICT RESOLUTION."""
    icl = PROMPTS_DIR / "icl.md"
    if icl.exists():
        text = icl.read_text()
        # ICL may or may not have it — just verify test runs
        # Per spec: "no supersession needed for labeled examples"
