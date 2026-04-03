"""Tests that EXTRACTION_PROMPT contains required quantitative rules."""
import pytest

from src.librarian import EXTRACTION_PROMPT


def test_quantitative_section_exists():
    assert "QUANTITATIVE ATTRIBUTES" in EXTRACTION_PROMPT


def test_duration_bad_good_examples():
    """Duration examples must be present."""
    assert "70 hours" in EXTRACTION_PROMPT
    assert "3-day camping" in EXTRACTION_PROMPT


def test_weight_bad_good_examples():
    """Weight/size examples must be present."""
    assert "50 lbs" in EXTRACTION_PROMPT
    assert "55-inch" in EXTRACTION_PROMPT


def test_financial_bad_good_examples():
    """Financial examples must be present."""
    assert "400,000" in EXTRACTION_PROMPT


def test_frequency_bad_good_examples():
    """Frequency/count examples must be present."""
    assert "4 times a week" in EXTRACTION_PROMPT
    assert "38 pre-1920" in EXTRACTION_PROMPT


def test_enumerable_items_count_item():
    """Enumerable items section must specify kind=count_item."""
    quant_section = EXTRACTION_PROMPT[
        EXTRACTION_PROMPT.find("QUANTITATIVE"):
        EXTRACTION_PROMPT.find("EXACT DATES")
    ]
    assert "count_item" in quant_section


def test_approximations_preserved():
    """Approximation rules must be present."""
    assert "APPROXIMATIONS" in EXTRACTION_PROMPT
    assert "never drop" in EXTRACTION_PROMPT.lower() or \
           "never drop" in EXTRACTION_PROMPT


def test_quantitative_before_exact_dates():
    """QUANTITATIVE section must come before EXACT DATES section."""
    q_pos = EXTRACTION_PROMPT.find("QUANTITATIVE ATTRIBUTES")
    d_pos = EXTRACTION_PROMPT.find("EXACT DATES")
    assert q_pos < d_pos, "QUANTITATIVE must appear before EXACT DATES"
