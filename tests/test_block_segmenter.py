"""Tests for block_segmenter — Stage 1 of ContentIR Phase 1.

Policy:
  SOURCE-FIRST: explicit speakers from caller beat inference.
  PRECISION-FIRST: ambiguous input stays unsplit.
"""

from src.block_segmenter import Block, detect_speakers, segment_conversation_blocks

# ═══════════════════════════════════════════════════════════════════
# detect_speakers — known roles
# ═══════════════════════════════════════════════════════════════════


def test_detect_speakers_user_assistant():
    text = "user: hello\nassistant: hi there\nuser: how are you?"
    speakers = detect_speakers(text)
    assert "user" in speakers
    assert "assistant" in speakers


def test_detect_speakers_known_role_single_occurrence():
    """Known roles qualify even with 1 occurrence."""
    text = "user: hello\nassistant: hi"
    speakers = detect_speakers(text)
    assert "user" in speakers
    assert "assistant" in speakers


# ═══════════════════════════════════════════════════════════════════
# detect_speakers — named speakers (no known roles)
# ═══════════════════════════════════════════════════════════════════


def test_detect_speakers_named_locomo():
    text = (
        "Caroline: I went to a support group yesterday.\n"
        "It was so powerful and inspiring.\n"
        "Melanie: That's great!\n"
        "What was it like?\n"
        "Caroline: The transgender stories were amazing.\n"
        "I learned so much about acceptance.\n"
        "Melanie: I'm happy for you."
    )
    speakers = detect_speakers(text)
    assert "caroline" in speakers
    assert "melanie" in speakers
    assert speakers["caroline"] == "Caroline"
    assert speakers["melanie"] == "Melanie"


def test_detect_speakers_dr_smith():
    text = (
        "Dr. Smith: The results are in.\n"
        "Everything looks good overall.\n"
        "Patient: What do they say?\n"
        "I've been worried.\n"
        "Dr. Smith: Everything looks normal.\n"
        "No need to worry at all."
    )
    speakers = detect_speakers(text)
    assert "dr. smith" in speakers
    assert speakers["dr. smith"] == "Dr. Smith"


def test_detect_speakers_single_occurrence_not_promoted():
    """Single-occurrence named speaker is NOT promoted (precision-first)."""
    text = (
        "Caroline: I adopted a golden retriever puppy.\n"
        "The vet says he's healthy.\n"
        "Rachel: How old is he?\n"
        "Caroline: 3 months old.\n"
        "He's growing fast."
    )
    speakers = detect_speakers(text)
    assert "caroline" in speakers
    assert "rachel" not in speakers


def test_detect_speakers_ignores_rare_candidates():
    """RandomPerson (freq 1) between known-role speakers is not promoted."""
    text = "user: hello\nRandomPerson: one time only\nuser: bye"
    speakers = detect_speakers(text)
    assert "user" in speakers
    assert "randomperson" not in speakers


# ═══════════════════════════════════════════════════════════════════
# detect_speakers — KV rejection (known roles present)
# ═══════════════════════════════════════════════════════════════════


def test_kv_lines_not_detected_as_speakers():
    """KV lines inside known-role turns: never promoted."""
    text = "user: Here are specs:\nSpec: PN16\nCost: 10\nSpec: PN25\nCost: 12\nassistant: ok"
    speakers = detect_speakers(text)
    assert "spec" not in speakers
    assert "cost" not in speakers
    assert set(speakers.keys()) == {"user", "assistant"}


def test_kv_long_values_with_known_roles():
    """Long KV values between known-role speakers: never promoted."""
    text = "user: Here are specs\nSpec: high pressure rating\nCost: twelve thousand dollars\nassistant: ok"
    speakers = detect_speakers(text)
    assert "spec" not in speakers
    assert "cost" not in speakers


# ═══════════════════════════════════════════════════════════════════
# detect_speakers — KV rejection (no known roles)
# ═══════════════════════════════════════════════════════════════════


def test_kv_lines_no_known_roles():
    """KV blob with no known roles: all singletons, no speakers."""
    text = "Spec: PN16\nCost: 10\nWeight: 4.2"
    speakers = detect_speakers(text)
    assert speakers == {}


def test_kv_alternating_no_known_roles():
    """Alternating KV labels (Spec/Cost/Spec/Cost): freq>=2 but self-consecutive."""
    text = "Spec: PN16\nCost: 10\nSpec: PN25\nCost: 12"
    speakers = detect_speakers(text)
    assert "spec" not in speakers
    assert "cost" not in speakers


def test_kv_alternating_with_periods():
    """Alternating KV with sentence-final periods: still KV, not speakers."""
    text = "Spec: PN16.\nCost: 10.\nSpec: PN25.\nCost: 12."
    speakers = detect_speakers(text)
    assert speakers == {}


def test_kv_name_role_alternating():
    """Name/Role alternating KV: no speakers."""
    text = "Name: Alice Johnson\nRole: Finance Lead\nName: Bob Smith\nRole: Operations Manager"
    speakers = detect_speakers(text)
    assert speakers == {}


def test_kv_long_values_no_known_roles():
    """Long KV values without known roles: no speakers."""
    text = "Spec: high pressure rating\nCost: twelve thousand dollars\nWeight: schedule B compliant"
    speakers = detect_speakers(text)
    assert speakers == {}


def test_kv_continuation_lines_no_known_roles():
    """KV with continuation lines: structurally indistinguishable from
    2-line conversation turns. Inference may promote them (known Phase 1
    limitation). Caller uses explicit speakers={} to prevent false split."""
    text = "Spec: PN16\n  schedule B\nCost: 10\n  usd\nSpec: PN25\n  schedule C\nCost: 12\n  usd"
    # Phase 1 limitation: continuation-line KV passes freq+alternation check.
    # Mitigation: caller passes explicit empty speakers to force no-split.
    blocks = segment_conversation_blocks(text, speakers={})
    assert len(blocks) == 1
    assert blocks[0].family == "PROSE"
    assert blocks[0].speaker is None
    assert "Spec" in blocks[0].text


# ═══════════════════════════════════════════════════════════════════
# detect_speakers — multilingual
# ═══════════════════════════════════════════════════════════════════


def test_multilingual_ambiguous_stays_unsplit():
    """Short single-line turns without multi-line content: ambiguous, no split."""
    text = "Алиса: очень рада\nБоб: тоже рад\nАлиса: отлично\nБоб: супер"
    speakers = detect_speakers(text)
    # All lines are prefix lines, no multi-line turns → ambiguous → no split.
    assert speakers == {}


# ═══════════════════════════════════════════════════════════════════
# segment_conversation_blocks — explicit speakers (source-first)
# ═══════════════════════════════════════════════════════════════════


def test_explicit_speakers_override_inference():
    """Caller-provided speakers bypass detect_speakers entirely."""
    text = "Caroline: hi\nRachel: ok\nCaroline: done"
    # Without explicit speakers, Rachel (freq 1) would NOT be detected
    assert "rachel" not in detect_speakers(text)
    # With explicit speakers, Rachel IS used
    blocks = segment_conversation_blocks(
        text, speakers={"caroline": "Caroline", "rachel": "Rachel"})
    rachel_blocks = [b for b in blocks if b.speaker == "Rachel"]
    assert len(rachel_blocks) == 1
    assert "ok" in rachel_blocks[0].text


def test_explicit_speakers_single_occurrence_splits():
    """Single-occurrence named speaker works when passed explicitly."""
    text = (
        "Caroline: I adopted a golden retriever puppy.\n"
        "The vet says he's healthy.\n"
        "Rachel: How old is he?\n"
        "Caroline: 3 months old.\n"
        "He's growing fast."
    )
    blocks = segment_conversation_blocks(
        text, speakers={"caroline": "Caroline", "rachel": "Rachel"})
    rachel_blocks = [b for b in blocks if b.speaker == "Rachel"]
    assert len(rachel_blocks) == 1
    assert "How old" in rachel_blocks[0].text


# ═══════════════════════════════════════════════════════════════════
# segment_conversation_blocks — block families
# ═══════════════════════════════════════════════════════════════════


def test_pure_prose_conversation():
    text = "user: I bought a new bookshelf from IKEA.\nassistant: That's nice! What color did you get?"
    blocks = segment_conversation_blocks(text)
    assert len(blocks) >= 2
    assert all(b.family == "PROSE" for b in blocks)
    for b in blocks:
        assert not b.text.startswith("user:")
        assert not b.text.startswith("assistant:")


def test_embedded_list():
    text = "user: Here are my favorite books:\n1. The Great Gatsby\n2. To Kill a Mockingbird\n3. 1984\nassistant: Great choices!"
    blocks = segment_conversation_blocks(text)
    assert "LIST" in [b.family for b in blocks]
    list_block = [b for b in blocks if b.family == "LIST"][0]
    assert "Great Gatsby" in list_block.text


def test_embedded_table():
    text = "user: Here's the schedule:\n| Day | Shift | Agent |\n| Mon | Morning | Alice |\n| Tue | Evening | Bob |\nassistant: Got it."
    blocks = segment_conversation_blocks(text)
    assert "TABLE" in [b.family for b in blocks]


def test_mixed_blocks():
    text = (
        "user: Let me share some info.\n"
        "Here are the tasks:\n"
        "1. Buy groceries\n"
        "2. Clean the house\n"
        "3. Walk the dog\n"
        "And here's the budget:\n"
        "| Item | Cost |\n"
        "| Groceries | $50 |\n"
        "| Cleaning | $20 |\n"
        "assistant: Thanks for sharing."
    )
    blocks = segment_conversation_blocks(text)
    families = [b.family for b in blocks]
    assert "PROSE" in families
    assert "LIST" in families
    assert "TABLE" in families


def test_single_bullet_stays_prose():
    text = "user: I have one item:\n- Buy milk\nassistant: Noted."
    blocks = segment_conversation_blocks(text)
    assert not any(b.family == "LIST" for b in blocks)


def test_single_table_row_stays_prose():
    text = "user: comparison:\n| Material | Weight |\nassistant: ok"
    blocks = segment_conversation_blocks(text)
    assert not any(b.family == "TABLE" for b in blocks)


# ═══════════════════════════════════════════════════════════════════
# segment_conversation_blocks — prose sub-segmentation
# ═══════════════════════════════════════════════════════════════════


def test_long_prose_sub_segmented():
    long_text = "user: " + " ".join([f"This is a sentence about topic number {i}." for i in range(80)])
    blocks = segment_conversation_blocks(long_text)
    prose_blocks = [b for b in blocks if b.family == "PROSE" and b.speaker == "user"]
    assert len(prose_blocks) >= 2


def test_long_prose_sub_segmented_raw_span():
    """Sub-segmented prose blocks have correct raw_span provenance."""
    long_text = "user: " + " ".join([f"This is a sentence about topic number {i}." for i in range(80)])
    blocks = segment_conversation_blocks(long_text)
    prose_blocks = [b for b in blocks if b.family == "PROSE" and b.speaker == "user"]
    assert len(prose_blocks) >= 2
    for b in prose_blocks:
        start, end = b.raw_span
        assert 0 <= start <= end <= len(long_text)
        span_text = long_text[start:end]
        assert b.text.strip() in span_text


# ═══════════════════════════════════════════════════════════════════
# segment_conversation_blocks — speaker inheritance + role
# ═══════════════════════════════════════════════════════════════════


def test_speaker_inheritance():
    text = "user: Hello there.\nassistant: Hi! How can I help?\nuser: I need advice."
    blocks = segment_conversation_blocks(text)
    user_blocks = [b for b in blocks if b.speaker == "user"]
    assistant_blocks = [b for b in blocks if b.speaker == "assistant"]
    assert len(user_blocks) >= 2
    assert len(assistant_blocks) >= 1
    for b in user_blocks:
        assert b.speaker_role == "user"
    for b in assistant_blocks:
        assert b.speaker_role == "assistant"


def test_named_speakers_locomo():
    text = (
        "Caroline: I went to a LGBTQ support group yesterday.\n"
        "It was so powerful and inspiring.\n"
        "Melanie: That's amazing!\n"
        "What was it like?\n"
        "Caroline: The transgender stories were so inspiring!\n"
        "I learned so much.\n"
        "Melanie: I'm so happy for you."
    )
    blocks = segment_conversation_blocks(text)
    caroline_blocks = [b for b in blocks if b.speaker == "Caroline"]
    melanie_blocks = [b for b in blocks if b.speaker == "Melanie"]
    assert len(caroline_blocks) >= 2
    assert len(melanie_blocks) >= 2
    for b in caroline_blocks:
        assert b.speaker_role is None
    for b in blocks:
        assert not b.text.startswith("Caroline:")
        assert not b.text.startswith("Melanie:")


# ═══════════════════════════════════════════════════════════════════
# segment_conversation_blocks — raw_span provenance
# ═══════════════════════════════════════════════════════════════════


def test_raw_span_provenance():
    text = "user: I bought a bookshelf.\nassistant: Where from?\nuser: IKEA."
    blocks = segment_conversation_blocks(text)
    for b in blocks:
        start, end = b.raw_span
        assert 0 <= start <= end <= len(text)
        span_text = text[start:end]
        if b.text.strip():
            assert b.text.strip() in span_text


def test_raw_span_includes_speaker_prefix():
    """First block of a turn includes speaker prefix in raw_span."""
    text = "user: hello\nassistant: hi there"
    blocks = segment_conversation_blocks(text)
    first_block = blocks[0]
    span_text = text[first_block.raw_span[0]:first_block.raw_span[1]]
    assert "user:" in span_text


# ═══════════════════════════════════════════════════════════════════
# segment_conversation_blocks — Fender Stratocaster
# ═══════════════════════════════════════════════════════════════════


def test_fender_stratocaster_short_utterance():
    """Short user utterance mentioning a specific model must not be lost."""
    text = (
        "user: I'm considering upgrading from a Fender Stratocaster to a Gibson Les Paul.\n"
        "assistant: Both are excellent guitars."
    )
    blocks = segment_conversation_blocks(text)
    all_text = " ".join(b.text for b in blocks)
    assert "Fender Stratocaster" in all_text
    assert "Gibson Les Paul" in all_text
    user_blocks = [b for b in blocks if b.speaker == "user"]
    assert any("Fender" in b.text for b in user_blocks)


# ═══════════════════════════════════════════════════════════════════
# segment_conversation_blocks — precision-first (no false splits)
# ═══════════════════════════════════════════════════════════════════


def test_single_occurrence_stays_in_turn():
    """Rachel (freq 1) stays inside Caroline's turn without explicit speakers."""
    text = (
        "Caroline: I adopted a golden retriever puppy.\n"
        "The vet says he's healthy.\n"
        "Rachel: How old is he?\n"
        "Caroline: 3 months old.\n"
        "He's growing fast."
    )
    blocks = segment_conversation_blocks(text)
    rachel_blocks = [b for b in blocks if b.speaker == "Rachel"]
    assert len(rachel_blocks) == 0
    all_text = " ".join(b.text for b in blocks)
    assert "How old" in all_text


def test_kv_inside_named_speaker_turn():
    """KV lines and single-occurrence names stay as turn content."""
    text = (
        "Caroline: specs below\n"
        "Spec: PN16\n"
        "Cost: 10\n"
        "Rachel: ok\n"
        "Caroline: done"
    )
    speakers = detect_speakers(text)
    assert "spec" not in speakers
    assert "cost" not in speakers
    assert "rachel" not in speakers


def test_kv_long_values_inside_named_turn():
    """Long KV between named speakers: ambiguous, no split."""
    text = "Caroline: specs below\nSpec: high pressure rating schedule B\nRachel: ok\nCaroline: done"
    speakers = detect_speakers(text)
    assert "spec" not in speakers


# ═══════════════════════════════════════════════════════════════════
# segment_conversation_blocks — lead_in
# ═══════════════════════════════════════════════════════════════════


def test_lead_in_for_list():
    text = "user: Here are my tasks:\n1. Buy groceries\n2. Walk dog\n3. Cook dinner"
    blocks = segment_conversation_blocks(text)
    list_blocks = [b for b in blocks if b.family == "LIST"]
    assert len(list_blocks) == 1
    assert list_blocks[0].lead_in is not None
    assert "tasks" in list_blocks[0].lead_in.lower()


# ═══════════════════════════════════════════════════════════════════
# segment_conversation_blocks — structural invariants
# ═══════════════════════════════════════════════════════════════════


def test_exclusive_ownership():
    """No character should appear in two blocks' raw_spans."""
    text = (
        "user: Hello.\n"
        "Here's a list:\n"
        "1. Item A\n"
        "2. Item B\n"
        "And a table:\n"
        "| Col1 | Col2 |\n"
        "| A | B |\n"
        "assistant: Thanks."
    )
    blocks = segment_conversation_blocks(text)
    covered = set()
    for b in blocks:
        start, end = b.raw_span
        for i in range(start, end):
            assert i not in covered
            covered.add(i)


def test_order_sequential():
    text = "user: First.\nassistant: Second.\nuser: Third."
    blocks = segment_conversation_blocks(text)
    orders = [b.order for b in blocks]
    assert orders == sorted(orders)
    assert len(set(orders)) == len(orders)
