You are extracting structured atomic facts from narrative prose.

SESSION DATE: {session_date}
SESSION NUMBER: {session_num}
SPEAKERS/CONTEXT: {speakers}

RULES:
1. Preserve exact names exactly as they appear in the text.
2. Preserve exact numbers, dates, places, titles, and identifiers.
3. Extract explicit actions, state changes, and consequential events.
4. Sequence matters: when narrative order is explicit, extract what happened next and populate temporal links.
5. Prefer concrete events over generic character traits or vague summaries.
6. Convert relative time to absolute time when the text and session date make that possible.
7. Do not invent causal chains or temporal links unless the sequence is explicit.
8. Quality over quantity. Extract the memory-relevant events and relationships, not every sentence.

RULE A — SEQUENCE EVENTS.
  Extract cause -> effect chains and sequential plot progression when the text states them.
  BAD:  "Blaire harbored resentment toward Pascal"
  GOOD: "After Pascal sprang onto the quay, Blaire followed him with his eyes"

RULE B — CHARACTER NAMES.
  Extract names exactly as they appear in the text.
  Never substitute with canonical or training-data variants.

OUTPUT JSON:
{{
  "facts": [
    {{
      "id": "f_01",
      "session": {session_num},
      "fact": "Self-contained event or relationship from the narrative",
      "kind": "fact|decision|preference|lesson_learned|action_item",
      "speaker": null,
      "speaker_role": null,
      "event_date": null,
      "entities": [],
      "tags": [],
      "depends_on": [],
      "supersedes_topic": null,
      "confidence": null,
      "reason": null
    }}
  ],
  "temporal_links": [
    {{"before": "f_01", "after": "f_02", "signal": "after"}}
  ]
}}
