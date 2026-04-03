You are extracting structured atomic facts from a prose block.

CONTEXT:
- Container: {container_kind}
- Section: {section_path}
- Speaker: {speaker}
- Session date: {session_date}
- Session number: {session_num}
- Lead-in context: {lead_in}

RULES:

1. Extract explicit claims only. Each fact must be self-contained.
2. Preserve conflicting claims separately. Do NOT collapse contradictions.
3. Preserve exact values: names, numbers, dates, places, identifiers.
4. Convert relative time to absolute using session_date.
   "yesterday" + session 2023-05-08 → "2023-05-07"
   "last Tuesday" + session 2023-05-15 → "2023-05-09"
5. Do NOT emit bare entities as facts.
   BAD: "IKEA" (just a name, no claim)
   GOOD: "User bought a bookshelf from IKEA"
6. Speech acts vs embedded propositions:
   - Preserve embedded propositions (facts stated within a request/question).
     "I'm considering upgrading from a Fender Stratocaster" → extract ownership fact.
   - Do NOT preserve pure requests/acknowledgments unless they are action_items or commitments.
7. Emit temporal_links only when there is an explicit temporal signal in the text.
   Do NOT invent temporal edges from sentence position alone.

OUTPUT JSON:
{{
  "facts": [
    {{
      "local_id": "b1",
      "fact": "Self-contained claim with exact values",
      "kind": "fact|decision|preference|constraint|rule|count_item|lesson_learned|action_item",
      "entities": [],
      "tags": [],
      "depends_on": [],
      "supersedes_topic": null,
      "confidence": null,
      "event_date": null
    }}
  ],
  "temporal_links": [
    {{
      "before": "b1",
      "after": "b2",
      "relation": "before",
      "signal": "then"
    }}
  ]
}}
