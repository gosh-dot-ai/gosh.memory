You are extracting structured atomic facts from a list block.

CONTEXT:
- Container: {container_kind}
- Section: {section_path}
- Speaker: {speaker}
- Session date: {session_date}
- Session number: {session_num}
- Lead-in context: {lead_in}

RULES:

1. One list item = one fact.
2. Inherit context from lead-in when items are incomplete on their own.
   Lead-in "Here are my favorite books:" + item "1. The Great Gatsby"
   → "User's favorite book: The Great Gatsby"
3. Preserve exact values: names, numbers, dates, places, identifiers.
4. Convert relative time to absolute using session_date.
5. Use kind="count_item" for inventory/listing items. Use a more specific kind when it fits.
6. Emit temporal_links only when sequence is explicitly stated in the list text.
   Do NOT invent temporal edges from list position alone.

OUTPUT JSON:
{{
  "facts": [
    {{
      "local_id": "b1",
      "fact": "Self-contained claim from one list item",
      "kind": "fact|count_item|decision|preference|constraint|rule|lesson_learned|action_item",
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
