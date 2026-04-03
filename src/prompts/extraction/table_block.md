You are extracting structured atomic facts from a table block.

CONTEXT:
- Container: {container_kind}
- Section: {section_path}
- Speaker: {speaker}
- Session date: {session_date}
- Session number: {session_num}
- Lead-in context: {lead_in}

RULES:

1. One row = one fact. Include column headers in each fact for self-containment.
   BAD: "Alice, Morning, Sunday" (missing column context)
   GOOD: "Alice works the Morning shift on Sunday"
2. Preserve exact values, units, identifiers from each cell.
3. Convert relative time to absolute using session_date.
4. Emit temporal_links only when table explicitly states ordering.
   Do NOT invent temporal edges from row position alone.

OUTPUT JSON:
{{
  "facts": [
    {{
      "local_id": "b1",
      "fact": "Self-contained claim from one table row with column context",
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
