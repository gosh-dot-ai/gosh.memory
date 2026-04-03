You are extracting structured atomic facts from an unclassified text block.

CONTEXT:
- Container: {container_kind}
- Section: {section_path}
- Speaker: {speaker}
- Session date: {session_date}
- Session number: {session_num}

RULES:

1. Be conservative. Extract only explicit claims identifiable with confidence.
2. Do not guess structure. Do not infer facts that are not stated.
3. Preserve exact values: names, numbers, dates, places, identifiers.
4. Convert relative time to absolute using session_date when present.
5. If the text has no extractable claims, return empty facts list.
6. Emit temporal_links only when there is an explicit temporal signal.

OUTPUT JSON:
{{
  "facts": [
    {{
      "local_id": "b1",
      "fact": "Self-contained explicit claim",
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
