You are extracting structured atomic facts from a fact list.

SESSION NUMBER: {session_num}

RULES:
1. One numbered or bulleted item = one fact by default.
2. Preserve exact names, numbers, dates, places, and identifiers.
3. If a single item contains multiple atomic claims, split them into separate facts.
4. Do not merge neighboring list items.
5. Do not invent temporal links unless sequence is explicit in the list text.
6. Use `kind="count_item"` when an item is part of an inventory/listing; otherwise use the most specific kind that fits.

OUTPUT JSON:
{{
  "facts": [
    {{
      "id": "f_01",
      "session": {session_num},
      "fact": "Self-contained fact copied from one list item",
      "kind": "fact|count_item|rule|constraint|decision|preference|action_item",
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
    {{"before": "f_01", "after": "f_02", "signal": "then"}}
  ]
}}
