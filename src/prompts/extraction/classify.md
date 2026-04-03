Read the text below. Do NOT rewrite or summarize it.
Return ONLY a JSON metadata object.

RULES:
- kind: most specific match from the list below
- entities: proper nouns, named things (people, places, orgs, products, works of art)
- tags: 2-5 topical keywords (lowercase, no spaces)
- event_date: when the event occurred (YYYY-MM-DD or "~YYYY"), null if no date mentioned
- supersedes_topic: if this corrects/updates earlier info, what it updates; null otherwise

KIND OPTIONS (pick most specific):
  "rule"           — policy, procedure, requirement
  "constraint"     — stated limitation or boundary
  "decision"       — choice with rationale
  "lesson_learned" — insight from experience
  "preference"     — explicit like/dislike or repeated choice
  "count_item"     — individual countable item (from a list or inventory)
  "action_item"    — task assigned to someone
  "observation"    — casual observation, transient note
  "fact"           — everything else (default)

Text:
{chunk_text}

Respond with ONLY valid JSON, no markdown fences, no explanation:
{{"kind": "...", "entities": [...], "tags": [...], "event_date": null, "supersedes_topic": null}}