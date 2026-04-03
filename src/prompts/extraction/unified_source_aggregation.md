You are extracting a unified memory substrate across multiple episodes of one {source_kind} source.

SOURCE ID: {source_id}

Return strict JSON only. No markdown. No explanation.

You must output this exact top-level shape:
{{
  "revision_currentness": [],
  "events": [],
  "records": [],
  "edges": []
}}

Extraction rules:
1. Extract only what is explicitly grounded in the episode texts.
2. If a value is uncertain or not explicit, omit that object or leave the field null.
3. Do not invent facts, dates, units, participants, locations, headings, or relations.
4. Do not re-extract or rewrite atomic facts. Base grounded facts are already provided below.
5. `support_fact_ids` and `anchor_basis_fact_ids` must reference fact ids from the grounded fact catalog below.
6. Prefer causally central events and shared root events over broad recurring themes.
7. For current/revised values, emit `revision_currentness` when the text explicitly supports supersession/currentness.
8. For tests, procedures, incidents, approvals, permits, schedules, and shared roots, emit structured `events` and `records` when grounded.
9. Emit `same_anchor` only when the support facts clearly refer to the same anchor.
10. Emit semantic edges like `causes`, `leads_to`, `supports`, `conflicts_with`, `bridge_to`, `resolver_for` only when grounded by the support facts.
11. If you cannot ground a higher-order object, omit that higher-order object.
12. Do not return empty success unless there is truly no grounded higher-order structure in the provided facts.

Schema notes:
- `revision_currentness[*]` fields:
  - `revision_id`, `topic_key`, `old_fact_id`, `new_fact_id`, `link_type`, `current_fact_id`, `effective_date`, `revision_source_fact_ids`
- `events[*]` fields:
  - `event_id`, `event_type`, `participants`, `object`, `time`, `location`, `parameters`, `outcome`, `status`, `support_fact_ids`
- `records[*]` fields:
  - `record_id`, `record_type`, `item_id`, `status`, `date`, `qualifier`, `owner`, `source_section`, `support_fact_ids`
- `edges[*]` fields:
  - `edge_id`, `edge_type`, `from_id`, `to_id`, `edge_evidence_text`, `anchor_key`, `anchor_basis_fact_ids`, `support_fact_ids`

Allowed enums:
- `polarity`: `positive`, `negative`, `uncertain`
- `link_type`: `supersedes`, `superseded_by`, `current_value_for`
- `edge_type`: `causes`, `leads_to`, `supports`, `same_anchor`, `conflicts_with`, `bridge_to`, `resolver_for`, `belongs_to_event`, `belongs_to_record`

Grounded fact catalog:
{grounded_fact_payload}

Episode texts:
{episode_payload}
