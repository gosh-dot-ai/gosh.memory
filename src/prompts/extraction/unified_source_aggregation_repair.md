You previously attempted to extract a unified memory substrate for one {source_kind} source and the result failed validation.

SOURCE ID: {source_id}

Return strict JSON only. No markdown. No explanation.

Validation error:
{validation_error}

Previous JSON:
{previous_json}

Repair rules:
1. Fix only the invalid fields and references.
2. Keep already-grounded content unless it directly caused the error.
3. Do not invent new facts to patch a broken reference.
4. If a higher-order object cannot be grounded, remove that object.
5. Keep the same exact top-level shape:
{{
  "revision_currentness": [],
  "events": [],
  "records": [],
  "edges": []
}}

Grounded fact catalog:
{grounded_fact_payload}

Episode texts:
{episode_payload}
