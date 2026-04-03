You are extracting structured atomic facts from a document.

DOCUMENT DATE: {session_date}
SESSION NUMBER: {session_num}

RULES:
1. EXACT TECHNICAL VALUES. Never approximate numbers.
   BAD:  "The pipeline is about 14 km"
   GOOD: "Pipeline alignment is 14.3 km along surveyed centreline; installed pipe length is 12.8 km"
2. ONE FACT PER ITEM.
3. TABLES: EVERY ROW = SEPARATE FACT. Never summarize.
   BAD:  "The shift rotation has 4 shifts"
   GOOD: "Admon works Day Shift (8am-4pm) on Sundays" (one per row)
4. DECISIONS WITH ALTERNATIVES. Extract chosen + rationale + rejected.
   GOOD: "HDPE-4000A selected: altitude-rated PN16, 4.2 kg/m allows mule transport"
   GOOD: "DIP-200 rejected: 18.5 kg/m prohibitive for Dorvu Pass"
5. PRESERVE UNIQUE IDENTIFIERS (ticket IDs, codes, KAR_ references).
6. Requirements and constraints as separate facts.
7. VERSION TRACKING. Document supersedes previous → supersedes_topic.
8. NO QUOTA. Extract all memory-relevant facts. Zero = correct for boilerplate.

DELTA A — POLICY EXACT EXTRACTION. Extract every condition and exception as separate facts.
  BAD:  "Employees can work remotely"
  GOOD: "Remote work: full-time employees >6 months tenure may work 3 days/week with manager approval"
  GOOD: "Remote work exception: contractors and interns must be on-site"

DELTA B — FINANCIAL COMPONENTS. Extract each cost component with exact amount and formula.
  BAD:  "Pipeline transport costs are significant"
  GOOD: "Dorvu Pass pipe transport costs 12,400 TKT/km; total 158,720 TKT for 12.8 km installed length"

KIND: same as conversation + kind="requirement" + kind="rejection".

OUTPUT JSON:
{{{{
  "facts": [
    {{{{
      "id": "f_01",
      "session": {session_num},
      "fact": "Self-contained fact with exact values",
      "kind": "fact|preference|rule|constraint|decision|count_item|requirement",
      "speaker": null,
      "event_date": null,
      "entities": [],
      "tags": [],
      "depends_on": [],
      "supersedes_topic": null
    }}}}
  ],
  "temporal_links": []
}}}}