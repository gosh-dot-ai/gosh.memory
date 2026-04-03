You are extracting structured atomic facts from a conversation.

Each fact must be SELF-CONTAINED — someone reading only this fact gets the complete information.

SESSION DATE: {session_date}
SPEAKERS: {speakers}
SESSION NUMBER: {session_num}

CORE RULES (proven — do not deviate):

RULE 1 — EXACT VALUES. Preserve exact names, numbers, places, dates. Never paraphrase.
  BAD:  "User has a dog"
  GOOD: "User adopted a golden retriever named Rex from the shelter in May 2023"

RULE 1b — EXACT NAMED TARGETS. CRITICAL.
  Preserve exact named venues, establishments, events, titles, products, and destinations in at least one fact.
  Do NOT replace a named target with a generic activity or category.
  BAD:  "Alice and Ben planned a workshop next Saturday"
  GOOD: "Alice and Ben planned to attend the robotics workshop next Saturday"
  BAD:  "They planned a sports outing next Sunday"
  GOOD: "They planned to go to the jazz concert next Sunday"
  If text contains both a named target and a generic activity, keep the named target in the fact.

RULE 2 — ONE FACT PER ITEM. Two pieces of information = two facts.

RULE 3 — NUMBERS STAY IN FACTS. If the text mentions a quantity, it must appear in the fact.
  BAD:  "User played Assassin's Creed Odyssey"
  GOOD: "User played Assassin's Creed Odyssey for approximately 70 hours"

RULE 4 — TEMPORAL RESOLUTION. Convert ALL relative time to absolute using session_date.
  "last year" + session_date 2023-05-15 → "2022"
  "yesterday" + session_date 2023-06-01 → "2023-05-31"
  event_date = date event OCCURRED, NOT session_date (unless speaker says "today")

RULE 4b — EXACT DATES. Five sub-rules:
  a. Verbatim dates: "I got married on June 15, 2019" → extract as-is
  b. Relative + session: "three years ago" + session Jan 2022 → "approximately January 2019"
  c. Compute from two dates: "graduated 2018, married two years before" → "married ~2016"
  d. NEVER leave relative without absolute:
     BAD: "had turtles for 3 years"
     GOOD: "got turtles approximately January 2019 (3 years before January 2022)"
  e. Age/duration → year: "I'm 34" + March 2024 → "born approximately 1989-1990"

RULE 4c — EVENT DATE vs SESSION DATE. CRITICAL.
  event_date = when the event OCCURRED, NOT when this conversation happened.
  - "last Tuesday" + session 2023-03-07 (Tuesday) → event_date = 2023-02-28
  - "3 weeks ago" + session 2023-04-20 → event_date = approximately 2023-03-30
  - "yesterday" + session 2023-05-15 → event_date = 2023-05-14
  NEVER use session_date as event_date unless speaker says "today" or "right now".

RULE 5 — IDENTITY FACTS. Extract direct identity even if mentioned in passing.

RULE 6 — RELATIONSHIP FACTS. Every stated relationship = separate fact.

RULE 7 — PHYSICAL OBJECTS. Every mentioned object = separate fact with full attributes.

RULE 7b — TABLES AND SCHEDULES. CRITICAL.
  Every ROW = separate fact. NEVER summarize a table.
  BAD:  "The shift rotation has 4 shifts and 7 agents"
  GOOD: "Admon works Day Shift (8am-4pm) on Sundays"
  (one fact per row, preserving all column values)

RULE 7c — VERBATIM QUOTES.
  When speaker quotes or cites exact text, preserve verbatim.
  BAD:  "The library has hexagonal galleries"
  GOOD: "Borges wrote: 'The Library is a sphere whose exact center is any one of its hexagons'"

RULE 7d — TEMPORAL ORDERING.
  When sequence is explicit, populate temporal_links.
  Signals: "then", "after that", "later", "before", "the next day", "two weeks later"
  ONLY link with CLEAR signal. Do NOT infer from sentence position.

DELTA D — ACQUISITION EVENTS. CRITICAL.
  Preserve acquisition/change events as events, not as static possession.
  If speaker says bought/purchased/ordered/booked/got/acquired something, extract that action
  with its time anchor if stated.
  BAD:  "Speaker owns a sports car"
  GOOD: "Speaker bought a sports car in March 2023"
  BAD:  "Speaker has a new house"
  GOOD: "Speaker bought a new house in March 2023"
  If both acquisition and resulting possession are explicit, prefer the acquisition fact.
  If the text names what was acquired, keep that exact named item in the fact.

RULE 8 — KNOWLEDGE UPDATES. When a fact UPDATES or CONTRADICTS a previous statement:
  Signals: "actually", "changed", "now", "no longer", "moved to", "switched to", "updated"
  Extract new fact WITH supersedes_topic = what it updates.
  BOTH old and new facts must be extracted with timestamps.

RULE 9 — BOTH SPEAKERS. Extract facts from BOTH user AND assistant messages.
  Prioritize USER actions, decisions, and stated preferences first.
  From assistant: extract specific recommendations, provided data, or commitments only when they are materially memory-relevant.
  BAD: ignoring "The restaurant I'd recommend is Trattoria Bella"
  GOOD: "Assistant recommended Trattoria Bella for Italian food" (speaker="assistant")

RULE 10 — PRIORITIZE QUALITY OVER QUANTITY. Extract 5-15 facts per session.
  Focus on USER actions, decisions, and stated preferences.
  Assistant recommendations and generic information = lower priority.
  If a session is short (<500 tokens), 3-5 facts is normal.

DELTA RULES:

DELTA A — FINANCIAL COMPONENTS. Extract each tax/cost component as a separate fact with exact amount and formula.
  BAD:  "There are several tax components totalling 334,400 TKT"
  GOOD: "TIL component is 222,000 TKT (12% × 1,850,000 TKT materials)"
  GOOD: "Penalty cap is 5% of contract value (230,000 TKT on 4.6M TKT contract)"

DELTA B — PREFERENCE WITH REASON. When extracting preferences, include why if stated.
  BAD:  "Tim prefers fantasy books"
  GOOD: "Tim prefers fantasy books because they help him escape daily stress (confidence=explicit)"

DELTA C — ORGANIZATIONAL FACTS. Extract role + channel + responsibility for every named person.
  BAD:  "Contact Alice about budget"
  GOOD: "Alice is Finance Controller (@alice, RocketChat #finance); approves reimbursements; fallback: Bob for amounts >500 TKT"

KIND CLASSIFICATION (most specific wins):
  kind="rule"           Policies, procedures, requirements
  kind="constraint"     Stated limitations
  kind="decision"       Choice with rationale
  kind="lesson_learned" Insight from experience
  kind="preference"     Explicit statement or 3+ behavioral repetitions
  kind="count_item"     Individual countable item from a list
  kind="action_item"    Task assigned to someone
  kind="fact"           Everything else (default)

OUTPUT JSON:
{{{{
  "facts": [
    {{{{
      "id": "f_01",
      "session": {session_num},
      "fact": "Self-contained fact with exact values",
      "kind": "fact|preference|rule|constraint|decision|count_item|lesson_learned|action_item",
      "speaker": "user|assistant|null",
      "speaker_role": "user|assistant|agent_name",
      "event_date": "YYYY-MM-DD or approximate",
      "entities": ["Entity1", "Entity2"],
      "tags": [],
      "depends_on": [],
      "supersedes_topic": null,
      "confidence": null,
      "reason": null
    }}}}
  ],
  "temporal_links": [
    {{{{"before": "f_01", "after": "f_03", "signal": "then"}}}}
  ]
}}}}

confidence and reason: fill only for kind="preference" and kind="decision".
Extract memory-relevant facts with precision. Prefer signal over volume.
