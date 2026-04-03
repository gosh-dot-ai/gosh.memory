You are extracting structured atomic facts from a conversation between friends.

Each fact must be a SELF-CONTAINED statement with EXACT values.
Someone reading ONLY this fact should get the complete information.

Rules:
1. Extract every specific fact: WHO did WHAT, with EXACT names, dates, numbers, places.
2. Convert ALL relative time references to absolute:
   - Session date is {session_date}. "last year" = {year_minus_1}. "two months ago" = calculate.
   - "recently" = approximate to session month.
   - ALWAYS include the computed absolute date in the fact.
3. One fact per item. If a sentence contains TWO pieces of information, extract TWO separate facts.
   BAD: "Caroline has a dog"
   GOOD: "Caroline adopted a golden retriever puppy named Rex in May 2023"
3b. NUMBERS STAY IN FACTS. If the text mentions a quantity, it must appear in the fact.
   BAD:  "User played Assassin's Creed Odyssey"
   GOOD: "User played Assassin's Creed Odyssey for approximately 70 hours"
4. Preserve exact names, places, numbers. Never paraphrase.
5. Each fact = self-contained sentence that directly answers a question.
6. Include relationship facts: "X and Y are friends/colleagues/siblings."
7. Include opinions/preferences: "Caroline prefers Italian food."
8. Extract DIRECT identity facts: where someone is from, what they do for work,
   family relationships, nationality, hometown.
9. Extract EVERY physical object mentioned: what was made, bought, received,
   given, created, painted. Each object = separate fact.
10. AIM FOR 15-25 FACTS PER SESSION. If you extracted fewer than 10,
    re-read the session — you are missing details.

FACT CLASSIFICATION — the "kind" field controls how this fact is retrieved later.
Choose the MOST SPECIFIC kind that applies:

  kind="fact"           Default. Standard information: what happened, who did what.
                        Example: "Caroline adopted a golden retriever named Rex in May 2023"

  kind="preference"     ONLY for repeated behavior OR explicit statement of preference.
                        Example: "User consistently chooses homemade granola over store-bought"
                        NOT for single mentions. Requires clear evidence.

  kind="rule"           Policies, procedures, requirements, guidelines that apply repeatedly.
  kind="constraint"     Example: "All .tmp files must be categorized as temporary"
                        Triggers: "always", "never", "must", "policy", "rule", "whenever"

  kind="count_item"     Individual countable items: purchases, visits, owned objects, services used.
                        EACH item in a list = separate fact with kind="count_item".
                        Example: "User uses Uber Eats for food delivery" (kind=count_item)
                                 "User uses Fresh Fusion for food delivery" (kind=count_item)
                                 "User uses DoorDash for food delivery" (kind=count_item)

  kind="decision"       A choice or decision made.
  kind="incident"       Something that went wrong or was unexpected.
  kind="lesson_learned" Insight gained from experience.
  kind="action_item"    Task assigned to someone.

SUPERSESSION: If a fact contradicts, updates, or replaces something the user
previously stated (visible in conversation history), set:
  "supersedes_topic": "brief description of what this updates"
  Examples:
    - User said "I have a 42-inch TV" earlier, now says "I got a 55-inch TV"
      → supersedes_topic: "user's TV size"
    - First mention, no contradiction → supersedes_topic: null

TABLES AND SCHEDULES — CRITICAL:
  When a table, schedule, roster, or matrix is presented, extract EVERY ROW
  as a separate fact. Do NOT summarize the table into one fact.
  BAD:  "The shift rotation has 4 shifts and 7 agents"
  GOOD: "Admon works Day Shift (8am-4pm) on Sundays"
        "Admon works Evening Shift (4pm-12am) on Mondays"
        "Magdy works Night Shift (12am-8am) on Sundays"
        (one fact per cell/assignment)

VERBATIM QUOTES — CRITICAL:
  When text is explicitly quoted or a specific phrase/definition is given,
  preserve the EXACT wording in the fact.
  BAD:  "The Library has hexagonal galleries"
  GOOD: "Borges wrote: 'The Library is a sphere whose exact center is any one of its hexagons'"
  BAD:  "The algorithm uses a specific method"
  GOOD: "The 6S algorithm is implemented in the SIAC_GEE tool for atmospheric correction"

ASSISTANT-GENERATED CONTENT:
  Extract facts from BOTH user and assistant messages.
  If the assistant provides factual information (definitions, data, recommendations,
  generated tables), extract those facts too with speaker="assistant".

QUANTITATIVE ATTRIBUTES — CRITICAL:
  Every number, measurement, duration, frequency, count, price MUST appear in the fact.
  Never drop numeric details. Categories:

  DURATIONS:
    BAD:  "User went camping"
    GOOD: "User went on a 3-day camping trip in Yosemite"
    BAD:  "User played a game"
    GOOD: "User played Assassin's Creed Odyssey for approximately 70 hours"

  WEIGHTS AND SIZES:
    BAD:  "User has a large TV"
    GOOD: "User owns a 55-inch Samsung TV"
    BAD:  "User carries heavy luggage"
    GOOD: "User's checked bag weighs 50 lbs"

  FINANCIAL AMOUNTS:
    BAD:  "The project has a large budget"
    GOOD: "The project budget is 400,000 TKT for Phase 1"
    Each line item = separate fact with exact amount.

  FREQUENCIES AND COUNTS:
    BAD:  "User exercises regularly"
    GOOD: "User goes to the gym 4 times a week"
    BAD:  "User has many records"
    GOOD: "User's vinyl collection contains 38 pre-1920 recordings"

  ENUMERABLE ITEMS (kind=count_item):
    When the speaker lists items they own, use, subscribe to, or have visited,
    extract EACH as kind="count_item" — these are retrieval targets for
    "how many" questions.

  APPROXIMATIONS:
    When exact numbers are unavailable, preserve approximations.
    Never drop "about", "roughly", "approximately" qualifiers.
    BAD:  "User drinks coffee"
    GOOD: "User drinks approximately 4 cups of coffee per day"

EXACT DATES — CRITICAL:
- When a conversation states an exact date, ALWAYS extract it verbatim.
  Example: "I got married on June 15, 2019" -> fact: "married June 15, 2019"
- When a relative date is computable, COMPUTE and state the absolute date.
  Example: conversation date is January 2022, speaker says "three years ago"
  -> fact: "approximately January 2019"
- When two dates allow computing a third, COMPUTE it.
  Example: "graduated 2018, got married two years before" -> "married approximately 2016"
- NEVER extract relative time without attempting absolute conversion.
  BAD:  "had turtles for 3 years"
  GOOD: "got turtles approximately January 2019 (3 years before conversation date January 2022)"
- When speaker mentions age or duration, compute birth year / start date.
  Example: "I'm 34" + conversation date March 2024 -> "born approximately 1989-1990"

CRITICAL — EVENT DATE vs SESSION DATE:
- The "event_date" field = the date the EVENT actually occurred, NOT the session date.
- If the user says "last Tuesday I attended a workshop" and the session is 2023-03-07 (Tuesday),
  then "last Tuesday" = 2023-02-28, NOT 2023-03-07.
- If the user says "3 weeks ago I started a course" and session_date is 2023-04-20,
  then event_date = 2023-03-30, NOT 2023-04-20.
- If the user says "yesterday I met John" and session_date is 2023-05-15,
  then event_date = 2023-05-14, NOT 2023-05-15.
- NEVER use session_date as the event_date unless the text explicitly says
  "today", "right now", or "this session".
- If event date is unclear, use approximate: "early 2023", "spring 2023", etc.
  Do NOT default to session_date.

DELTA RULES:

DELTA A — FINANCIAL COMPONENTS. Extract each tax/cost component as a separate fact
with exact amount and formula.
  BAD:  "There are several tax components totalling 334,400 TKT"
  GOOD: "TIL component is 222,000 TKT (12% x 1,850,000 TKT materials)"
  GOOD: "Penalty cap is 5% of contract value (230,000 TKT on 4.6M TKT contract)"

DELTA B — PREFERENCE WITH REASON. When extracting preferences, include why if stated.
  BAD:  "Tim prefers fantasy books"
  GOOD: "Tim prefers fantasy books because they help him escape daily stress"

DELTA C — ORGANIZATIONAL FACTS. Extract role + channel + responsibility for every named person.
  BAD:  "Contact Alice about budget"
  GOOD: "Alice is Finance Controller; approves reimbursements; fallback: Bob for amounts >500 TKT"

TEMPORAL ORDERING:
When events in the conversation have clear temporal sequence,
output a "temporal_links" array pairing fact IDs:

"temporal_links": [
  {{"before": "f_03", "after": "f_05", "signal": "then"}},
  {{"before": "f_07", "after": "f_12", "signal": "two weeks later"}}
]

Signals to look for: "then", "after that", "later", "before",
"the next day", "two weeks later", "last year", "when X happened
Y followed", "first... then...", any chronological ordering.

ONLY link facts with CLEAR temporal signal in the conversation text.
Don't infer order from conversation position alone.

Format: JSON object with "facts" and "temporal_links" arrays:
{{
  "facts": [
    {{
      "id": "f_01",
      "session": {session_num},
      "fact": "Example fact with exact details.",
      "entities": ["Entity1", "Entity2"],
      "tags": ["tag1"],
      "depends_on": [],
      "speaker": "Name of person who stated/reported this, or null if unknown",
      "speaker_role": "client|contractor|pm|engineer|env_officer|procurement|null",
      "kind": "fact|preference|rule|constraint|count_item|decision|incident|lesson_learned|action_item",
      "event_date": "YYYY-MM-DD when this event actually occurred (NOT the session date). Compute from explicit dates ('on January 5th' → '2023-01-05') or relative references ('last Tuesday' → compute from session date {session_date}). null if no specific event date.",
      "supersedes_topic": "Brief description of what prior fact this updates/replaces, or null if first mention.",
      "confidence": "Fill only for kind=preference and kind=decision. null otherwise.",
      "reason": "Fill only for kind=preference and kind=decision. null otherwise."
    }}
  ],
  "temporal_links": [
    {{"before": "f_01", "after": "f_03", "signal": "then"}}
  ]
}}

Extract ALL facts. Be thorough. Do NOT skip minor details.
Do NOT merge multiple details into one fact. Each detail = separate fact.