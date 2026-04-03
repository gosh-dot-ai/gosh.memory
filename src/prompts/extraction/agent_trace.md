You are extracting memory-relevant facts from an agent execution trace.

EPISODE ID: {episode_id}
DOMAIN: {domain}
CHUNK: {chunk_num} of {total_chunks}

CRITICAL: Aggressive extraction HURTS agent traces. Extract ONLY facts that answer
"what happened" and "what changed".

EXTRACT:
- Actions that changed state: "Agent picked up box 3 from sofa 1"
- Observations that reveal information: "Sidetable 3 contains a houseplant"
- Errors and their causes: "Action failed: door is locked"
- Goals achieved: "Agent placed box 3 in the cabinet"

DO NOT EXTRACT:
- Pure navigation steps that reveal nothing new
- Repeated identical observations
- DOM chrome (nav bars, footers, cookie banners)
- Empty tool outputs

DOMAIN-SPECIFIC:
  WEB: Extract current page URL+title, element interacted with, key info found
  GAME_BOARD: Extract action (coord→coord), score change, state change
  CODE: Extract command, exit_code, key output, error if any
  EMBODIED_AI: Standard extraction — what agent did, found, changed

OUTPUT JSON:
{{{{
  "facts": [
    {{{{
      "id": "f_01",
      "session": {chunk_num},
      "fact": "Agent [action] → [result/observation]",
      "kind": "fact",
      "speaker": "agent",
      "event_date": null,
      "entities": [],
      "tags": [],
      "depends_on": [],
      "supersedes_topic": null
    }}}}
  ],
  "temporal_links": []
}}}}

Zero facts acceptable for boilerplate steps.

DELTA A — PRE/POST STATE. For actions that change state, note what changed.
  BAD:  "Agent moved box 3 to cabinet"
  GOOD: "Agent moved box 3 from sofa 1 to cabinet [pre: box 3 on sofa 1] → [post: box 3 in cabinet, sofa 1 empty]"
  Only for significant state changes. Skip for pure navigation.