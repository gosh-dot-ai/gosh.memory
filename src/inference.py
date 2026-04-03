#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOSH Memory — inference / answer generation.

Prompts for generating answers from retrieved context.
Tool-augmented inference for temporal/counting queries.
"""

import json
from dataclasses import dataclass
from pathlib import Path as _Path

from src.tools import count_items, date_diff

# ── Inference prompts ──

INF_PROMPT = """Based on the following memory about conversations between {speakers}, answer the question.

APPROACH (Think step by step):
1. Examine ALL provided facts from every source.
2. Identify facts related to the question — even INDIRECTLY related.
3. If no single fact answers directly, COMBINE multiple facts to reason.
   Example: "loves reading" + "wants to help people" -> "might enjoy library science or counseling"
4. Connect people, events, and contexts across different sessions.
5. Formulate a precise answer based on your reasoning.
6. Only say "No information available" if the facts are ENTIRELY unrelated to the question.

COUNTING QUESTIONS ("how many", "how often", "count", "number of"):
1. List EVERY individual instance you find in the provided facts, numbered:
   1) event one
   2) event two
   3) event three
2. Count your list.
3. Respond with the specific number as the final answer, not a list.
   Example: "3" not "1) navy blazer 2) dress 3) jacket"
4. If you cannot find ALL instances, state what you found and note uncertainty.

MULTI-HOP / INFERENCE QUESTIONS:
When no single fact directly answers the question:
1. Identify what TYPE of information the question needs.
2. Check if combining two or more facts produces the answer.
3. Follow logical chains: if A implies B, and B implies C, conclude C.
4. For "what X would satisfy Y" questions:
   - Find Y's constraints from the facts.
   - Find X's that match those constraints.
   - State the match explicitly.
5. Only say "No information available" if facts are ENTIRELY unrelated.

RULES:
- Use EXACT values: names, dates, numbers, places.
- If multiple facts about the same topic, prefer the most recent session.
- For date questions, give the specific date.
- Answer concisely but ALWAYS attempt an answer before saying "no info."

{context}

Question: {question}
Short answer:"""

INF_PROMPT_TEMP = """Based on the following memory about conversations between {speakers}, answer with an approximate date.
If no relevant information is found, say "No information available".

{context}

Question: {question}
Short answer:"""

INF_PROMPT_ADV = """Based ONLY on the following context about conversations between {speakers}, answer briefly.
If the answer is NOT in the context, you MUST say "No information available".

{context}

Question: {question}
Short answer:"""

# ── Specialized inference prompts ──

INF_PROMPT_TEMPORAL = """You have memory facts about a person. Answer the question using ONLY these facts.

{context}

Question: {question}

IMPORTANT: The reference date is {reference_date} (date of the last conversation session).
Do NOT use today's real-world date. All time calculations must use {reference_date} as "now".

TEMPORAL REASONING PROTOCOL — follow exactly:
Step 1: List ALL dates relevant to this question from the facts above.
        Format: [YYYY-MM-DD or approximate] — [what happened]
Step 2: If the question asks "how long", "how many days/weeks/months/years",
        "when", or "how much time" — call date_diff() with the exact dates from Step 1.
        Use {reference_date} as the end date if the question asks "since then" or "how long ago".
        Do NOT compute in your head.
Step 3: State your answer using the tool result.

If no dates are found in the facts: say "No date information available for [topic]."
"""

INF_PROMPT_TEMPORAL_NOTOOL = """You have memory facts about a person. Answer the question using ONLY these facts.

IMPORTANT: The reference date is {reference_date} (date of the last conversation session).
Do NOT use today's real-world date. All time calculations relative to "now" or "today" use {reference_date}.

{context}

Question: {question}
Short answer:
"""

INF_PROMPT_COUNTING = """You have memory facts about a person. Answer the question using ONLY these facts.

{context}

Question: {question}

COUNTING PROTOCOL — follow exactly:
Step 1: Find ALL facts with kind="count_item" plus any fact listing individual items.
Step 2: Write out EVERY item as an explicit list:
        Item 1: [name]
        Item 2: [name]
        ...
Step 3: Call count_items([list]) to get the exact count.
Step 4: Answer: "The total is [N]: [list]"

If totals conflict between facts — use count_items() on the explicit item list.
Do NOT guess or round.
"""

INF_PROMPT_SYNTHESIS = """You have memory facts about a person. Answer the question using ONLY these facts.

{context}

Question: {question}

PREFERENCE ANALYSIS PROTOCOL — follow exactly:
Step 1: Find ALL facts relevant to this topic: explicit preferences,
        behavioral patterns, repeated choices. List each:
        [fact] (confidence: explicit / behavioral / inferred)
Step 2: Look for patterns:
        - Explicit statement ("I prefer", "I love", "I hate") → certain
        - Repeated behavior (3+ times) → strong
        - Single mention → weak, flag as uncertain
Step 3: Synthesize into this EXACT format:
        "The user would prefer [X] because [evidence].
         They would not prefer [Y] / They may not be interested in [Z]."

CRITICAL OUTPUT RULES:
- Output MUST start with "The user would prefer"
- Do NOT output bullet points, numbered lists, or markdown tables
- Do NOT output "Step 1 / Step 2" headers in your answer
- One or two sentences maximum
- "Limited evidence suggests X" is better than "No information available"
"""

# ── Tool definitions ──

TEMPORAL_TOOLS = [
    {
        "name": "date_diff",
        "description": "Compute exact difference between two dates. Always use this for duration questions — never compute in your head.",
        "input_schema": {
            "type": "object",
            "properties": {
                "date1": {"type": "string", "description": "First date — ISO (2021-03-15) or natural language (March 2021, summer 2023, early 2022)"},
                "date2": {"type": "string", "description": "Second date — same formats"},
                "unit":  {"type": "string", "enum": ["days", "weeks", "months", "years"]}
            },
            "required": ["date1", "date2", "unit"]
        }
    }
]

COUNTING_TOOLS = [
    {
        "name": "count_items",
        "description": "Count a list of items exactly. Always use this for counting questions — never count in your head.",
        "input_schema": {
            "type": "object",
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}, "description": "List of items to count"}
            },
            "required": ["items"]
        }
    }
]

# ── get_more_context tool (Unit 9, run_23p.py lines 248-284) ──

GET_CONTEXT_TOOL = {
    "name": "get_more_context",
    "description": "Retrieve full raw text of a specific session. Use ONLY if facts and raw context don't contain enough detail.",
    "input_schema": {
        "type": "object",
        "properties": {
            "session_id": {"type": "integer", "description": "Session number (e.g. 12 for S12)"}
        },
        "required": ["session_id"]
    }
}


def get_more_context(session_id: int, raw_sessions: list = None) -> dict:
    """Return full text of a session. Truncated at 15K chars. (Unit 9, run_23p.py)"""
    if raw_sessions is None:
        raw_sessions = []
    if 0 < session_id <= len(raw_sessions):
        rs = raw_sessions[session_id - 1]
        # Check raw_session visibility (status field)
        if isinstance(rs, dict) and rs.get("status", "active") != "active":
            return {"result": f"Session {session_id} not found."}
        # Check TTL expiry on raw session
        if isinstance(rs, dict) and rs.get("retention_ttl") is not None:
            from datetime import datetime as _dt
            from datetime import timezone as _tz
            _now = _dt.now(_tz.utc)
            created = rs.get("stored_at") or rs.get("created_at")
            if created:
                try:
                    created_dt = _dt.fromisoformat(created.replace("Z", "+00:00"))
                    elapsed = (_now - created_dt).total_seconds()
                    if elapsed > rs["retention_ttl"]:
                        return {"result": f"Session {session_id} not found."}
                except (ValueError, TypeError):
                    pass
        # Reference mode degradation (Unit 12): if storage_mode is "reference"
        # and content is empty, return a degradation message.
        if isinstance(rs, dict):
            mode = rs.get("storage_mode", "inline")
            text = rs.get("content", "")
            if mode == "reference" and not text.strip():
                return {"result": f"Session {session_id} content unavailable "
                        f"(reference mode — original source not inline)."}
        else:
            text = str(rs)
        if len(text) > 15000:
            text = text[:15000] + "\n[...truncated]"
        return {"result": f"Full text of Session {session_id}:\n{text}"}
    return {"result": f"Session {session_id} not found."}


# ── Tool execution ──

TOOL_REGISTRY = {
    "date_diff":   date_diff,
    "count_items": count_items,
    "get_more_context": get_more_context,
}


def execute_tool(tool_name: str, tool_input: dict, context: dict = None) -> str:
    """Execute a tool call and return result as JSON string.

    context: extra kwargs injected by the caller (not from model).
    Used for get_more_context which needs raw_sessions from MemoryServer.
    """
    fn = TOOL_REGISTRY.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    merged = {**tool_input}
    if context:
        merged.update(context)
    result = fn(**merged)
    return json.dumps(result)


async def call_inference_with_tools(
    client,
    model: str,
    prompt: str,
    context: str,
    question: str,
    tools: list,
    max_tool_rounds: int = 2,
    reference_date: str = "2023-01-01",
    tool_context: dict = None,
    max_tokens: int = 512,
    return_metadata: bool = False,
    format_kwargs: dict = None,
) -> str | dict:
    """Run inference with tool round-trip support.

    If model issues tool_use → execute locally → send tool_result → get final answer.
    Max 2 tool rounds per question (safety limit).

    return_metadata: if True, return dict with "answer", "tool_called", "tool_results".
    format_kwargs: extra format kwargs for prompt (merged with context/question/reference_date).
    """
    fmt = {"context": context, "question": question, "reference_date": reference_date}
    if format_kwargs:
        fmt.update(format_kwargs)
    formatted_prompt = prompt.format(**fmt)

    # Non-Anthropic models: fall back to call_oai (no tool-use)
    if not model.startswith("anthropic/") and not hasattr(client, "messages"):
        from .common import _api_model, _get_client, call_oai
        answer = await call_oai(model, formatted_prompt, max_tokens=max_tokens)
        if return_metadata:
            return {"answer": answer, "tool_called": False, "tool_results": []}
        return answer

    messages = [{"role": "user", "content": formatted_prompt}]

    text_blocks = []
    tool_called = False
    all_tool_results = []
    for _ in range(max_tool_rounds + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            tools=tools,
            messages=messages,
        )

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_uses:
            break

        # Execute all tool calls
        tool_called = True
        tool_results = []
        for tu in tool_uses:
            result_str = execute_tool(tu.name, tu.input, context=tool_context)
            all_tool_results.append({"tool": tu.name, "input": tu.input, "result": result_str})
            # Audit raw-text access via tool_context
            if tu.name == "get_more_context" and tool_context and tool_context.get("audit"):
                tool_context["audit"].log("get_more_context",
                    caller_id=tool_context.get("caller_id", "unknown"),
                    details={"session_id": tu.input.get("session_id")})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_str,
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    answer = text_blocks[0].text.strip() if text_blocks else ""
    if return_metadata:
        return {"answer": answer, "tool_called": tool_called, "tool_results": all_tool_results}
    return answer


# ── Context building (for sprint-9 style artifact retrieval) ──

def build_context(top5, art_map):
    """Build context string from top-5 retrieval results.

    Args:
        top5: list of {"id": ..., "s": ...}
        art_map: dict mapping id -> artifact dict
    """
    parts = []
    for i, t in enumerate(top5, 1):
        art = art_map.get(t["id"])
        if art:
            date_str = art.get("updated_at", "")[:10]
            body = art.get("body", art.get("summary", ""))
            parts.append(f"[{i}] ({date_str}) {body}")
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Query-type-adaptive inference prompts (from sprint 23d)
# ═══════════════════════════════════════════════════════════════════════════
# All 8 query types loaded from src/prompts/inference/*.md.
# Each prompt expects {context} and {question} format placeholders.

_INF_PROMPT_DIR = _Path(__file__).parent / "prompts" / "inference"

_INF_PROMPT_TYPES = [
    "lookup", "temporal", "aggregate", "current",
    "synthesize", "procedural", "prospective", "summarize", "icl",
    "hybrid", "tool", "summarize_with_metadata", "list_set", "slot_query", "compositional",
]


def _load_inf_prompt(name: str) -> str:
    """Load inference prompt from .md file."""
    path = _INF_PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Inference prompt '{name}' not found at {path}. "
            f"Expected src/prompts/inference/{name}.md"
        )
    return path.read_text(encoding="utf-8")


INF_PROMPTS = {name: _load_inf_prompt(name) for name in _INF_PROMPT_TYPES}

# Fix Bug 2: retrieval returns query_type names that differ from inference
# prompt names.  This mapping bridges the two naming conventions.
_RETRIEVAL_TO_INF = {
    "default":      "lookup",
    "counting":     "aggregate",
    "temporal":     "temporal",
    "current":      "current",
    "rule":         "procedural",
    "synthesis":    "synthesize",
    "prospective":  "prospective",
    "summarize":    "summarize",
    "icl":          "icl",
    "supersession": "current",
}


@dataclass(frozen=True)
class InferenceLeafPlugin:
    """Isolated leaf prompt rule selected on top of a base prompt type."""

    name: str
    prompt_name: str
    base_prompt_types: tuple[str, ...]
    requires_enabled: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    priority: int = 0

    def matches(self, prompt_type: str, operator_plan: dict) -> bool:
        if prompt_type not in self.base_prompt_types:
            return False
        for op_name in self.requires_enabled:
            if not operator_plan.get(op_name, {}).get("enabled", False):
                return False
        for op_name in self.blocked_by:
            if operator_plan.get(op_name, {}).get("enabled", False):
                return False
        return True


INFERENCE_LEAF_PLUGINS: tuple[InferenceLeafPlugin, ...] = (
    InferenceLeafPlugin(
        name="slot_query",
        prompt_name="slot_query",
        base_prompt_types=("lookup", "hybrid", "synthesize", "synthesis"),
        requires_enabled=("slot_query",),
        blocked_by=("ordinal", "commonality", "compare_diff", "list_set", "bounded_chain", "local_anchor", "temporal_grounding"),
        priority=110,
    ),
    InferenceLeafPlugin(
        name="list_set",
        prompt_name="list_set",
        base_prompt_types=("lookup", "hybrid"),
        requires_enabled=("list_set",),
        blocked_by=("ordinal", "commonality", "compare_diff", "bounded_chain"),
        priority=100,
    ),
    InferenceLeafPlugin(
        name="compositional",
        prompt_name="compositional",
        base_prompt_types=("lookup", "hybrid", "synthesis", "synthesize"),
        requires_enabled=("compositional",),
        blocked_by=("list_set", "ordinal", "commonality", "compare_diff"),
        priority=90,
    ),
)

DEFAULT_INFERENCE_LEAF_PLUGIN_STATE = {
    plugin.name: True for plugin in INFERENCE_LEAF_PLUGINS
}


def resolve_inference_prompt_key(
    prompt_type: str,
    operator_plan: dict | None = None,
    *,
    plugin_state: dict[str, bool] | None = None,
) -> str:
    """Resolve the final inference prompt key after applying enabled leaf plugins."""

    operator_plan = operator_plan or {}
    canonical_prompt_type = _RETRIEVAL_TO_INF.get(prompt_type, prompt_type)
    state = dict(DEFAULT_INFERENCE_LEAF_PLUGIN_STATE)
    if plugin_state:
        state.update({str(name): bool(enabled) for name, enabled in plugin_state.items()})
    ordered_plugins = sorted(INFERENCE_LEAF_PLUGINS, key=lambda plugin: plugin.priority, reverse=True)
    for plugin in ordered_plugins:
        if not state.get(plugin.name, True):
            continue
        if plugin.matches(canonical_prompt_type, operator_plan):
            return plugin.prompt_name
    return canonical_prompt_type


def get_inf_prompt(query_type: str) -> str:
    """Return the inference prompt for a given query type.

    Maps retrieval query_type names to inference prompt names, then
    falls back to the 'lookup' prompt for unknown query types.
    """
    inf_name = _RETRIEVAL_TO_INF.get(query_type, query_type)
    return INF_PROMPTS.get(inf_name, INF_PROMPTS["lookup"])
