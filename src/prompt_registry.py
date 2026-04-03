"""gosh.memory — Librarian Prompt Registry.

Prompts-as-data for memory extraction.
Built-in prompts ship with the package.
Custom prompts persist to {data_dir}/librarian_prompts/{content_type}.md.
"""

from pathlib import Path

# ── Built-in prompts ──

BUILTIN_PROMPTS: dict[str, str] = {
    "default": """You are extracting structured atomic facts from a conversation.

Each fact must be a SELF-CONTAINED statement with EXACT values.
Extract: WHO did WHAT, with exact names, dates, numbers, places.
Convert relative time references to absolute using session_date={session_date}.
One fact per item. Extract every kind: fact, preference, decision, rejection,
requirement, constraint, rule, lesson_learned, action_item, action, count_item, observation.
""",

    "financial": """You are extracting financial facts from a conversation.

Focus exclusively on:
- Budget amounts, costs, expenses (exact figures and currencies)
- Financial decisions and approvals
- Spending constraints and limits
- Transactions and payments
- Revenue and profit figures
- Financial obligations and deadlines

Ignore non-financial content. Every fact must include exact amounts.
Session date: {session_date}.
""",

    "technical": """You are extracting technical facts from a conversation.

Focus on:
- Architecture and design decisions (with rationale and rejected alternatives)
- Technology choices: languages, frameworks, libraries, databases
- Infrastructure components and configurations
- API contracts and interfaces
- Performance characteristics and measurable limits
- Known bugs, limitations, and workarounds
- Build and deployment procedures

Session date: {session_date}.
""",

    "personal": """You are extracting personal facts from a conversation.

Focus on:
- Preferences and dislikes (food, hobbies, activities, media)
- Biographical facts (location, occupation, relationships, background)
- Habits and routines
- Goals and aspirations
- Health and lifestyle information

Extract only explicitly stated facts. Do not infer.
Session date: {session_date}.
""",

    "regulatory": """You are extracting regulatory and policy facts from a conversation.

Focus on:
- Rules and policies (explicit obligations and prohibitions)
- Compliance requirements
- Approval workflows and thresholds
- Deadlines and time constraints
- Responsible parties for each rule
- Penalties and consequences for violations

Every rule fact must state: WHAT is required/prohibited, WHO it applies to,
and any threshold or condition.
Session date: {session_date}.
""",

    "agent_trace": """You are extracting agent action facts from a conversation or trace log.

Focus on:
- Actions taken by the agent (tool calls, API requests, file operations)
- Results and outputs of each action (success/failure, return values)
- Errors encountered and how they were handled
- Decisions made during execution
- Resources created, modified, or deleted
- Task completion status

Each fact must specify: WHAT action, WHAT result, WHEN (if available).
Session date: {session_date}.
""",
}

BUILTIN_CONTENT_TYPES = list(BUILTIN_PROMPTS.keys())


class PromptRegistry:
    """Manages Librarian prompts by content_type.

    Resolution order:
      1. Custom prompt file in data_dir (if exists)
      2. Built-in prompt
      3. "default" built-in (fallback)
    """

    _UNSET = object()

    def __init__(self, data_dir: str, key: str = _UNSET):
        self._data_dir = Path(data_dir)
        self._key = key
        if key is None:
            # Explicit key=None: MAL direct-directory mode
            self._prompts_dir = self._data_dir / "prompts" / "conversation"
        elif key is not self._UNSET:
            self._prompts_dir = self._data_dir / "librarian_prompts" / key
        else:
            # key not provided: legacy behavior
            self._prompts_dir = self._data_dir / "librarian_prompts"

    def _custom_path(self, content_type: str) -> Path:
        return self._prompts_dir / f"{content_type}.md"

    def get(self, content_type: str) -> str:
        """Return prompt for content_type. Falls back to 'default' silently."""
        path = self._custom_path(content_type)
        if path.exists():
            return path.read_text(encoding="utf-8")
        if content_type in BUILTIN_PROMPTS:
            return BUILTIN_PROMPTS[content_type]
        return BUILTIN_PROMPTS["default"]

    def set(self, content_type: str, prompt: str) -> None:
        """Persist a custom prompt."""
        self._prompts_dir.mkdir(parents=True, exist_ok=True)
        self._custom_path(content_type).write_text(prompt, encoding="utf-8")

    def list(self) -> list[dict]:
        """List all available content types with source."""
        result = []
        for ct in BUILTIN_CONTENT_TYPES:
            path = self._custom_path(ct)
            source = "custom" if path.exists() else "builtin"
            result.append({"content_type": ct, "source": source})
        if self._prompts_dir.exists():
            for f in sorted(self._prompts_dir.iterdir()):
                if f.suffix == ".md":
                    ct = f.stem
                    if ct not in BUILTIN_CONTENT_TYPES:
                        result.append({"content_type": ct, "source": "custom"})
        return result

    def exists(self, content_type: str) -> bool:
        """Return True if content_type resolves to something other than default fallback."""
        return (self._custom_path(content_type).exists()
                or content_type in BUILTIN_PROMPTS)
