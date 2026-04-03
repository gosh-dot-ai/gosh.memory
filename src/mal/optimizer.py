"""MAL optimizer — LLM-based hypothesis proposal with history awareness.

Proposes one atom per run. Uses LLM to reason about what to try,
reads rejected history to avoid repeating failed experiments.
Falls back to rule-based when no LLM available.
"""

import asyncio
import json
import logging

from .atom import (
    ATOM_SCHEMA,
    EXCLUDED_FIELDS,
    VALID_GROUPING_MODES,
    VALID_INFERENCE_LEAF_NAMES,
)

log = logging.getLogger(__name__)


def _clamp(value, old, lo_factor=0.3, hi_factor=3.0):
    lo = old * lo_factor
    hi = old * hi_factor
    return round(max(lo, min(hi, value)), 4)


_RETRIEVAL_DEFAULTS = {
    "word_overlap_bonus": 0.45,
    "number_overlap_bonus": 1.4,
    "entity_phrase_bonus": 2.0,
    "currentness_bonus": 0.8,
    "generic_penalty": 1.2,
    "supporting_facts_per_episode": 3,
    "supporting_facts_total": 10,
    "budget": 8000,
    "late_fusion_per_family": 3,
    "max_sources_per_family": 2,
    "rrf_k": 60,
}

_SIGNATURE_FIELD_RELEVANCE = {
    "entity": ["entity_phrase_bonus", "word_overlap_bonus"],
    "phrase": ["entity_phrase_bonus", "word_overlap_bonus"],
    "overlap": ["word_overlap_bonus", "number_overlap_bonus"],
    "number": ["number_overlap_bonus"],
    "currentness": ["currentness_bonus"],
    "recency": ["currentness_bonus"],
    "generic": ["generic_penalty"],
    "penalty": ["generic_penalty"],
    "fusion": ["rrf_k", "late_fusion_per_family", "max_sources_per_family"],
    "cross": ["late_fusion_per_family", "max_sources_per_family"],
    "budget": ["budget", "supporting_facts_total"],
    "support": ["supporting_facts_per_episode", "supporting_facts_total"],
    "facts": ["supporting_facts_per_episode", "supporting_facts_total"],
    "window": ["supporting_facts_per_episode", "supporting_facts_total", "budget"],
}

_FIELD_TO_BUNDLE = {
    "word_overlap_bonus": "lexical_signal_bundle",
    "number_overlap_bonus": "lexical_signal_bundle",
    "entity_phrase_bonus": "lexical_signal_bundle",
    "currentness_bonus": "locality_bundle",
    "generic_penalty": "locality_bundle",
    "supporting_facts_per_episode": "window_bundle",
    "supporting_facts_total": "window_bundle",
    "budget": "window_bundle",
    "late_fusion_per_family": "fusion_bundle",
    "max_sources_per_family": "fusion_bundle",
    "rrf_k": "fusion_bundle",
}

_OPTIMIZER_SYSTEM_PROMPT = """You are the MAL optimizer for a memory retrieval pipeline.

Your job: propose ONE experiment atom that will improve retrieval quality
for the diagnosed failure pattern.

Rules:
- Return ONLY a JSON object: {"atom_type": "...", "atom_payload": {...}}
- For retrieval-only atoms, payload fields use {"old": number, "new": number}
- Numeric bounds: new must be in [old * 0.3, old * 3.0]
- Do NOT repeat atoms from the rejected history
- Do NOT use excluded fields
- Pick the field most likely to fix the diagnosed failure
- One field per atom (single-field degenerate atoms preferred for isolation)
"""


class Optimizer:

    def propose(self, mode: str, family: dict, current_state: dict,
                snapshot=None, rejected_history: list[dict] = None,
                call_llm=None) -> dict:
        """Propose one experiment atom.

        call_llm: async fn(model, system, user_msg, max_tokens) -> raw
            When provided, LLM proposes the atom. Falls back to
            rule-based on LLM failure or when call_llm is None.
        """
        rejected_history = rejected_history or []

        if call_llm is not None:
            try:
                result = self._propose_via_llm(
                    mode, family, current_state, rejected_history, call_llm,
                )
                if result is not None:
                    return result
            except Exception as exc:
                log.warning("LLM optimizer failed, falling back to rules: %s", exc)

        return self._propose_rule_based(mode, family, current_state, rejected_history)

    def _propose_via_llm(self, mode, family, current_state, rejected_history, call_llm):
        """Use LLM to propose an atom."""
        user_msg = _build_optimizer_context(mode, family, current_state, rejected_history)

        # call_llm is async — run it
        loop = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                raw = pool.submit(asyncio.run, call_llm(
                    None, _OPTIMIZER_SYSTEM_PROMPT, user_msg, 2048,
                )).result(timeout=30)
        else:
            raw = asyncio.run(call_llm(
                None, _OPTIMIZER_SYSTEM_PROMPT, user_msg, 2048,
            ))

        return _parse_llm_atom(raw)

    def _propose_rule_based(self, mode, family, current_state, rejected_history):
        """Rule-based proposal. Strategy order:
        1. Try different extraction model (if zero facts / extraction failure)
        2. Tune retrieval parameters
        3. Change extraction prompts
        4. Toggle inference leaf plugins
        5. Change grouping
        """
        sig = family.get("signature", "")

        # Strategy 1: model switch — first thing to try when extraction fails
        if self._should_try_model_switch(sig, current_state, rejected_history):
            return self._propose_model_switch(current_state, rejected_history)

        if mode == "reprocessing":
            return self._propose_reprocessing(family, current_state, rejected_history)
        if mode == "extraction":
            return self._propose_extraction(family, current_state, rejected_history)
        if mode == "inference":
            return self._propose_inference_leaf(family, current_state, rejected_history)
        return self._propose_retrieval(family, current_state, rejected_history)

    def _should_try_model_switch(self, sig, current_state, rejected_history):
        """Model switch is first strategy when extraction produces zero facts."""
        zero_fact_signals = {"zero_facts_extracted", "no_extraction_path", "extraction_failed"}
        if sig in zero_fact_signals:
            # Check if model switch was already rejected
            for art in (rejected_history or []):
                if art.get("atom_type") == "extraction_model_switch":
                    return False
            return True
        return False

    def _propose_model_switch(self, current_state, rejected_history):
        """Propose switching extraction model. Try known good alternatives."""
        current_model = current_state.get("extraction_model", "")
        rejected_models = {
            art.get("atom_payload", {}).get("model_id")
            for art in (rejected_history or [])
            if art.get("atom_type") == "extraction_model_switch"
        }

        # Known extraction models in priority order
        candidates = [
            "qwen/qwen3-32b",
            "anthropic/claude-sonnet-4-6",
            "gpt-4.1-mini",
        ]
        for model_id in candidates:
            if model_id != current_model and model_id not in rejected_models:
                return {
                    "atom_type": "extraction_model_switch",
                    "atom_payload": {"model_id": model_id},
                }
        # All candidates tried — fall back
        return {
            "atom_type": "extraction_model_switch",
            "atom_payload": {"model_id": candidates[0]},
        }

    def _propose_retrieval(self, family, current_state, rejected_history):
        overrides = current_state.get("selector_config_overrides", {})
        rejected_fields = self._rejected_fields(rejected_history)
        sig = family.get("signature", "")

        candidate_fields = self._signature_relevant_fields(sig)
        candidate_fields = [f for f in candidate_fields if f not in rejected_fields]

        if not candidate_fields:
            all_fields = [f for f in _RETRIEVAL_DEFAULTS if f not in EXCLUDED_FIELDS]
            candidate_fields = [f for f in all_fields if f not in rejected_fields]

        if not candidate_fields:
            candidate_fields = list(_RETRIEVAL_DEFAULTS.keys())

        field = candidate_fields[0]
        old_val = overrides.get(field, _RETRIEVAL_DEFAULTS[field])
        bundle = _FIELD_TO_BUNDLE[field]

        if field in ("generic_penalty", "rrf_k"):
            new_val = _clamp(old_val * 0.75, old_val)
        else:
            new_val = _clamp(old_val * 1.5, old_val)

        return {
            "atom_type": bundle,
            "atom_payload": {field: {"old": old_val, "new": new_val}},
        }

    def _propose_reprocessing(self, family, current_state, rejected_history):
        current_mode = current_state.get("grouping_prompt_mode", "strict_small")
        rejected_modes = set()
        for art in rejected_history:
            payload = art.get("atom_payload", {})
            if payload.get("grouping_prompt_mode", {}).get("new"):
                rejected_modes.add(payload["grouping_prompt_mode"]["new"])

        modes = sorted(VALID_GROUPING_MODES)
        candidates = [m for m in modes if m != current_mode and m not in rejected_modes]
        if not candidates:
            candidates = [m for m in modes if m != current_mode]
        next_mode = candidates[0] if candidates else modes[0]

        return {
            "atom_type": "grouping_bundle",
            "atom_payload": {
                "grouping_prompt_mode": {"old": current_mode, "new": next_mode},
            },
        }

    def _propose_extraction(self, family, current_state, rejected_history):
        hint = family.get("signature", "missing_fact_support")
        rejected_targets = set()
        for art in rejected_history:
            payload = art.get("atom_payload", {})
            if payload.get("prompt_target"):
                rejected_targets.add(payload["prompt_target"])

        target = "conversation_content_type:default"
        if target in rejected_targets:
            target = "document_block_prompt:prose_block"
        if target in rejected_targets:
            target = "document_source_aggregation_prompt:unified_source_aggregation"

        return {
            "atom_type": "extraction_example_append",
            "atom_payload": {
                "prompt_target": target,
                "example": (
                    f"# Extraction improvement for {hint}\n"
                    f"When encountering '{hint}' patterns, ensure all relevant "
                    f"entities, relationships, and factual details are extracted."
                ),
            },
        }

    def _propose_inference_leaf(self, family, current_state, rejected_history):
        current_overrides = current_state.get("inference_leaf_plugin_overrides", {})
        rejected_toggles = set()
        for art in rejected_history:
            payload = art.get("atom_payload", {})
            if payload.get("plugin_name"):
                rejected_toggles.add((payload["plugin_name"], payload.get("enabled")))

        for name in sorted(VALID_INFERENCE_LEAF_NAMES):
            current_enabled = current_overrides.get(name, True)
            new_enabled = not current_enabled
            if (name, new_enabled) not in rejected_toggles:
                return {
                    "atom_type": "inference_leaf_toggle",
                    "atom_payload": {"plugin_name": name, "enabled": new_enabled},
                }

        name = sorted(VALID_INFERENCE_LEAF_NAMES)[0]
        return {
            "atom_type": "inference_leaf_toggle",
            "atom_payload": {"plugin_name": name, "enabled": not current_overrides.get(name, True)},
        }

    def _signature_relevant_fields(self, signature):
        sig_lower = signature.lower()
        fields = []
        seen = set()
        for keyword, field_list in _SIGNATURE_FIELD_RELEVANCE.items():
            if keyword in sig_lower:
                for f in field_list:
                    if f not in seen and f not in EXCLUDED_FIELDS:
                        fields.append(f)
                        seen.add(f)
        if not fields:
            fields = ["entity_phrase_bonus"]
        return fields

    def _rejected_fields(self, rejected_history):
        rejected = set()
        for art in rejected_history:
            payload = art.get("atom_payload", {})
            for key in payload:
                if key not in ("prompt_target", "example", "plugin_name", "enabled",
                               "grouping_prompt_mode", "size_cap_chars"):
                    rejected.add(key)
        return rejected


def _build_optimizer_context(mode, family, current_state, rejected_history):
    """Build user message for LLM-based atom proposal."""
    overrides = current_state.get("selector_config_overrides", {})

    # Show current values for all tunable fields
    current_values = {}
    for field, default in _RETRIEVAL_DEFAULTS.items():
        current_values[field] = overrides.get(field, default)

    rejected_summary = []
    for art in rejected_history[-10:]:
        rejected_summary.append({
            "atom_type": art.get("atom_type"),
            "atom_payload": art.get("atom_payload"),
            "score_delta": round(
                (art.get("score_after", {}).get("episode_hit_rate", 0)
                 - art.get("score_before", {}).get("episode_hit_rate", 0)), 4
            ),
        })

    context = {
        "failure_family": family,
        "mode": mode,
        "current_selector_values": current_values,
        "current_grouping_mode": current_state.get("grouping_prompt_mode", "strict_small"),
        "current_size_cap": current_state.get("size_cap_chars", 12000),
        "current_leaf_overrides": current_state.get("inference_leaf_plugin_overrides", {}),
        "allowed_retrieval_bundles": {
            "lexical_signal_bundle": ["word_overlap_bonus", "number_overlap_bonus", "entity_phrase_bonus"],
            "locality_bundle": ["currentness_bonus", "generic_penalty"],
            "window_bundle": ["supporting_facts_per_episode", "supporting_facts_total", "budget"],
            "fusion_bundle": ["late_fusion_per_family", "max_sources_per_family", "rrf_k"],
        },
        "excluded_fields": sorted(EXCLUDED_FIELDS),
        "recently_rejected": rejected_summary,
        "n_rejected_total": len(rejected_history),
    }
    return json.dumps(context, indent=2)


def _parse_llm_atom(raw) -> dict | None:
    """Parse LLM response into a valid atom dict."""
    if isinstance(raw, dict):
        obj = raw
    elif isinstance(raw, str):
        try:
            obj = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    else:
        return None

    if "atom_type" not in obj or "atom_payload" not in obj:
        return None
    if obj["atom_type"] not in ATOM_SCHEMA:
        return None
    return {"atom_type": obj["atom_type"], "atom_payload": obj["atom_payload"]}
