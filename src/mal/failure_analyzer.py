"""MAL failure analyzer — diagnose broken stage, select mode, detect CODE_REQUIRED."""

STAGE_ORDER = [
    "first_pass", "cross-contamination", "late_fusion",
    "query_operators", "packet", "episodes", "facts",
]

STAGE_TO_MODE = {
    "first_pass": "retrieval-only",
    "cross-contamination": "retrieval-only",
    "late_fusion": "retrieval-only",
    "query_operators": "retrieval-only",
    "packet": "retrieval-only",
    "episodes": "reprocessing",
    "facts": "extraction",
}

# Signatures that indicate the failure is beyond MAL's tunable surface
# and requires new code (leaf plugin, extractor, etc.)
# Signatures where model switch should be tried FIRST before CODE_REQUIRED
MODEL_SWITCH_FIRST_SIGNATURES = {
    "zero_facts_extracted",
}

# Signatures that are immediately CODE_REQUIRED (no model can help)
CODE_REQUIRED_SIGNATURES = {
    "unsupported_content_format",
    "no_extraction_path",
    "unrecognized_source_family",
    "missing_tool_trace_support",
    "structured_trace_unhandled",
}


class FailureAnalyzer:

    def diagnose(self, trace: dict) -> dict:
        stages = trace.get("stages", {})
        earliest_stage = None
        earliest_detail = "unknown"

        for stage in STAGE_ORDER:
            stage_info = stages.get(stage, {})
            if stage_info.get("status") == "failed":
                earliest_stage = stage
                earliest_detail = stage_info.get("detail", "unknown")
                break

        if earliest_stage is None:
            earliest_stage = "first_pass"

        operator = trace.get("query_type", "mixed_shape")

        return {
            "stage": earliest_stage,
            "operator_class_or_shape": operator,
            "signature": earliest_detail,
        }

    def derive_family_key(self, family: dict) -> str:
        return f"{family['stage']}|{family['operator_class_or_shape']}|{family['signature']}"

    def should_try_model_switch(self, family: dict, rejected_history: list[dict] = None) -> bool:
        """Check if model switch should be tried before CODE_REQUIRED."""
        sig = family.get("signature", "")
        if sig not in MODEL_SWITCH_FIRST_SIGNATURES:
            return False
        # Only if no model switch has been accepted yet
        for art in (rejected_history or []):
            if art.get("atom_type") == "extraction_model_switch" and art.get("status") == "applied":
                return False
        return True

    def is_code_required(self, family: dict, rejected_history: list[dict] = None) -> bool:
        """Determine if the failure requires new code, not parameter tuning.

        Returns True when:
        - Signature explicitly indicates missing pipeline support
        - OR the same family has been rejected >= 5 times
        - BUT NOT if model switch should be tried first
        """
        sig = family.get("signature", "")

        # Model-switch-first signatures: only CODE_REQUIRED after model switch exhausted
        if sig in MODEL_SWITCH_FIRST_SIGNATURES:
            model_switches_tried = sum(
                1 for art in (rejected_history or [])
                if art.get("atom_type") == "extraction_model_switch"
            )
            return model_switches_tried >= 3  # all candidate models tried

        if sig in CODE_REQUIRED_SIGNATURES:
            return True

        if rejected_history:
            family_key = self.derive_family_key(family)
            same_family_rejections = sum(
                1 for art in rejected_history
                if art.get("failure_family_key") == family_key
            )
            if same_family_rejections >= 5:
                return True

        return False

    def build_code_request(self, family: dict, family_key: str) -> dict:
        """Build the courier task payload for agent_id='coding'."""
        return {
            "task_type": "create_leaf_plugin",
            "agent_id": "coding",
            "cluster_id": family_key,
            "requested_by": "mal",
            "scope": "sandbox_only",
            "suspected_family": family.get("signature", "unknown"),
            "problem_statement": (
                f"Failure cluster '{family_key}' cannot be resolved by parameter tuning. "
                f"Stage: {family.get('stage', 'unknown')}, "
                f"operator: {family.get('operator_class_or_shape', 'unknown')}, "
                f"signature: {family.get('signature', 'unknown')}. "
                f"Add a leaf plugin, extraction path, or prompt that addresses this failure "
                f"without regressing existing passing cases."
            ),
            "deliverables": [
                "leaf manifest",
                "prompt file",
                "runtime wiring if required",
                "regression tests",
            ],
            "allowed_write_scope": ["src/", "tests/", "src/prompts/"],
            "acceptance_tests": [
                "target cluster improves on snapshot eval",
                "no regression on guard set",
                "new tests pass",
            ],
        }

    def select_mode(self, family: dict, trace: dict) -> str:
        stage = family["stage"]
        mode = STAGE_TO_MODE.get(stage, "retrieval-only")

        if mode == "reprocessing":
            source_families = trace.get("source_families", [])
            if source_families == ["conversation"]:
                return "MODE_RESTRICTED"

        return mode
