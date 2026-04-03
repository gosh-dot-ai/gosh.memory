"""MAL experiment atom validation — SPEC Section 2 + 17."""

EXCLUDED_FIELDS = {"km_bonus", "code_bonus", "step_bonus", "max_episodes_default"}

VALID_GROUPING_MODES = {"baseline", "strict_partition", "strict_small", "metadata_first"}

ATOM_SCHEMA = {
    "lexical_signal_bundle": {
        "fields": {"word_overlap_bonus", "number_overlap_bonus", "entity_phrase_bonus"},
        "mode": "retrieval-only",
    },
    "locality_bundle": {
        "fields": {"currentness_bonus", "generic_penalty"},
        "mode": "retrieval-only",
    },
    "window_bundle": {
        "fields": {"supporting_facts_per_episode", "supporting_facts_total", "budget"},
        "mode": "retrieval-only",
    },
    "fusion_bundle": {
        "fields": {"late_fusion_per_family", "max_sources_per_family", "rrf_k"},
        "mode": "retrieval-only",
    },
    "grouping_bundle": {
        "fields": {"grouping_prompt_mode", "size_cap_chars"},
        "mode": "reprocessing",
    },
    "extraction_example_append": {
        "fields": {"prompt_target", "example"},
        "mode": "extraction",
    },
    "inference_leaf_toggle": {
        "fields": {"plugin_name", "enabled"},
        "mode": "inference",
    },
    "extraction_model_switch": {
        "fields": {"model_id"},
        "mode": "extraction",
    },
}

VALID_INFERENCE_LEAF_NAMES = {"list_set", "compositional"}

VALID_PROMPT_TARGET_PREFIXES = (
    "conversation_content_type:",
    "document_block_prompt:",
    "document_source_aggregation_prompt:",
)

VALID_DOCUMENT_BLOCK_NAMES = {"prose_block", "list_block", "table_block", "fallback_block"}
VALID_SOURCE_AGG_NAMES = {"unified_source_aggregation", "unified_source_aggregation_repair"}


class AtomValidator:

    def validate(self, atom: dict) -> bool:
        atom_type = atom.get("atom_type", "")
        payload = atom.get("atom_payload", {})

        if atom_type not in ATOM_SCHEMA:
            raise ValueError(f"unknown atom_type: {atom_type}")

        if not payload:
            raise ValueError("empty atom_payload")

        schema = ATOM_SCHEMA[atom_type]

        if atom_type == "extraction_example_append":
            return self._validate_extraction(payload)

        if atom_type == "extraction_model_switch":
            return self._validate_model_switch(payload)

        if atom_type == "inference_leaf_toggle":
            return self._validate_inference_leaf(payload)

        return self._validate_parameter_atom(atom_type, payload, schema)

    def _validate_model_switch(self, payload: dict) -> bool:
        if "model_id" not in payload:
            raise ValueError("extraction_model_switch requires model_id")
        model_id = payload["model_id"]
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string")
        extra = set(payload.keys()) - {"model_id"}
        if extra:
            raise ValueError(f"invalid fields in extraction_model_switch: {extra}")
        return True

    def _validate_inference_leaf(self, payload: dict) -> bool:
        allowed = {"plugin_name", "enabled"}
        extra = set(payload.keys()) - allowed
        if extra:
            raise ValueError(f"invalid fields in inference_leaf_toggle atom: {extra}")
        if "plugin_name" not in payload or "enabled" not in payload:
            raise ValueError("inference_leaf_toggle requires plugin_name and enabled")
        if payload["plugin_name"] not in VALID_INFERENCE_LEAF_NAMES:
            raise ValueError(f"unknown inference leaf plugin: {payload['plugin_name']}")
        if not isinstance(payload["enabled"], bool):
            raise ValueError("enabled must be a boolean")
        return True

    def _validate_extraction(self, payload: dict) -> bool:
        allowed = {"prompt_target", "example"}
        extra = set(payload.keys()) - allowed
        if extra:
            raise ValueError(f"invalid fields in extraction atom: {extra}")
        if "prompt_target" not in payload or "example" not in payload:
            raise ValueError("extraction atom requires prompt_target and example")

        target = payload["prompt_target"]
        if not any(target.startswith(p) for p in VALID_PROMPT_TARGET_PREFIXES):
            raise ValueError(f"invalid prompt_target: {target}")

        if target.startswith("document_block_prompt:"):
            name = target.split(":", 1)[1]
            if name not in VALID_DOCUMENT_BLOCK_NAMES:
                raise ValueError(f"invalid document block prompt name: {name}")
        elif target.startswith("document_source_aggregation_prompt:"):
            name = target.split(":", 1)[1]
            if name not in VALID_SOURCE_AGG_NAMES:
                raise ValueError(f"invalid source aggregation prompt name: {name}")

        return True

    def _validate_parameter_atom(self, atom_type: str, payload: dict, schema: dict) -> bool:
        allowed_fields = schema["fields"]

        for field in payload:
            if field in EXCLUDED_FIELDS:
                raise ValueError(f"excluded field: {field}")
            if field not in allowed_fields:
                raise ValueError(f"invalid field '{field}' for atom_type '{atom_type}'")

        for field, val in payload.items():
            if atom_type == "grouping_bundle":
                self._validate_grouping_field(field, val)
            else:
                self._validate_numeric_field(field, val)

        return True

    def _validate_numeric_field(self, field: str, val: dict) -> None:
        old = val.get("old")
        new = val.get("new")
        if old is None or new is None:
            raise ValueError(f"missing old or new value for {field}")
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            raise ValueError(f"non-numeric old/new for {field}")
        # Absolute floor: old=0 means the field is untunable from zero
        if old == 0:
            raise ValueError(f"cannot tune {field} from old=0 (untunable base)")
        lower = old * 0.3
        upper = old * 3.0
        if not (lower <= new <= upper):
            raise ValueError(f"bounds violation for {field}: {new} not in [{lower}, {upper}]")

    def _validate_grouping_field(self, field: str, val: dict) -> None:
        old = val.get("old")
        new = val.get("new")
        if field == "grouping_prompt_mode":
            if new not in VALID_GROUPING_MODES:
                raise ValueError(f"invalid grouping_prompt_mode: {new}")
        elif field == "size_cap_chars":
            if new is not None and not (4000 <= new <= 50000):
                raise ValueError(f"bounds violation for size_cap_chars: {new}")
