#!/usr/bin/env python3
"""gosh.memory runtime configuration.

Single source of truth for model selection.
Override via environment variables or programmatically.
"""

import os
from dataclasses import dataclass, field

LEGACY_ENV_ALIASES = {
    "GOSH_EXTRACTION_MODEL": "EXTRACTION_MODEL",
    "GOSH_INFERENCE_MODEL": "INFERENCE_MODEL",
    "GOSH_JUDGE_MODEL": "JUDGE_MODEL",
    "GOSH_EMBED_MODEL": "EMBED_MODEL",
}


def _env_default(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value
    legacy_name = LEGACY_ENV_ALIASES.get(name)
    if legacy_name:
        return os.getenv(legacy_name, "")
    return ""


@dataclass
class MemoryConfig:
    """Runtime configuration for gosh.memory.

    Controls which model is used for each pipeline stage.
    Stages are independent — mix providers freely.

    Example (production):
        cfg = MemoryConfig()  # reads from env

    Example (experiment):
        cfg = MemoryConfig(inference_model="anthropic/claude-sonnet-4-6")

    Example (CLI):
        cfg = MemoryConfig.from_args(args)
    """
    extraction_model: str = field(default_factory=lambda: _env_default("GOSH_EXTRACTION_MODEL"))
    inference_model:  str = field(default_factory=lambda: _env_default("GOSH_INFERENCE_MODEL"))
    judge_model:      str = field(default_factory=lambda: _env_default("GOSH_JUDGE_MODEL"))
    embed_model:      str = field(default_factory=lambda: _env_default("GOSH_EMBED_MODEL"))

    def summary(self) -> str:
        return (
            f"extraction={self.extraction_model} | "
            f"inference={self.inference_model} | "
            f"judge={self.judge_model} | "
            f"embed={self.embed_model}"
        )

    @classmethod
    def from_args(cls, args) -> "MemoryConfig":
        """Build from argparse Namespace. Individual flags override --model shortcut."""
        base = getattr(args, "model", None)
        return cls(
            extraction_model=getattr(args, "extraction_model", None) or base or _env_default("GOSH_EXTRACTION_MODEL"),
            inference_model =getattr(args, "inference_model",  None) or base or _env_default("GOSH_INFERENCE_MODEL"),
            judge_model     =getattr(args, "judge_model",      None) or base or _env_default("GOSH_JUDGE_MODEL"),
            embed_model     =getattr(args, "embed_model",      None) or _env_default("GOSH_EMBED_MODEL"),
        )
