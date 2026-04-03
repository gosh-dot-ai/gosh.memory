#!/usr/bin/env python3

from src.config import MemoryConfig


def test_memory_config_reads_env_dynamically(monkeypatch):
    monkeypatch.delenv("GOSH_EXTRACTION_MODEL", raising=False)
    monkeypatch.delenv("GOSH_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("GOSH_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("GOSH_EMBED_MODEL", raising=False)
    monkeypatch.delenv("EXTRACTION_MODEL", raising=False)
    monkeypatch.delenv("INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)

    empty = MemoryConfig()
    assert empty.extraction_model == ""
    assert empty.inference_model == ""
    assert empty.judge_model == ""
    assert empty.embed_model == ""

    monkeypatch.setenv("GOSH_EXTRACTION_MODEL", "extract-x")
    monkeypatch.setenv("GOSH_INFERENCE_MODEL", "infer-y")
    monkeypatch.setenv("GOSH_JUDGE_MODEL", "judge-z")
    monkeypatch.setenv("GOSH_EMBED_MODEL", "embed-w")

    cfg = MemoryConfig()
    assert cfg.extraction_model == "extract-x"
    assert cfg.inference_model == "infer-y"
    assert cfg.judge_model == "judge-z"
    assert cfg.embed_model == "embed-w"


def test_memory_config_falls_back_to_legacy_env(monkeypatch):
    monkeypatch.delenv("GOSH_EXTRACTION_MODEL", raising=False)
    monkeypatch.delenv("GOSH_INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("GOSH_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("GOSH_EMBED_MODEL", raising=False)

    monkeypatch.setenv("EXTRACTION_MODEL", "extract-legacy")
    monkeypatch.setenv("INFERENCE_MODEL", "infer-legacy")
    monkeypatch.setenv("JUDGE_MODEL", "judge-legacy")
    monkeypatch.setenv("EMBED_MODEL", "embed-legacy")

    cfg = MemoryConfig()
    assert cfg.extraction_model == "extract-legacy"
    assert cfg.inference_model == "infer-legacy"
    assert cfg.judge_model == "judge-legacy"
    assert cfg.embed_model == "embed-legacy"


def test_memory_config_prefers_gosh_env_over_legacy(monkeypatch):
    monkeypatch.setenv("EXTRACTION_MODEL", "extract-legacy")
    monkeypatch.setenv("INFERENCE_MODEL", "infer-legacy")
    monkeypatch.setenv("JUDGE_MODEL", "judge-legacy")
    monkeypatch.setenv("EMBED_MODEL", "embed-legacy")

    monkeypatch.setenv("GOSH_EXTRACTION_MODEL", "extract-gosh")
    monkeypatch.setenv("GOSH_INFERENCE_MODEL", "infer-gosh")
    monkeypatch.setenv("GOSH_JUDGE_MODEL", "judge-gosh")
    monkeypatch.setenv("GOSH_EMBED_MODEL", "embed-gosh")

    cfg = MemoryConfig()
    assert cfg.extraction_model == "extract-gosh"
    assert cfg.inference_model == "infer-gosh"
    assert cfg.judge_model == "judge-gosh"
    assert cfg.embed_model == "embed-gosh"


def test_memory_config_from_args_reads_dynamic_env(monkeypatch):
    class Args:
        extraction_model = None
        inference_model = None
        judge_model = None
        embed_model = None
        model = None

    monkeypatch.setenv("GOSH_EXTRACTION_MODEL", "extract-x")
    monkeypatch.setenv("GOSH_INFERENCE_MODEL", "infer-y")
    monkeypatch.setenv("GOSH_JUDGE_MODEL", "judge-z")
    monkeypatch.setenv("GOSH_EMBED_MODEL", "embed-w")

    cfg = MemoryConfig.from_args(Args())
    assert cfg.extraction_model == "extract-x"
    assert cfg.inference_model == "infer-y"
    assert cfg.judge_model == "judge-z"
    assert cfg.embed_model == "embed-w"
