#!/usr/bin/env python3
"""Unified transport-only ingest entrypoint."""

from __future__ import annotations

from .source_detect import detect_source_family
from .source_loader import load_source


async def ingest_input(
    server,
    text: str | None = None,
    path: str | None = None,
    url: str | None = None,
    metadata: dict | None = None,
    retention_ttl: int | None = None,
    target=None,
    agent_id: str | None = None,
    swarm_id: str | None = None,
    scope: str | None = None,
    source_id: str | None = None,
    **kwargs,
) -> dict:
    """Unified ingest: load -> detect -> route."""
    loaded = await load_source(text=text, path=path, url=url)
    family, evidence = detect_source_family(
        loaded.raw_text,
        filename=loaded.filename,
        mime=loaded.mime,
        is_directory=loaded.is_directory,
        is_repo=loaded.is_repo,
    )
    signals = list(evidence.get("signals", []))
    has_any_conversation_fields = any(
        kwargs.get(name) is not None
        for name in ("session_num", "session_date", "speakers")
    )
    if has_any_conversation_fields:
        if "conversation_fields_present" not in signals:
            signals.append("conversation_fields_present")
        evidence = {**evidence, "signals": signals}

    if family == "conversation":
        session_num = kwargs.get("session_num")
        if session_num is None:
            session_nums = [
                rs.get("session_num", 0)
                for rs in getattr(server, "_raw_sessions", [])
                if isinstance(rs, dict) and isinstance(rs.get("session_num"), int)
            ]
            session_num = (max(session_nums) if session_nums else 0) + 1
            evidence.setdefault("signals", []).append("auto_session_num")
        session_date = kwargs.get("session_date") or ""
        speakers = kwargs.get("speakers") or "User and Assistant"
        result = await server.store(
            content=loaded.raw_text,
            session_num=session_num,
            session_date=session_date,
            speakers=speakers,
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            metadata=metadata,
            retention_ttl=retention_ttl,
            target=target,
            source_id=source_id or loaded.filename,
        )
    elif family == "document":
        sid = source_id or loaded.filename or "doc"
        facts_extracted = await server.ingest_document(
            content=loaded.raw_text,
            source_id=sid,
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            metadata=metadata,
            retention_ttl=retention_ttl,
            target=target,
            source_meta={
                "ingest_transport": loaded.transport,
                "ingest_locator": loaded.locator,
                "ingest_mime": loaded.mime,
            },
        )
        result = {"facts_extracted": facts_extracted, "source_id": sid}
    elif family in ("codebase", "media"):
        raise ValueError(
            f"Source family '{family}' is not yet supported. "
            "Only conversation and document are implemented."
        )
    else:
        raise ValueError(f"unknown source family: {family}")

    result["source_family"] = family
    result["detection_evidence"] = evidence
    result["transport"] = loaded.transport
    result["locator"] = loaded.locator
    return result
