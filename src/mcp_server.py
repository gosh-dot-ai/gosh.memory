#!/usr/bin/env python3
"""GOSH Memory MCP Server.

Exposes MemoryServer + Courier over MCP (HTTP + SSE transport).
Two channels:
- Tools: request/response via POST /mcp
- SSE push: Courier subscriptions via GET /mcp/sse
"""

import argparse
import asyncio
import json
import logging
import os
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

from src.config import MemoryConfig
from src.courier import Courier
from src.memory import MemoryServer, _is_visible

log = logging.getLogger(__name__)


def _safe_tool(fn):
    """Wrap MCP tool handler with structured error handling + logging."""
    import functools

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except ValueError as e:
            return {"error": str(e), "code": "VALIDATION_ERROR", "tool": fn.__name__}
        except Exception as e:
            tool_name = fn.__name__
            log.error("MCP tool %s failed: %s", tool_name, e, exc_info=True)
            return {"error": str(e), "code": "INTERNAL_ERROR", "tool": tool_name}

    return wrapper


# ── Identity context ──

@dataclass
class ConnectionContext:
    """Resolved caller identity for an MCP tool call."""
    owner_id: str = "system"
    agent_id: str = "default"
    swarm_id: str = "default"
    caller_role: str = "user"  # "user" | "admin"
    memberships: list[str] = field(default_factory=list)


# Token→identity map. Empty in production — real MCP OAuth populates
# verified identity via connection context, not a local dict.
_TOKEN_IDENTITIES: dict[str, str] = {}

ADMIN_TOKEN = os.environ.get("GOSH_MEMORY_ADMIN_TOKEN", "")

# ── Verified auth adapter (R35) ──
# Pluggable resolver: token → canonical owner_id, or None if unverified.
# Real MCP OAuth implementation sets this at startup.
# Default: None (no verified auth configured — falls back to param-derived).
_verified_auth_resolver = None  # Optional[Callable[[str], Optional[str]]]


def _resolve_identity(
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> ConnectionContext:
    """Resolve caller identity from verified auth / params.

    Priority: admin token > verified MCP auth (token) > agent_key > param-derived.

    When a token is present it represents an auth attempt — secondary sources
    (agent_key, agent_id) must NOT override it, even if the token is unknown.
    """
    ctx = ConnectionContext(swarm_id=swarm_id)
    ctx.agent_id = agent_id

    # 0. Admin token (env-based, always checked first)
    if token and ADMIN_TOKEN and token == ADMIN_TOKEN:
        ctx.owner_id = "system"
        ctx.caller_role = "admin"
        return ctx

    # 1. Verified MCP auth (R35) — pluggable adapter resolves token
    #    to canonical owner_id. Takes precedence over agent_key/agent_id.
    #    Unrecognised or unverified tokens → system (no impersonation).
    if token:
        if _verified_auth_resolver is not None:
            verified_id = _verified_auth_resolver(token)
            if verified_id is not None:
                # Validate canonical form (user:/agent:/swarm: or bare system/anonymous)
                if (verified_id in ("system", "anonymous")
                        or any(verified_id.startswith(p) for p in ("user:", "agent:", "swarm:"))):
                    ctx.owner_id = verified_id
                else:
                    log.warning("Auth resolver returned non-canonical identity: %s", verified_id)
                    ctx.owner_id = "system"
                return ctx
        # Token present but unverified — block fallback to param-derived
        ctx.owner_id = "system"
        return ctx

    # 2. X-Agent-Key header (no auth token present)
    if agent_key:
        ctx.owner_id = agent_key
        return ctx

    # 3. Param-derived (no auth, no agent_key)
    if agent_id != "default":
        ctx.owner_id = f"agent:{agent_id}"
    else:
        ctx.owner_id = "system"

    # Backward compat: implicitly add swarm membership from swarm_id param
    # so scope-based "swarm-shared" facts are accessible via ACL
    if swarm_id and swarm_id != "default":
        ctx.memberships.append(f"swarm:{swarm_id}")

    return ctx


# ── Token authentication ──

SERVER_TOKEN = os.environ.get("GOSH_MEMORY_TOKEN", secrets.token_urlsafe(32))

# ── Module-level state ──

mcp = FastMCP(name="gosh-memory", streamable_http_path="/mcp")

registry: dict[str, MemoryServer] = {}
courier_registry: dict[str, Courier] = {}
connections: dict[str, asyncio.Queue] = {}
sub_to_conn: dict[str, str] = {}
_registry_lock = threading.Lock()
_instance_config_lock = threading.Lock()
# C2: track active SSE connections for hijack prevention
_active_connections: dict[str, str] = {}  # connection_id -> remote address or session id

data_dir: str = "./data"
cfg: MemoryConfig = MemoryConfig()
_write_log_worker_task: asyncio.Task | None = None

VALID_SCOPES = {"agent-private", "swarm-shared", "system-wide"}


def _get_memory(key: str) -> MemoryServer:
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be non-empty")
    server = registry.get(key)
    if server is not None:
        return server
    with _registry_lock:
        server = registry.get(key)
        if server is not None:
            return server
        tier_mode = os.environ.get("GOSH_MEMORY_TIER_MODE", "eager")
        server = MemoryServer(
            data_dir=data_dir,
            key=key,
            extract_model=cfg.extraction_model,
            tier_mode=tier_mode,
        )
        registry[key] = server
        return server


def _ensure_instance_config(server: MemoryServer, owner_id: str) -> bool:
    """Create instance config on first MCP write if not exists. Returns True if created."""
    with _instance_config_lock:
        if server._instance_config is not None:
            return False
        server._instance_config = {
            "owner_id": owner_id,
            "read": [],
            "_derived_read": [],
            "_derived_write": [],
            "write": [],
        }
        return True


def _expand_instance_read_for_scope(
    server: MemoryServer,
    *,
    scope: str,
    swarm_id: str,
) -> bool:
    """Mirror shared/public fact visibility at the instance read gate.

    Instance ACL stays owner-only by default. When the owner writes shared/public
    content, widen only the read gate so collaborators can reach the visible
    facts without implicitly granting write access to the whole memory key.
    """
    cfg = getattr(server, "_instance_config", None)
    if not cfg:
        return False

    if scope == "system-wide" or (scope == "swarm-shared" and swarm_id == "default"):
        grant = "agent:PUBLIC"
    elif scope == "swarm-shared" and swarm_id:
        grant = f"swarm:{swarm_id}"
    else:
        return False

    read = list(cfg.get("_derived_read", []))
    if grant in read:
        return False
    read.append(grant)
    cfg["_derived_read"] = read
    return True


def _expand_instance_write_for_scope(
    server: MemoryServer,
    *,
    scope: str,
    swarm_id: str,
) -> bool:
    """Enable collaborative writes only for named swarm-shared keys."""
    cfg = getattr(server, "_instance_config", None)
    if not cfg or scope != "swarm-shared" or not swarm_id or swarm_id == "default":
        return False

    grant = f"swarm:{swarm_id}"
    write = list(cfg.get("_derived_write", []))
    if grant in write:
        return False
    write.append(grant)
    cfg["_derived_write"] = write
    return True


def _expand_instance_acl_for_scope(
    server: MemoryServer,
    *,
    scope: str,
    swarm_id: str,
) -> bool:
    """Apply all derived instance ACL grants implied by a stored scope."""
    changed = _expand_instance_read_for_scope(server, scope=scope, swarm_id=swarm_id)
    if _expand_instance_write_for_scope(server, scope=scope, swarm_id=swarm_id):
        changed = True
    return changed


def _should_widen_instance_acl(
    result: dict | None,
    *,
    scope: str,
    swarm_id: str,
    write_kind: str,
) -> bool:
    """Apply derived instance ACL only for writes that establish visible shared state.

    Rules:
    - named swarm-shared writes always widen so collaborators can reach the key
      even when extraction yields zero facts
    - document ingests widen on success because the raw document itself is the
      shared payload, not just extracted facts
    - generic public/default writes widen only when actual facts were produced
    """
    if not isinstance(result, dict):
        return False
    if result.get("status") == "duplicate":
        return False
    if scope == "swarm-shared" and swarm_id and swarm_id != "default":
        return True
    if write_kind == "document" and scope in {"swarm-shared", "system-wide"}:
        return True
    try:
        return int(result.get("facts_extracted") or 0) > 0
    except Exception:
        return False


def _check_instance_acl(server: MemoryServer, owner_id: str, need: str,
                         caller_role: str = "user",
                         memberships: list[str] = None,
                         include_derived: bool = False) -> dict | None:
    """Check instance-level ACL. Returns error dict if denied, None if allowed.

    Uses the same ACL model as per-fact _acl_allows():
    admin → owner → system-sees-system → agent:PUBLIC → direct grant → membership → deny.
    need: "read" or "write".
    """
    cfg = server._instance_config
    if cfg is None:
        return None  # No instance config yet — ACL not active
    if caller_role == "admin":
        return None
    if cfg["owner_id"] == owner_id:
        return None
    # system caller sees system-owned instances
    if owner_id == "system" and cfg["owner_id"] == "system":
        return None
    granted = list(cfg.get(need, []))
    if include_derived and need == "read":
        granted.extend(cfg.get("_derived_read", []))
    if need == "write":
        granted.extend(cfg.get("_derived_write", []))
    # Public grant
    if "agent:PUBLIC" in granted:
        return None
    # Direct grant
    if owner_id in granted:
        return None
    # Membership grant (registry + caller-provided)
    _reg = (server._membership_registry.memberships_for(owner_id)
            if hasattr(server, '_membership_registry') else [])
    all_memberships = set(_reg)
    if memberships:
        all_memberships.update(memberships)
    for m in all_memberships:
        if m in granted:
            return None
    return {"error": "Access denied by instance ACL", "code": "FORBIDDEN"}


async def _run_write_log_workers(poll_interval: float = 0.5, batch_size: int = 8) -> None:
    while True:
        try:
            for server in list(registry.values()):
                try:
                    await server.process_write_log_once(batch_size=batch_size)
                except Exception:
                    log.exception("write-log worker failed for key=%s", getattr(server, "key", "?"))
            await asyncio.sleep(poll_interval)
        except asyncio.CancelledError:
            raise



def _get_courier(key: str) -> Courier:
    if key not in courier_registry:
        mem = _get_memory(key)
        courier = Courier(mem)
        courier_registry[key] = courier
        asyncio.create_task(courier.run(poll_interval=1.0))
    return courier_registry[key]


# ── MCP Tools ──

@mcp.tool(name="memory_store")
@_safe_tool
async def memory_store(
    key: str,
    content: str,
    session_num: int,
    session_date: str,
    speakers: str = "User and Assistant",
    agent_id: str = "default",
    swarm_id: str = "default",
    scope: str = "swarm-shared",
    upsert_by_key: str = None,
    content_type: str = "default",
    librarian_prompt: str = None,
    source_id: str = None,
    retention_ttl: int = None,
    metadata: dict = None,
    target: str | list[str] = None,
) -> dict:
    """Store a conversation turn. Extracts atomic facts and persists to disk.

    upsert_by_key: if set, replaces existing session with same key (agent-private only).
    content_type: prompt registry key (default, financial, technical, personal, regulatory, agent_trace).
    librarian_prompt: inline extraction prompt override (agent-private only).
    source_id: optional identifier for dedup (same source_id + session_num = dedup key).
    retention_ttl: seconds until facts expire (None = never).
    metadata: optional dict of metadata to attach to extracted facts.
    target: optional delivery target(s). Normalized to top-level list[str].
    """
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}

    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id)
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    if created:
        async with server._file_lock:
            server._save_cache()
    result = await server.store(content, session_num, session_date, speakers,
                                agent_id=agent_id, swarm_id=swarm_id, scope=scope,
                                upsert_by_key=upsert_by_key,
                                content_type=content_type,
                                librarian_prompt=librarian_prompt,
                                source_id=source_id,
                                retention_ttl=retention_ttl,
                                metadata=metadata,
                                target=target)
    if _should_widen_instance_acl(result, scope=scope, swarm_id=swarm_id, write_kind="store") and _expand_instance_acl_for_scope(server, scope=scope, swarm_id=swarm_id):
        async with server._file_lock:
            server._save_cache()
    return result


@mcp.tool(name="memory_write")
@_safe_tool
async def memory_write(
    key: str,
    message_id: str,
    session_id: str,
    content: str,
    content_family: str,
    timestamp_ms: int,
    agent_id: str = "default",
    swarm_id: str = "default",
    scope: str = "swarm-shared",
    metadata: dict = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Append a raw write-log entry without blocking on extraction."""
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    if created:
        async with server._file_lock:
            server._save_cache()
    try:
        result = await server.write(
            message_id=message_id,
            session_id=session_id,
            content=content,
            content_family=content_family,
            timestamp_ms=timestamp_ms,
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            metadata=metadata,
        )
    except ValueError as e:
        return {"error": str(e), "code": "VALIDATION_ERROR"}
    if _expand_instance_acl_for_scope(server, scope=scope, swarm_id=swarm_id):
        async with server._file_lock:
            server._save_cache()
    return result


@mcp.tool(name="memory_write_status")
@_safe_tool
async def memory_write_status(
    key: str,
    message_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Check extraction state for a write-log entry."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships, include_derived=True)
    if denied:
        return denied
    status = server.write_status(message_id)
    if status is None:
        return {"error": f"Write {message_id} not found", "code": "NOT_FOUND"}
    _reg_memberships = server._membership_registry.memberships_for(ictx.owner_id) if hasattr(server, '_membership_registry') else []
    _memberships = list(set(_reg_memberships + ictx.memberships))
    if not server._raw_entry_acl_allows(status, ictx.owner_id, _memberships, ictx.caller_role):
        return {"error": f"Write {message_id} not found", "code": "NOT_FOUND"}
    return {
        "message_id": status.get("message_id"),
        "extraction_state": status.get("extraction_state"),
        "extraction_attempts": status.get("extraction_attempts"),
        "last_extraction_attempt_ms": status.get("last_extraction_attempt_ms"),
    }



@mcp.tool(name="memory_recall")
@_safe_tool
async def memory_recall(
    key: str,
    query: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    search_family: str = "auto",
    token_budget: int = 4000,
    query_type: str = "auto",
    kind: str = "all",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Query memory. Returns relevant context from stored facts.

    search_family: auto | conversation | document
    query_type: auto | lookup | temporal | aggregate | current |
                synthesize | procedural | prospective
    kind: all | fact | preference | decision | constraint | rule | ...
    token: OAuth bearer token for identity resolution
    agent_key: API key for identity resolution
    """
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    denied = _check_instance_acl(
        server,
        ictx.owner_id,
        "read",
        ictx.caller_role,
        ictx.memberships,
        include_derived=True,
    )
    if denied:
        return denied
    _reg_memberships = server._membership_registry.memberships_for(ictx.owner_id) if hasattr(server, '_membership_registry') else []
    _memberships = list(set(_reg_memberships + ictx.memberships))
    try:
        result = await server.recall(
            query=query,
            agent_id=agent_id,
            swarm_id=swarm_id,
            search_family=search_family,
            token_budget=token_budget,
            query_type=query_type,
            kind=kind,
            caller_memberships=_memberships,
            caller_role=ictx.caller_role,
            caller_id=ictx.owner_id,
        )
    except Exception as e:
        import traceback
        log.error("memory_recall error:\n%s", traceback.format_exc())
        return {"error": str(e), "code": "RECALL_ERROR"}
    max_chars = token_budget * 4
    if "context" not in result:
        resp = {
            "error": result.get("error", "Recall failed"),
            "code": result.get("code", "RECALL_ERROR"),
            "query_type": result.get("query_type", "default"),
        }
        if "runtime_trace" in result:
            resp["runtime_trace"] = result["runtime_trace"]
        return resp
    if len(result.get("context", "")) > max_chars:
        result["context"] = result["context"][:max_chars] + "\n[...truncated]"
        result.pop("payload", None)
        result.pop("payload_meta", None)
    default_hint = {
        "score": 0.0, "level": 1, "signals": [],
        "retrieval_complexity": 0.0, "content_complexity": 0.0, "dominant": "tie",
    }
    resp = {
        "telemetry_version": 1,
        "context": result["context"],
        "retrieved_count": len(result.get("retrieved", [])),
        "query_type": result.get("query_type", "default"),
        "token_estimate": len(result.get("context", "")) // 4,
        "complexity_hint": result.get("complexity_hint", default_hint),
        "sessions_in_context": result.get("sessions_in_context", 0),
        "total_sessions": result.get("total_sessions", 0),
        "coverage_pct": result.get("coverage_pct", 0),
        "raw_budget": result.get("raw_budget", 5000),
        "recommended_prompt_type": result.get("recommended_prompt_type", ""),
        "use_tool": result.get("use_tool", False),
    }
    if "recommended_profile" in result:
        resp["recommended_profile"] = result["recommended_profile"]
    if "retrieval_families" in result:
        resp["retrieval_families"] = result["retrieval_families"]
    if "search_family" in result:
        resp["search_family"] = result["search_family"]
    if "payload" in result:
        resp["payload"] = result["payload"]
    if "payload_meta" in result:
        resp["payload_meta"] = result["payload_meta"]
    if "actual_injected_episode_ids" in result:
        resp["actual_injected_episode_ids"] = result["actual_injected_episode_ids"]
    if "retrieved_episode_ids" in result:
        resp["retrieved_episode_ids"] = result["retrieved_episode_ids"]
    if "selection_scores" in result:
        resp["selection_scores"] = result["selection_scores"]
    if "runtime_trace" in result:
        resp["runtime_trace"] = result["runtime_trace"]
    if "raw_recall_count" in result:
        resp["raw_recall_count"] = result["raw_recall_count"]
    return resp


@mcp.tool(name="memory_ask")
@_safe_tool
async def memory_ask(
    key: str,
    query: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    search_family: str = "auto",
    query_type: str = "auto",
    kind: str = "all",
    inference_model: str = None,
    max_tokens: int = None,
    use_tool: bool = None,
    shell_budget: float = None,
    speakers: str = "User and Assistant",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Ask a question and get an answer using memory + LLM inference.

    search_family: auto | conversation | document
    Calls recall() internally, selects model via profiles, runs inference.
    Returns answer + metadata (profile_used, tool_called, budget_exceeded).
    """
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    denied = _check_instance_acl(
        server,
        ictx.owner_id,
        "read",
        ictx.caller_role,
        ictx.memberships,
        include_derived=True,
    )
    if denied:
        return denied
    _reg_memberships = server._membership_registry.memberships_for(ictx.owner_id) if hasattr(server, '_membership_registry') else []
    _memberships = list(set(_reg_memberships + ictx.memberships))
    return await server.ask(
        query=query,
        agent_id=agent_id,
        swarm_id=swarm_id,
        search_family=search_family,
        query_type=query_type,
        kind=kind,
        caller_memberships=_memberships,
        caller_role=ictx.caller_role,
        caller_id=ictx.owner_id,
        inference_model=inference_model or (cfg.inference_model if not server._has_profiles() else None),
        max_tokens=max_tokens,
        use_tool=use_tool,
        shell_budget=shell_budget,
        speakers=speakers,
    )


@mcp.tool(name="memory_set_profiles")
@_safe_tool
async def memory_set_profiles(
    key: str,
    profiles: dict,
    profile_configs: dict,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Set inference profiles. Requires instance write; creation-capable."""
    server = _get_memory(key)
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key,
    )
    created = _ensure_instance_config(server, ictx.owner_id)
    denied = _check_instance_acl(
        server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships,
    )
    if denied:
        return denied
    if created:
        async with server._file_lock:
            server._save_cache()
    await server.set_profiles(profiles, profile_configs)
    return {"status": "ok", "profiles": len(profiles), "configs": len(profile_configs)}


@mcp.tool(name="memory_set_config")
@_safe_tool
async def memory_set_config(
    key: str,
    config: dict,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Set canonical memory-owned runtime config. Requires instance write; creation-capable."""
    server = _get_memory(key)
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key,
    )
    created = _ensure_instance_config(server, ictx.owner_id)
    denied = _check_instance_acl(
        server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships,
    )
    if denied:
        return denied
    if created:
        async with server._file_lock:
            server._save_cache()
    await server.set_config(config)
    return {"status": "ok", "schema_version": config.get("schema_version")}


@mcp.tool(name="memory_get_config")
@_safe_tool
async def memory_get_config(
    key: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Get canonical memory-owned runtime config. Requires instance read; not creation-capable."""
    server = _get_memory(key)
    if not getattr(server, "_instance_config", None):
        return {"error": "No memory instance exists for this key", "code": "NOT_FOUND"}
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key,
    )
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return server.get_config()


@mcp.tool(name="memory_get_profiles")
@_safe_tool
async def memory_get_profiles(
    key: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Get inference profiles. Requires instance read; not creation-capable."""
    server = _get_memory(key)
    if not getattr(server, "_instance_config", None):
        return {"error": "No memory instance exists for this key", "code": "NOT_FOUND"}
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key,
    )
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return server.get_profiles()


@mcp.tool(name="memory_ingest_document")
@_safe_tool
async def memory_ingest_document(
    key: str,
    content: str,
    source_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    scope: str = "swarm-shared",
    retention_ttl: int = None,
    metadata: dict = None,
    target: str | list[str] = None,
) -> dict:
    """Ingest a document. Extracts facts across all 3 tiers."""
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}

    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id)
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    if created:
        async with server._file_lock:
            server._save_cache()
    try:
        n = await server.ingest_document(
            content,
            source_id,
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            retention_ttl=retention_ttl,
            metadata=metadata,
            target=target,
        )
    except ValueError as e:
        return {"error": str(e), "code": "VALIDATION_ERROR"}
    if _should_widen_instance_acl({"facts_extracted": n}, scope=scope, swarm_id=swarm_id, write_kind="document") and _expand_instance_acl_for_scope(server, scope=scope, swarm_id=swarm_id):
        async with server._file_lock:
            server._save_cache()
    return {"facts_extracted": n}


@mcp.tool(name="memory_ingest")
@_safe_tool
async def memory_ingest(
    key: str,
    text: str = None,
    path: str = None,
    url: str = None,
    source_id: str = None,
    session_num: int = None,
    session_date: str = None,
    speakers: str = "User and Assistant",
    agent_id: str = "default",
    swarm_id: str = "default",
    scope: str = "swarm-shared",
    retention_ttl: int = None,
    metadata: dict = None,
    target: str | list[str] = None,
) -> dict:
    """Unified transport-only ingest. Exactly one of text/path/url."""
    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}

    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id)
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    if created:
        async with server._file_lock:
            server._save_cache()

    from src.ingest import ingest_input

    try:
        result = await ingest_input(
            server,
            text=text,
            path=path,
            url=url,
            source_id=source_id,
            session_num=session_num,
            session_date=session_date,
            speakers=speakers,
            agent_id=agent_id,
            swarm_id=swarm_id,
            scope=scope,
            retention_ttl=retention_ttl,
            metadata=metadata,
            target=target,
        )
        if _should_widen_instance_acl(result, scope=scope, swarm_id=swarm_id, write_kind="store") and _expand_instance_acl_for_scope(server, scope=scope, swarm_id=swarm_id):
            async with server._file_lock:
                server._save_cache()
        return result
    except ValueError as e:
        return {"error": str(e), "code": "VALIDATION_ERROR"}


def _acl_defaults(owner_id: str, memberships: list[str]) -> tuple:
    """Derive default (read, write) ACL from identity."""
    if owner_id in ("system", "anonymous"):
        return ["agent:PUBLIC"], ["agent:PUBLIC"]
    if memberships:
        return list(memberships), list(memberships)
    return [], []


@mcp.tool(name="memory_ingest_asserted_facts")
@_safe_tool
async def memory_ingest_asserted_facts(
    key: str,
    facts: list[dict],
    consolidated: list[dict] = None,
    cross_session: list[dict] = None,
    raw_sessions: list[dict] = None,
    provenance: dict = None,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
    enrich_l0: bool = True,
) -> dict:
    """Ingest pre-extracted facts. ACL derived from caller identity."""
    server = _get_memory(key)
    ctx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key)
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    if created:
        async with server._file_lock:
            server._save_cache()
    memberships = server._membership_registry.memberships_for(ctx.owner_id) if hasattr(server, '_membership_registry') else []
    _read, _write = _acl_defaults(ctx.owner_id, memberships)
    result = await server.ingest_asserted_facts(
        facts=facts, consolidated=consolidated,
        cross_session=cross_session, raw_sessions=raw_sessions,
        provenance=provenance,
        owner_id=ctx.owner_id,
        read=_read, write=_write,
        enrich_l0=enrich_l0)
    changed = False
    for fact_group in (facts, consolidated or [], cross_session or []):
        for fact in fact_group:
            fact_scope = fact.get("scope", "swarm-shared")
            fact_swarm_id = fact.get("swarm_id", swarm_id)
            if _expand_instance_acl_for_scope(server, scope=fact_scope, swarm_id=fact_swarm_id):
                changed = True
    if changed:
        async with server._file_lock:
            server._save_cache()
    return result


@mcp.tool(name="memory_build_index")
@_safe_tool
async def memory_build_index(key: str, agent_id: str = "default") -> dict:
    """Build embedding index for all three tiers. Blocking."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    try:
        result = await server.build_index()
        # Add embedding config to result
        try:
            from .setup_store import get_config
            _cfg = get_config()
            result["embed_provider"] = _cfg.get("embed_provider", "openai")
            result["embed_model"] = _cfg.get("embed_model", "text-embedding-3-large")
        except Exception:
            result["embed_provider"] = "openai"
            result["embed_model"] = "text-embedding-3-large"
        return result
    except AssertionError as e:
        msg = str(e)
        code = "NO_FACTS" if "No granular facts" in msg else "INDEX_NOT_BUILT"
        return {"error": msg, "code": code}


@mcp.tool(name="memory_flush")
@_safe_tool
async def memory_flush(key: str, agent_id: str = "default") -> dict:
    """Run consolidation + cross-session synthesis. Blocking — returns real counts."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    return await server.flush_background()


@mcp.tool(name="memory_stats")
@_safe_tool
async def memory_stats(key: str, agent_id: str = "default") -> dict:
    """Return memory stats for a key."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id)
    denied = _check_instance_acl(server, ctx.owner_id, "read", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    stats = server.stats()
    # Add embedding config to stats
    try:
        from .setup_store import get_config
        _cfg = get_config()
        stats["embed_provider"] = _cfg.get("embed_provider", "openai")
        stats["embed_model"] = _cfg.get("embed_model", "text-embedding-3-large")
    except Exception:
        stats["embed_provider"] = "openai"
        stats["embed_model"] = "text-embedding-3-large"
    return stats


@mcp.tool(name="memory_reextract")
@_safe_tool
async def memory_reextract(
    key: str,
    model: str = None,
    agent_id: str = "default",
) -> dict:
    """Re-run Librarian extraction on stored raw sessions.

    Use when extraction prompt has been improved.
    Preserves raw sessions, replaces extracted facts.
    """
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    result = await server.reextract(model=model)
    if "error" in result:
        return {"error": result["error"], "code": "NO_RAW_SESSIONS"}
    return result


@mcp.tool(name="memory_list")
@_safe_tool
async def memory_list(
    key: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    kind: str = None,
    limit: int = None,
    offset: int = 0,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """List facts in memory, filtered by ACL and optional kind. Supports pagination."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships, include_derived=True)
    if denied:
        return denied
    server._audit.log("list", ictx.owner_id, {"kind": kind})
    all_facts = server._all_granular + server._all_cons + server._all_cross
    _reg_memberships = server._membership_registry.memberships_for(ictx.owner_id) if hasattr(server, '_membership_registry') else []
    _memberships = list(set(_reg_memberships + ictx.memberships))
    from datetime import datetime as _dt
    from datetime import timezone as _tz
    _now = _dt.now(_tz.utc)
    _fl = server._fact_lookup if hasattr(server, '_fact_lookup') else None
    visible = [f for f in all_facts
               if _is_visible(f, now=_now, fact_lookup=_fl) and server._acl_allows(f, ictx.owner_id, _memberships, ictx.caller_role)]

    if kind:
        visible = [f for f in visible if f.get("kind") == kind]

    total = len(visible)

    if offset:
        visible = visible[offset:]
    if limit is not None:
        visible = visible[:limit]

    return {"total": total, "facts": visible}


@mcp.tool(name="memory_get")
@_safe_tool
async def memory_get(
    key: str,
    fact_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Get a specific fact by ID. Searches all three tiers."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships, include_derived=True)
    if denied:
        return denied
    server._audit.log("get", ictx.owner_id, {"fact_id": fact_id})
    all_facts = server._all_granular + server._all_cons + server._all_cross

    _reg_memberships = server._membership_registry.memberships_for(ictx.owner_id) if hasattr(server, '_membership_registry') else []
    _memberships = list(set(_reg_memberships + ictx.memberships))
    from datetime import datetime as _dt
    from datetime import timezone as _tz
    _now = _dt.now(_tz.utc)
    _fl = server._fact_lookup if hasattr(server, '_fact_lookup') else None
    for f in all_facts:
        if f.get("id") == fact_id:
            if not _is_visible(f, now=_now, fact_lookup=_fl):
                return {"code": "NOT_FOUND", "error": f"Fact {fact_id} not found"}
            if server._acl_allows(f, ictx.owner_id, _memberships, ictx.caller_role):
                return {"fact": f}
            else:
                return {"code": "ACL_FORBIDDEN", "error": "Access denied by ACL"}

    return {"code": "NOT_FOUND", "error": f"Fact {fact_id} not found"}


@mcp.tool(name="memory_edit")
@_safe_tool
async def memory_edit(
    key: str,
    artifact_id: str,
    new_content: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Edit an artifact: create a new version with new content, supersede old."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return await server.edit(artifact_id, new_content,
                             caller_id=ictx.owner_id, caller_role=ictx.caller_role)


@mcp.tool(name="memory_retract")
@_safe_tool
async def memory_retract(
    key: str,
    artifact_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Retract an artifact — makes all versions invisible."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return await server.retract(artifact_id,
                                caller_id=ictx.owner_id, caller_role=ictx.caller_role)


@mcp.tool(name="memory_query")
@_safe_tool
async def memory_query(
    key: str,
    filter: dict = None,
    sort_by: str = "session_date",
    sort_order: str = "desc",
    limit: int = 10,
    offset: int = 0,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Structured query on facts. No vectors, no LLM.

    Filter by any field from fact schema or metadata.* fields:
    - Scalar fields (kind, owner_id, session, scope, ...): exact match
    - List fields (entities, tags, read, write, ...): contains match
    - metadata.* fields: exact match or range operators
    - Range: {"metadata.price": {"gte": 180, "lt": 200}}
    """
    server = _get_memory(key)
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships, include_derived=True)
    if denied:
        return denied
    _reg_memberships = (server._membership_registry.memberships_for(ictx.owner_id)
                        if hasattr(server, '_membership_registry') else [])
    _memberships = list(set(_reg_memberships + ictx.memberships))
    return await server.query(
        filter=filter,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=limit,
        offset=offset,
        caller_id=ictx.owner_id,
        caller_role=ictx.caller_role,
        caller_memberships=_memberships,
    )


@mcp.tool(name="memory_set_schema")
@_safe_tool
async def memory_set_schema(
    key: str, schema: dict,
    agent_id: str = "default", swarm_id: str = "default",
    token: str = None, agent_key: str = None,
) -> dict:
    """Declare metadata schema. Requires instance write access."""
    server = _get_memory(key)
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key)
    created = _ensure_instance_config(server, ictx.owner_id)
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    if created:
        async with server._file_lock:
            server._save_cache()
    await server.set_metadata_schema(schema)
    return {"status": "ok", "fields": len(schema)}


@mcp.tool(name="memory_get_schema")
@_safe_tool
async def memory_get_schema(
    key: str,
    agent_id: str = "default", swarm_id: str = "default",
    token: str = None, agent_key: str = None,
) -> dict:
    """Get current metadata schema. Requires instance read access."""
    server = _get_memory(key)
    ictx = _resolve_identity(
        agent_id=agent_id, swarm_id=swarm_id,
        token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return {"schema": server.get_metadata_schema()}


@mcp.tool(name="memory_purge")
@_safe_tool
async def memory_purge(
    key: str,
    artifact_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Purge an artifact — requires instance write access."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return await server.purge(artifact_id,
                              caller_id=ictx.owner_id, caller_role=ictx.caller_role)


@mcp.tool(name="memory_redact")
@_safe_tool
async def memory_redact(
    key: str,
    artifact_id: str,
    fields: list[str] = None,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Redact fields of an artifact — requires instance write access."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return await server.redact(artifact_id, fields or ["fact", "entities", "content"],
                               caller_id=ictx.owner_id, caller_role=ictx.caller_role)


@mcp.tool(name="memory_get_versions")
@_safe_tool
async def memory_get_versions(
    key: str,
    artifact_id: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Get version chain for an artifact."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    return server.get_versions(artifact_id,
                               caller_id=ictx.owner_id, caller_role=ictx.caller_role)


@mcp.tool(name="courier_subscribe")
@_safe_tool
async def courier_subscribe(
    key: str,
    connection_id: str = "",
    deliver_existing: bool = False,
    filter: dict = None,
    agent_id: str = "default",
    swarm_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Subscribe to new facts matching filter. Push via SSE stream."""
    # C2: verify connection_id is a known active SSE connection
    if connection_id and connection_id not in _active_connections:
        return {"error": "Unknown connection_id — connect via /mcp/sse first",
                "code": "INVALID_CONNECTION"}

    courier = _get_courier(key)

    # Pre-generate sub_id and wire routing BEFORE subscribe()
    # so deliver_existing can route events through the SSE queue
    from uuid import uuid4 as _uuid4
    pre_sub_id = f"sub_{_uuid4().hex[:8]}"
    sub_to_conn[pre_sub_id] = connection_id

    async def _push(fact: dict):
        cid = sub_to_conn.get(pre_sub_id)
        if cid and cid in connections:
            await connections[cid].put({
                "type": "artifact",
                "sub_id": pre_sub_id,
                "payload": fact,
            })

    # Resolve subscriber identity for ACL
    ictx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id, token=token, agent_key=agent_key)
    server = _get_memory(key)
    _reg_memberships = server._membership_registry.memberships_for(ictx.owner_id) if hasattr(server, '_membership_registry') else []
    _memberships = list(set(_reg_memberships + ictx.memberships))

    sub_id = await courier.subscribe(
        filter=filter or {},
        callback=_push,
        deliver_existing=deliver_existing,
        owner_id=ictx.owner_id,
        memberships=_memberships,
        pre_sub_id=pre_sub_id,
        caller_role=ictx.caller_role,
    )
    return {"sub_id": sub_id}


@mcp.tool(name="courier_unsubscribe")
@_safe_tool
async def courier_unsubscribe(sub_id: str) -> dict:
    """Unsubscribe from Courier push. Idempotent."""
    for courier in courier_registry.values():
        await courier.unsubscribe(sub_id)
    sub_to_conn.pop(sub_id, None)
    return {"status": "ok"}


@mcp.tool(name="membership_register")
@_safe_tool
async def membership_register(
    key: str,
    identity: str,
    group: str,
) -> dict:
    """Register identity as member of group. E.g. agent:alice → swarm:alpha."""
    server = _get_memory(key)
    server._membership_registry.register(identity, group)
    return {"registered": True, "identity": identity, "group": group}


@mcp.tool(name="membership_unregister")
@_safe_tool
async def membership_unregister(
    key: str,
    identity: str,
    group: str,
) -> dict:
    """Remove identity from group. Idempotent."""
    server = _get_memory(key)
    server._membership_registry.unregister(identity, group)
    return {"unregistered": True, "identity": identity, "group": group}


@mcp.tool(name="membership_list")
@_safe_tool
async def membership_list(
    key: str,
    identity: str,
) -> dict:
    """List all group memberships for an identity."""
    server = _get_memory(key)
    groups = server._membership_registry.memberships_for(identity)
    return {"identity": identity, "memberships": groups}


@mcp.tool(name="memory_store_secret")
@_safe_tool
async def memory_store_secret(
    key: str,
    name: str,
    value: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    scope: str = "system-wide",
) -> dict:
    """Store a secret. Never enters fact index. Upsert by name within scope context."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    return server.store_secret(name, value, agent_id, swarm_id, scope)


@mcp.tool(name="memory_get_secret")
@_safe_tool
async def memory_get_secret(
    key: str,
    name: str,
    agent_id: str = "default",
    swarm_id: str = "default",
) -> dict:
    """Fetch secret by exact name. Value is never logged."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id)
    denied = _check_instance_acl(server, ctx.owner_id, "read", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    return server.get_secret(name, agent_id, swarm_id)


@mcp.tool(name="memory_import")
@_safe_tool
async def memory_import(
    key: str,
    source_format: str,
    content: str = None,
    path: str = None,
    source_uri: str = None,
    token: str = None,
    options: str = None,
    content_type: str = "default",
    agent_id: str = "default",
    swarm_id: str = "default",
    scope: str = "swarm-shared",
    agent_key: str = None,
    auth_token: str = None,
) -> dict:
    """Import data into memory. Same capabilities as the CLI.

    source_format:
      - conversation_json — Claude/ChatGPT export (pass content)
      - text — plain text (pass content)
      - directory — local folder path (pass path)
      - git — clone a repo (pass source_uri, optionally token)

    For conversation_json/text: pass file content as `content` string.
    For directory: pass folder path as `path`.
    For git: pass repo URL as `source_uri`, optionally `token` for private repos.

    options: JSON string with format-specific options.
      git: {"branch": "dev", "file_patterns": ["*.py"], "max_files": 200}
    content_type: extraction prompt — default | technical | financial | personal |
                  regulatory | agent_trace

    token: source/repo auth (git clone). NOT used for caller identity.
    agent_key: caller identity auth (separate from source auth).
    auth_token: verified caller auth token (routed to _verified_auth_resolver).
    """
    ALL_FORMATS = {"conversation_json", "text", "directory", "git"}

    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id, swarm_id=swarm_id,
                             token=auth_token, agent_key=agent_key)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied

    if scope not in VALID_SCOPES:
        return {"error": f"Unknown scope: {scope}", "code": "INVALID_SCOPE"}

    if source_format not in ALL_FORMATS:
        return {"error": f"Unknown format: {source_format!r}. "
                         f"Supported: {sorted(ALL_FORMATS)}",
                "code": "UNKNOWN_FORMAT"}

    try:
        opts = json.loads(options) if options else {}
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in options: {e}", "code": "INVALID_OPTIONS"}

    # ── git: clone repo → sessions ──
    if source_format == "git":
        if not source_uri:
            return {"error": "source_uri required for git format",
                    "code": "MISSING_PARAM"}
        from .git_importer import import_git
        if token:
            opts["token"] = token
        opts.setdefault("content_type", content_type)
        try:
            sessions = import_git(source_uri, opts)
        except Exception as e:
            return {"error": str(e), "code": "GIT_ERROR"}

    # ── directory: read files from local path ──
    elif source_format == "directory":
        if not path:
            return {"error": "path required for directory format",
                    "code": "MISSING_PARAM"}
        from pathlib import Path as P
        dir_path = P(path)
        if not dir_path.exists():
            return {"error": f"Path not found: {path}", "code": "PATH_NOT_FOUND"}
        if not dir_path.is_dir():
            return {"error": f"Not a directory: {path}", "code": "NOT_A_DIRECTORY"}

        IMPORTABLE_SUFFIXES = {
            '.txt', '.md', '.json', '.py', '.rst', '.yaml', '.yml',
            '.csv', '.xml', '.html', '.log', '.cfg', '.ini', '.toml',
            '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.java', '.c',
            '.cpp', '.h', '.hpp', '.rb', '.sh', '.sql', '.proto',
        }
        parts = []
        for f in sorted(dir_path.rglob("*")):
            if f.is_file() and f.suffix in IMPORTABLE_SUFFIXES:
                try:
                    rel = f.relative_to(dir_path)
                except ValueError:
                    rel = f.name
                parts.append(f"---FILE: {rel}---")
                parts.append(f.read_text(encoding="utf-8", errors="replace"))
        concatenated = "\n".join(parts)

        from .importers import parse_history
        try:
            sessions = parse_history("directory", concatenated)
        except Exception as e:
            return {"error": str(e), "code": "PARSE_ERROR"}

    # ── conversation_json / text: parse content string ──
    else:
        if not content:
            return {"error": "content required for this format",
                    "code": "MISSING_PARAM"}
        from .importers import parse_history
        try:
            sessions = parse_history(source_format, content)
        except Exception as e:
            return {"error": str(e), "code": "PARSE_ERROR"}

    if not sessions:
        return {"error": "No sessions parsed from input", "code": "EMPTY_INPUT"}

    # ── Store all sessions ──
    server = _get_memory(key)
    total_facts = 0
    errors = []
    skipped = 0

    for session in sessions:
        # Git dedup: check _git_dedup_index
        if source_format == "git" and session.get("source_id") and session.get("artifact_path"):
            dedup_key = (session["source_id"], session["artifact_path"])
            existing = server._git_dedup_index.get(dedup_key)
            if existing and existing.get("blob_sha") == session.get("blob_sha"):
                skipped += 1
                continue  # unchanged — skip
            # New or changed — will store below

        try:
            from .identity import _generate_artifact_id, _generate_version_id, content_hash_text
            store_kwargs = {
                "content": session["content"],
                "session_num": session["session_num"],
                "session_date": session["session_date"],
                "speakers": session.get("speakers", "User and Assistant"),
                "agent_id": agent_id,
                "swarm_id": swarm_id,
                "scope": scope,
                "content_type": session.get("content_type", content_type),
            }
            # For git imports, pass identity + source_meta + skip_dedup
            if source_format == "git" and session.get("artifact_path"):
                dedup_key = (session.get("source_id", ""), session["artifact_path"])
                existing = server._git_dedup_index.get(dedup_key)
                if existing:
                    # Changed file → new version of same artifact
                    art_id = existing.get("artifact_id", _generate_artifact_id())
                    ver_id = _generate_version_id()
                    parent_ver = existing.get("version_id")
                else:
                    # New file
                    art_id = _generate_artifact_id()
                    ver_id = _generate_version_id()
                    parent_ver = None

                from .identity import content_hash_git
                blob = session.get("blob_sha")
                ch = content_hash_git(blob) if blob else content_hash_text(session["content"])
                store_kwargs["source_id"] = session.get("source_id")
                store_kwargs["artifact_id"] = art_id
                store_kwargs["version_id"] = ver_id
                store_kwargs["parent_version"] = parent_ver
                store_kwargs["content_hash"] = ch
                store_kwargs["skip_dedup"] = True
                store_kwargs["source_meta"] = {
                    "artifact_path": session["artifact_path"],
                    "blob_sha": session.get("blob_sha"),
                    "storage_mode": session.get("storage_mode", "inline"),
                }
            result = await server.store(**store_kwargs)
            total_facts += result.get("facts_extracted", 0)

            # Update git dedup index + persist
            if source_format == "git" and session.get("source_id") and session.get("artifact_path"):
                dedup_key = (session["source_id"], session["artifact_path"])
                server._git_dedup_index[dedup_key] = {
                    "blob_sha": session.get("blob_sha"),
                    "artifact_id": store_kwargs.get("artifact_id", result.get("artifact_id", "")),
                    "version_id": store_kwargs.get("version_id", ""),
                }
                server._save_cache()
        except Exception as e:
            errors.append({"session": session["session_num"], "error": str(e)})

    resp = {
        "sessions_processed": len(sessions) - len(errors) - skipped,
        "total_sessions":     len(sessions),
        "facts_extracted":    total_facts,
        "errors":             errors,
    }
    if skipped:
        resp["skipped_unchanged"] = skipped
    if resp["sessions_processed"] > 0 and _expand_instance_acl_for_scope(
        server,
        scope=scope,
        swarm_id=swarm_id,
    ):
        async with server._file_lock:
            server._save_cache()
    return resp


# Backward compat alias
@mcp.tool(name="memory_import_history")
@_safe_tool
async def memory_import_history(
    key: str,
    source_format: str,
    content: str,
    agent_id: str = "default",
    swarm_id: str = "default",
    scope: str = "swarm-shared",
) -> dict:
    """Import conversation history. Alias for memory_import with content param."""
    return await memory_import(
        key=key, source_format=source_format, content=content,
        agent_id=agent_id, swarm_id=swarm_id, scope=scope,
    )


@mcp.tool(name="memory_list_prompts")
@_safe_tool
async def memory_list_prompts(key: str, agent_id: str = "default") -> dict:
    """List all available content_types in the prompt registry."""
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id)
    denied = _check_instance_acl(server, ctx.owner_id, "read", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    return {"prompts": server._prompt_registry.list()}


@mcp.tool(name="memory_get_prompt")
@_safe_tool
async def memory_get_prompt(key: str, content_type: str, agent_id: str = "default") -> dict:
    """Get the prompt text for a content_type.

    Returns the resolved prompt (custom overrides builtin).
    """
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id)
    denied = _check_instance_acl(server, ctx.owner_id, "read", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    if not server._prompt_registry.exists(content_type):
        return {"error": f"prompt not found: {content_type!r}",
                "code": "PROMPT_NOT_FOUND"}
    prompt = server._prompt_registry.get(content_type)
    custom_path = server._prompt_registry._custom_path(content_type)
    source = "custom" if custom_path.exists() else "builtin"
    return {"content_type": content_type, "prompt": prompt, "source": source}


@mcp.tool(name="memory_set_prompt")
@_safe_tool
async def memory_set_prompt(
    key: str,
    content_type: str,
    prompt: str,
    agent_id: str = "default",
) -> dict:
    """Register or update a Librarian prompt for a content_type.

    Persists to {data_dir}/librarian_prompts/{content_type}.md.
    Overrides builtin if same name.
    """
    if not content_type or not content_type.replace("_", "").isalnum():
        return {"error": "content_type must be alphanumeric with underscores",
                "code": "INVALID_CONTENT_TYPE"}
    server = _get_memory(key)
    ctx = _resolve_identity(agent_id=agent_id)
    created = _ensure_instance_config(server, ctx.owner_id)
    denied = _check_instance_acl(server, ctx.owner_id, "write", ctx.caller_role, ctx.memberships)
    if denied:
        return denied
    server._prompt_registry.set(content_type, prompt)
    if created:
        async with server._file_lock:
            server._save_cache()
    return {"stored": True, "content_type": content_type}


# ── MAL tools ──

_mal_stores: dict[tuple[str, str], dict] = {}


def _get_mal_stores(data_dir_path: str, server=None) -> dict:
    key = server.key if server else ""
    cache_key = (data_dir_path, key)
    if cache_key not in _mal_stores:
        from src.mal.artifact_store import ArtifactStore
        from src.mal.control_store import ControlStore
        from src.mal.feedback_store import FeedbackStore
        from src.mal.scheduler import Scheduler
        control = ControlStore(data_dir_path)
        feedback = FeedbackStore(data_dir_path, control)
        artifacts = ArtifactStore(data_dir_path)
        _mal_stores[cache_key] = {
            "control": control,
            "feedback": feedback,
            "artifacts": artifacts,
            "scheduler": Scheduler(data_dir_path, control, feedback,
                                   artifacts=artifacts, server=server),
        }
    elif server is not None:
        _mal_stores[cache_key]["scheduler"]._server = server
    return _mal_stores[cache_key]


@mcp.tool(name="memory_mal_configure")
@_safe_tool
async def memory_mal_configure(
    key: str,
    agent_id: str = "default",
    enabled: bool = False,
    auto_collect_feedback: bool = False,
    auto_trigger: bool = False,
    min_signals: int = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Enable/disable MAL and set control flags for a binding. Requires write ACL.

    min_signals: minimum independent failure signals before MAL accepts
    any pipeline change (default 10). Lower = more aggressive adaptation.
    """
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    fields = {
        "enabled": enabled,
        "auto_collect_feedback": auto_collect_feedback,
        "auto_trigger": auto_trigger,
    }
    if min_signals is not None:
        fields["min_signals"] = max(2, int(min_signals))
    stores["control"].set(key, agent_id, **fields)
    return {
        "status": "ok",
        "key": key,
        "agent_id": agent_id,
        "config": stores["control"].get(key, agent_id),
    }


@mcp.tool(name="memory_mal_feedback")
@_safe_tool
async def memory_mal_feedback(
    key: str,
    verdict: str,
    query: str,
    agent_id: str = "default",
    signal_source: str = "user",
    runtime_trace_ref: str = None,
    runtime_trace: dict = None,
    response_excerpt: str = None,
    corrected_answer: str = None,
    retry_chain_id: str = None,
    source_ids_hint: list[str] = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Submit a single feedback event to the MAL queue. Requires write ACL.

    runtime_trace_ref: stable ID from ask() result for linking
    runtime_trace: full trace payload from ask() for diagnosis (required for trigger)
    """
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    event = {
        "signal_source": signal_source,
        "verdict": verdict,
        "query": query,
        "runtime_trace_ref": runtime_trace_ref,
        "runtime_trace": runtime_trace,
        "response_excerpt": response_excerpt,
        "corrected_answer": corrected_answer,
        "retry_chain_id": retry_chain_id,
        "source_ids_hint": source_ids_hint,
    }
    event_id = stores["feedback"].submit(key, agent_id, event)
    return {"status": "ok", "feedback_event_id": event_id}


@mcp.tool(name="memory_mal_trigger")
@_safe_tool
async def memory_mal_trigger(
    key: str,
    agent_id: str = "default",
    feedback_event_ids: list[str] = None,
    estimate_only: bool = False,
    force: bool = False,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Trigger a MAL adaptation run. Requires write ACL."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    return stores["scheduler"].trigger(
        key, agent_id=agent_id,
        feedback_event_ids=feedback_event_ids,
        estimate_only=estimate_only,
        force=force,
    )


@mcp.tool(name="memory_mal_status")
@_safe_tool
async def memory_mal_status(
    key: str,
    agent_id: str = "default",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Report MAL binding state, queued feedback count, and convergence. Requires read ACL."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    control = stores["control"].get(key, agent_id)
    queued = stores["feedback"].list_queued(key, agent_id)
    convergence = stores["scheduler"].get_convergence_state(key, agent_id)
    latest = stores["artifacts"].get_latest(key, agent_id)
    return {
        "key": key,
        "agent_id": agent_id,
        "enabled": control.get("enabled", False),
        "auto_collect_feedback": control.get("auto_collect_feedback", False),
        "auto_trigger": control.get("auto_trigger", False),
        "queued_feedback_count": len(queued),
        "convergence_state": convergence.get("convergence_state", "active"),
        "rejected_streak": convergence.get("rejected_streak", 0),
        "latest_artifact_id": latest["artifact_id"] if latest else None,
        "latest_artifact_version": latest["version"] if latest else None,
    }


@mcp.tool(name="memory_mal_list_feedback")
@_safe_tool
async def memory_mal_list_feedback(
    key: str,
    agent_id: str = "default",
    status_filter: str = "queued",
    token: str = None,
    agent_key: str = None,
) -> dict:
    """List feedback events in the MAL queue. Requires read ACL."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    if status_filter == "queued":
        events = stores["feedback"].list_queued(key, agent_id)
    elif status_filter == "eligible":
        events = stores["feedback"].list_trigger_eligible(key, agent_id)
    else:
        events = stores["feedback"]._all_events(key, agent_id)
    return {"events": events, "count": len(events)}


@mcp.tool(name="memory_mal_get_artifact")
@_safe_tool
async def memory_mal_get_artifact(
    key: str,
    agent_id: str = "default",
    artifact_id: str = None,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Get a MAL artifact by id, or the latest if no id given. Requires read ACL."""
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "read", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    if artifact_id:
        artifact = stores["artifacts"].get(key, agent_id, artifact_id)
    else:
        artifact = stores["artifacts"].get_latest(key, agent_id)
    if artifact is None:
        return {"error": "No artifact found", "code": "NOT_FOUND"}
    return {"artifact": artifact}


@mcp.tool(name="memory_mal_rollback")
@_safe_tool
async def memory_mal_rollback(
    key: str,
    agent_id: str = "default",
    to_artifact_id: str = None,
    confirm: bool = False,
    token: str = None,
    agent_key: str = None,
) -> dict:
    """Rollback MAL binding to a prior artifact state. Two-step protocol.

    confirm=False: preview rollback plan (no side effects)
    confirm=True: execute rollback
    """
    server = _get_memory(key)
    ictx = _resolve_identity(agent_id=agent_id, swarm_id="default", token=token, agent_key=agent_key)
    denied = _check_instance_acl(server, ictx.owner_id, "write", ictx.caller_role, ictx.memberships)
    if denied:
        return denied
    stores = _get_mal_stores(str(server.data_dir), server=server)
    from src.mal.apply import ApplyEngine, current_gen_dir, plan_rollback

    latest = stores["artifacts"].get_latest(key, agent_id)
    if latest is None:
        return {"error": "No artifact to rollback from", "code": "NO_ARTIFACT"}

    if to_artifact_id:
        target = stores["artifacts"].get(key, agent_id, to_artifact_id)
        if target is None:
            return {"error": f"Target artifact {to_artifact_id} not found", "code": "NOT_FOUND"}
        target_state = target["materialized_state"]
    else:
        # Rollback to zero state (defaults)
        target_state = {
            "selector_config_overrides": {},
            "grouping_prompt_mode": "strict_small",
            "size_cap_chars": 12000,
            "extraction_prompts": {},
            "inference_leaf_plugin_overrides": {},
        }

    current_state = latest["materialized_state"]
    rollback_plan = plan_rollback(current_state, target_state)

    if not confirm:
        return {
            "target_artifact_id": to_artifact_id,
            "rollback_plan": rollback_plan,
            "requires_confirmation": True,
        }

    # Execute rollback
    data_dir = str(server.data_dir)
    engine = ApplyEngine(data_dir)
    gen_dir = current_gen_dir(data_dir, key, agent_id)
    current_gen_num = int(gen_dir.name.replace("gen_", "")) if gen_dir.name.startswith("gen_") else 0

    result = engine.apply_generation(
        key=key, agent_id=agent_id,
        materialized_state=target_state,
        previous_gen=current_gen_num,
    )

    return {
        "target_artifact_id": to_artifact_id,
        "rollback_plan": rollback_plan,
        "apply_status": result.get("final_status"),
        "confirmed": True,
    }


# ── SSE endpoint ──

async def sse_cleanup(conn_id: str):
    """Remove SSE connection and all its subscriptions."""
    for sid, cid in list(sub_to_conn.items()):
        if cid == conn_id:
            for courier in courier_registry.values():
                await courier.unsubscribe(sid)
            sub_to_conn.pop(sid, None)
    connections.pop(conn_id, None)
    _active_connections.pop(conn_id, None)


async def sse_endpoint(request):
    """SSE stream endpoint. Sends connected event, then pushes Courier artifacts."""
    from starlette.responses import StreamingResponse

    conn_id = str(uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    connections[conn_id] = queue
    # C2: register this connection as active (for hijack prevention)
    remote = ""
    if request is not None:
        client = getattr(request, "client", None)
        remote = f"{client.host}:{client.port}" if client else "unknown"
    _active_connections[conn_id] = remote

    await queue.put({"type": "connected", "connection_id": conn_id})

    async def event_generator():
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            await sse_cleanup(conn_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── App factory ──

def create_app(app_data_dir="./data", app_cfg=None):
    """Create combined Starlette app with MCP + SSE.

    The MCP SDK's session manager requires lifespan initialization
    (task group setup). We forward it from the outer Starlette app.
    """
    global data_dir, cfg
    data_dir = app_data_dir
    if app_cfg:
        cfg = app_cfg

    # H2 fix: persist embed_model to config so embed_batch() picks it up
    if app_cfg and app_cfg.embed_model:
        try:
            from .setup_store import load_config, save_config
            file_cfg = load_config()
            if file_cfg.get("embed_model") != app_cfg.embed_model:
                file_cfg["embed_model"] = app_cfg.embed_model
                save_config(file_cfg)
        except Exception:
            pass  # best-effort — don't block server startup

    import contextlib

    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route

    # C1: Token authentication middleware
    class TokenAuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            # /health is public (monitoring probes need it)
            if request.url.path == "/health":
                return await call_next(request)
            token = request.headers.get("x-server-token", "")
            if token != SERVER_TOKEN:
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    mcp_app = mcp.streamable_http_app()

    async def health(request):
        return JSONResponse({"status": "ok"})

    # C3: Admin reload endpoint — lets CLI import notify running server
    async def admin_reload(request):
        body = await request.json()
        key = body.get("key")
        if key and key in registry:
            server = registry[key]
            server._data_dict = None  # force re-index on next recall
            # Reload facts from disk
            if server._storage.exists:
                cached = server._storage.load_facts()
                server._all_granular = cached.get("granular", [])
                server._all_cons = cached.get("cons", [])
                server._all_cross = cached.get("cross", [])
                server._all_tlinks = cached.get("tlinks", [])
                server._n_sessions = cached.get("n_sessions", 0)
                server._n_sessions_with_facts = cached.get("n_sessions_with_facts", 0)
                server._raw_sessions = cached.get("raw_sessions", [])
                server._secrets = cached.get("secrets", [])
                # Restore _temporal_links on granular facts
                for f in server._all_granular:
                    f["_temporal_links"] = []
                if server._all_granular and server._all_tlinks:
                    server._all_granular[0]["_temporal_links"] = server._all_tlinks
        return JSONResponse({"status": "reloaded"})

    @contextlib.asynccontextmanager
    async def lifespan(app):
        global _write_log_worker_task
        async with mcp.session_manager.run():
            _write_log_worker_task = asyncio.create_task(_run_write_log_workers())
            try:
                yield
            finally:
                if _write_log_worker_task is not None:
                    _write_log_worker_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await _write_log_worker_task
                    _write_log_worker_task = None

    return Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Route("/admin/reload", admin_reload, methods=["POST"]),
            Route("/mcp/sse", endpoint=sse_endpoint),
            Mount("/", app=mcp_app),
        ],
        lifespan=lifespan,
        middleware=[Middleware(TokenAuthMiddleware)],
    )


# ── CLI ──

def parse_args():
    p = argparse.ArgumentParser(description="GOSH Memory MCP Server")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--model", type=str, default=None)
    p.add_argument("--extraction-model", type=str, default=None)
    p.add_argument("--inference-model", type=str, default=None)
    p.add_argument("--judge-model", type=str, default=None)
    p.add_argument("--embed-model", type=str, default=None)
    p.add_argument("--server-token", type=str, default=None,
                   help="Server auth token (if not set, auto-generated)")
    return p.parse_args()


def _save_token():
    """Save server token to ~/.gosh-memory/token on startup."""
    token_path = Path.home() / ".gosh-memory" / "token"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(SERVER_TOKEN)
    token_path.chmod(0o600)
    return token_path


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    if args.server_token:
        SERVER_TOKEN = args.server_token
    app_cfg = MemoryConfig.from_args(args)
    app = create_app(app_data_dir=args.data_dir, app_cfg=app_cfg)

    token_path = _save_token()

    # Resolve embedding config for startup display
    try:
        from .setup_store import get_config as _get_cfg
        _scfg = _get_cfg()
        _embed_prov = _scfg.get("embed_provider", "openai")
        _embed_mod = _scfg.get("embed_model", "text-embedding-3-large")
    except Exception:
        _embed_prov, _embed_mod = "openai", "text-embedding-3-large"

    print(f"GOSH Memory MCP Server — {app_cfg.summary()}")
    print(f"Listening on http://{args.host}:{args.port}")
    print(f"  Embeddings:   {_embed_prov} / {_embed_mod}")
    print("  POST /mcp     → MCP tool calls")
    print("  GET  /mcp/sse → Courier SSE stream")
    print(f"Server token: {SERVER_TOKEN}")
    print(f"Token saved to: {token_path}")

    uvicorn.run(app, host=args.host, port=args.port)
