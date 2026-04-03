#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOSH Memory — Courier.

Lightweight change-detection and delivery layer on top of MemoryServer.
Polls disk for new facts, filters by subscriber criteria, calls async callbacks.
Read-only — never calls store(), build_index(), or any LLM-triggering method.
"""

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .memory import MemoryServer, _fact_matches_structured_filter, _is_visible

log = logging.getLogger(__name__)


@dataclass
class SubscriptionEntry:
    sub_id: str
    filter: dict[str, Any]
    callback: Callable[..., Coroutine]
    owner_id: str = ""          # caller identity at subscribe time
    memberships: list[str] = field(default_factory=list)  # caller's group memberships
    caller_role: str = "user"  # "admin" bypasses ACL


class Courier:
    """Change-detection and delivery layer for MemoryServer.

    One instance per swarm. Polls disk for new facts, filters by subscriber
    criteria, delivers via async callbacks.
    """

    def __init__(self, memory: MemoryServer):
        self.memory = memory
        self._subscriptions: dict[str, SubscriptionEntry] = {}
        self._running: bool = False

        # Initialize _last_seen_at from max(created_at) of existing facts,
        # not wall clock — avoids re-delivering old facts on restart
        existing = self._load_facts()
        if existing:
            max_ts = max(
                (f.get("created_at", "") for f in existing), default=""
            )
            self._last_seen_at = max_ts if max_ts else "1970-01-01T00:00:00+00:00"
        else:
            self._last_seen_at = "1970-01-01T00:00:00+00:00"

    # ── Public API ──

    async def subscribe(
        self,
        filter: dict[str, Any],
        callback: Callable[..., Coroutine],
        deliver_existing: bool = False,
        owner_id: str = "",
        memberships: list[str] = None,
        pre_sub_id: str = None,
        caller_role: str = "user",
    ) -> str:
        sub_id = pre_sub_id or f"sub_{uuid4().hex[:8]}"
        self._subscriptions[sub_id] = SubscriptionEntry(
            sub_id, filter, callback,
            owner_id=owner_id,
            memberships=memberships or [],
            caller_role=caller_role,
        )

        if deliver_existing:
            all_facts = self._load_facts()
            _fl = {f.get("id", ""): f for f in all_facts}
            _now = datetime.now(timezone.utc)
            for fact in all_facts:
                if (_is_visible(fact, now=_now, fact_lookup=_fl)
                        and self._matches(fact, filter)
                        and self._acl_check(fact, owner_id, memberships or [], caller_role)):
                    try:
                        await callback(fact)
                    except Exception as e:
                        log.warning("callback error in %s during deliver_existing: %s", sub_id, e)

        return sub_id

    async def unsubscribe(self, sub_id: str) -> None:
        self._subscriptions.pop(sub_id, None)

    async def run(self, poll_interval: float = 1.0) -> None:
        self._running = True
        while self._running:
            await self._poll()
            await asyncio.sleep(poll_interval)

    async def stop(self) -> None:
        self._running = False

    # ── Internal ──

    async def _poll(self) -> int:
        all_facts = self._load_facts()

        new_facts = [
            f for f in all_facts
            if f.get("created_at", "") > self._last_seen_at
        ]
        if not new_facts:
            return 0

        self._last_seen_at = max(f["created_at"] for f in new_facts)

        delivered = 0
        _now = datetime.now(timezone.utc)
        _fl = {f.get("id", ""): f for f in all_facts}
        for fact in new_facts:
            if not _is_visible(fact, now=_now, fact_lookup=_fl):
                continue
            for sub in list(self._subscriptions.values()):
                if (self._matches(fact, sub.filter)
                        and self._acl_check(fact, sub.owner_id, sub.memberships, sub.caller_role)):
                    try:
                        await sub.callback(fact)
                        delivered += 1
                    except Exception as e:
                        log.warning("callback error in %s: %s", sub.sub_id, e)
        return delivered

    def _matches(self, fact: dict, filter: dict) -> bool:
        return _fact_matches_structured_filter(
            fact,
            filter,
            getattr(self.memory, "_metadata_schema", None),
        )

    def _acl_check(self, fact: dict, caller_id: str, caller_memberships: list[str],
                   caller_role: str = "user") -> bool:
        """Check if subscriber is allowed to see this fact.

        If no owner_id is set on subscription (legacy), allow all (backward compat).
        Otherwise delegate to MemoryServer._acl_allows.
        """
        if not caller_id:
            return True  # backward compat: no identity set → allow all
        return self.memory._acl_allows(fact, caller_id, caller_memberships, caller_role)

    def _load_facts(self) -> list[dict]:
        """Load facts from the configured storage backend.

        Tests and migration paths may still materialize legacy `{key}.json`
        directly, so fall back to it when storage has no visible snapshot.
        """
        storage = getattr(self.memory, "_storage", None)
        if storage is not None and getattr(storage, "exists", False):
            data = storage.load_facts()
            facts = data.get("granular", []) + data.get("cons", []) + data.get("cross", [])
            if facts:
                return facts
        cache_path = self.memory.data_dir / f"{self.memory.key}.json"
        if not cache_path.exists():
            return []
        data = json.loads(cache_path.read_text())
        return data.get("granular", []) + data.get("cons", []) + data.get("cross", [])
