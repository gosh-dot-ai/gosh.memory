#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GOSH Memory — Membership Registry.

Tracks which identities belong to which groups (swarms, teams, etc.).
All IDs are canonical: agent:alice != user:alice.
"""

from collections import defaultdict


class MembershipRegistry:
    """In-memory membership registry.

    Maps identity → set of group memberships.
    """

    def __init__(self):
        self._memberships: dict[str, set[str]] = defaultdict(set)

    def register(self, identity: str, group: str) -> None:
        """Add identity to group."""
        self._memberships[identity].add(group)

    def unregister(self, identity: str, group: str) -> None:
        """Remove identity from group. No-op if not a member."""
        self._memberships[identity].discard(group)
        if not self._memberships[identity]:
            del self._memberships[identity]

    def memberships_for(self, identity: str) -> list[str]:
        """Return list of groups identity belongs to."""
        return list(self._memberships.get(identity, set()))
