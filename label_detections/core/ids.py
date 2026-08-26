"""Identifiers shared by the schema modules."""
from __future__ import annotations

import uuid


def ensure_id() -> str:
    """A short, stable id for a box, so regions and findings can refer to it."""
    return uuid.uuid4().hex[:8]
