"""Durable episodic memory (Minimum Viable Memory).

Public surface: `MemoryStore` (the episodic log) and `Episode` (one turn).
See `memory-architecture.md` for the full v2 design this grows into.
"""

from robot.memory.store import Episode, MemoryStore

__all__ = ["Episode", "MemoryStore"]
