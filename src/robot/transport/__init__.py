"""Edge ↔ Brain transport. InProcessTransport for the single-process default;
robot.transport.websocket has the Phase 8 client/server pair (imported lazily
by consumers so an Edge-only or Brain-only process loads just its half)."""

from robot.transport.base import Transport
from robot.transport.inproc import InProcessTransport

__all__ = ["Transport", "InProcessTransport"]
