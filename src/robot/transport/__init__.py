"""Edge ↔ Brain transport. In-process today; WebSocket in Phase 8."""

from robot.transport.base import Transport
from robot.transport.inproc import InProcessTransport

__all__ = ["Transport", "InProcessTransport"]
