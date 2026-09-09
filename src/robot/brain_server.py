"""Brain runtime: the Mac side of the Phase 8 Pi/Mac split.

Mirror image of main.py (the Edge runtime): builds the Agent, then serves it
over a WebSocket instead of wiring it to a local mic and speaker. Run with
`just brain` (starts Ollama too) or `python -m robot.brain_server`.

The Edge — `TRANSPORT=websocket python -m robot.main`, on this machine for
the loopback smoke test or on the Pi later — connects with the same
TRANSPORT_TOKEN.
"""

import asyncio

from robot.brain import Agent
from robot.config import BRAIN_WS_HOST, BRAIN_WS_PORT, TRANSPORT_TOKEN
from robot.core.logging import configure_logging, get_logger
from robot.transport.websocket import BrainServer

log = get_logger(__name__)


async def amain():
    configure_logging()
    if not TRANSPORT_TOKEN:
        raise SystemExit(
            "TRANSPORT_TOKEN is not set. Generate one with\n"
            '  python -c "import secrets; print(secrets.token_hex(32))"\n'
            "and put it in .env on both machines."
        )
    log.info("brain_server_starting")
    brain = Agent()
    # Same prompt-cache prewarm as the inproc path: overlap it with the
    # server coming up so the first Edge request isn't the one that pays it.
    brain.start_prewarm()
    server = BrainServer(brain, BRAIN_WS_HOST, BRAIN_WS_PORT, TRANSPORT_TOKEN)
    log.info("brain_server_listening", host=BRAIN_WS_HOST, port=BRAIN_WS_PORT)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(amain())
