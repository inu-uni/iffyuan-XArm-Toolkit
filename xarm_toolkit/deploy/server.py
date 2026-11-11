"""VLA inference server — runs on the GPU machine.

Serves a VLA policy (e.g. Pi0.5) over WebSocket + msgpack so that the
robot-side client can request action predictions without needing a local GPU.

Usage::

    server = VLAServer(policy=my_policy, port=10093)
    server.run()          # blocking
    # or: asyncio.run(server.serve())
"""

from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any, Protocol, runtime_checkable

from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb
from xarm_toolkit.utils.logger import get_logger

try:
    import websockets
    from websockets.asyncio.server import serve as ws_serve
except ImportError as exc:
    raise ImportError(
        "websockets is required for VLAServer. Install with: pip install websockets"
    ) from exc

logger = get_logger("xarm_toolkit.deploy.server")


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PolicyProtocol(Protocol):
    """Minimal interface that a VLA policy must implement."""

    def predict_action(self, **kwargs) -> dict[str, Any]:
        """Return a dict containing at least ``"actions"``."""
        ...


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _ok_response(data: dict) -> bytes:
    return packb({"status": "ok", "data": data})


def _error_response(msg: str, detail: str = "") -> bytes:
    return packb({"status": "error", "error": {"message": msg, "detail": detail}})


# ---------------------------------------------------------------------------
# VLAServer
# ---------------------------------------------------------------------------

class VLAServer:
    """Asynchronous WebSocket server that wraps a VLA policy.

    Parameters
    ----------
    policy : PolicyProtocol
        Any object with a ``predict_action(**kwargs) -> dict`` method.
    host : str
        Bind address.
    port : int
        Bind port.
    idle_timeout : float
        Seconds of inactivity before auto-closing a connection (0 = disabled).
    metadata : dict | None
        Extra info sent to clients on connect (model name, action format, etc.).
    """

    def __init__(
        self,
        policy: PolicyProtocol,
        host: str = "0.0.0.0",
        port: int = 10093,
        idle_timeout: float = 0,
        metadata: dict | None = None,
    ):
        self.policy = policy
        self.host = host
        self.port = port
        self.idle_timeout = idle_timeout
        self.metadata = metadata or {}

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle(self, ws):
        """Handle one WebSocket connection."""
        remote = ws.remote_address
        logger.info("Client connected: %s", remote)

        # Handshake — send metadata
        await ws.send(packb({
            "status": "ok",
            "data": {"type": "metadata", **self.metadata},
        }))

        last_active = time.monotonic()

        try:
            async for raw in ws:
                last_active = time.monotonic()

                try:
                    msg = unpackb(raw)
                except Exception:
                    await ws.send(_error_response("bad_payload", "Failed to decode msgpack"))
                    continue

                msg_type = msg.get("type", "")

                if msg_type == "ping":
                    await ws.send(_ok_response({"type": "pong"}))

                elif msg_type == "reset":
                    try:
                        self.policy.reset()
                        await ws.send(_ok_response({"type": "reset"}))
                    except Exception as exc:
                        tb = traceback.format_exc()
                        logger.error("Policy reset error: %s\n%s", exc, tb)
                        await ws.send(_error_response("reset_error", str(exc)))

                elif msg_type == "infer":
                    try:
                        # Strip "type" key, pass everything else to policy
                        kwargs = {k: v for k, v in msg.items() if k != "type"}
                        result = self.policy.predict_action(**kwargs)
                        await ws.send(_ok_response({"type": "actions", **result}))
                    except Exception as exc:
                        tb = traceback.format_exc()
                        logger.error("Inference error: %s\n%s", exc, tb)
                        await ws.send(_error_response(
                            "inference_error", str(exc),
                        ))
                else:
                    await ws.send(_error_response(
                        "unknown_type", f"Unknown message type: {msg_type!r}"
                    ))

                # Idle timeout check
                if self.idle_timeout > 0 and (time.monotonic() - last_active) > self.idle_timeout:
                    logger.info("Idle timeout reached for %s", remote)
                    break

        except Exception:
            pass
        finally:
            logger.info("Client disconnected: %s", remote)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def serve(self):
        """Start the server (coroutine)."""
        logger.info("Starting VLA server on %s:%d ...", self.host, self.port)
        async with ws_serve(self._handle, self.host, self.port, max_size=50 * 1024 * 1024):
            await asyncio.Future()  # run forever

    def run(self):
        """Blocking entry point."""
        asyncio.run(self.serve())
