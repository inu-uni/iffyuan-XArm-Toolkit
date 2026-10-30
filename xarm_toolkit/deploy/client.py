"""VLA inference client — runs on the robot machine.

Connects to a :class:`VLAServer` over WebSocket + msgpack and provides a
simple synchronous ``predict()`` method for the robot control loop.

Usage::

    client = VLAClient(host="192.168.1.100", port=10093)
    result = client.predict(
        images=[rgb_arm, rgb_fix],
        instruction="pick up the cup",
        state=obs["servo_angle"],
    )
    actions = result["actions"]   # np.ndarray
    client.close()
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from xarm_toolkit.deploy.msgpack_numpy import packb, unpackb
from xarm_toolkit.utils.logger import get_logger

try:
    from websockets.sync.client import connect as ws_connect
except ImportError as exc:
    raise ImportError(
        "websockets is required for VLAClient. Install with: pip install websockets"
    ) from exc

logger = get_logger("xarm_toolkit.deploy.client")


class VLAClient:
    """Synchronous WebSocket client for VLA inference.

    Parameters
    ----------
    host : str
        Server hostname / IP.
    port : int
        Server port.
    reconnect_interval : float
        Seconds between reconnection attempts.
    reconnect_timeout : float
        Give up after this many seconds of failed reconnections.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 10093,
        reconnect_interval: float = 2.0,
        reconnect_timeout: float = 300.0,
    ):
        self.url = f"ws://{host}:{port}"
        self.reconnect_interval = reconnect_interval
        self.reconnect_timeout = reconnect_timeout

        self._ws = None
        self.metadata: dict[str, Any] = {}

        self._connect()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self):
        """Establish (or re-establish) the WebSocket connection."""
        deadline = time.monotonic() + self.reconnect_timeout

        while True:
            try:
                logger.info("Connecting to %s ...", self.url)
                self._ws = ws_connect(
                    self.url,
                    max_size=50 * 1024 * 1024,
                    open_timeout=10,
                )
                # Read handshake metadata
                raw = self._ws.recv()
                resp = unpackb(raw)
                if resp.get("status") == "ok":
                    self.metadata = resp.get("data", {})
                    logger.info("Connected. Server metadata: %s", self.metadata)
                return

            except Exception as exc:
                elapsed = time.monotonic() - (deadline - self.reconnect_timeout)
                if time.monotonic() > deadline:
                    raise ConnectionError(
                        f"Failed to connect to {self.url} after "
                        f"{self.reconnect_timeout:.0f}s"
                    ) from exc
                logger.warning(
                    "Connection failed (%.1fs elapsed): %s — retrying in %.1fs",
                    elapsed, exc, self.reconnect_interval,
                )
                time.sleep(self.reconnect_interval)

    def _ensure_connected(self):
        """Reconnect if the socket is closed."""
        if self._ws is None:
            self._connect()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        images: list[np.ndarray],
        instruction: str,
        state: np.ndarray,
        coinft_left: np.ndarray | None = None,
        coinft_right: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Send an inference request and return the server's response.

        Parameters
        ----------
        images : list[np.ndarray]
            RGB images, typically ``[arm_rgb, fix_rgb]``.
        instruction : str
            Language instruction for the task.
        state : np.ndarray
            Robot state vector (e.g. ``servo_angle`` or ``cart_pos``).
        coinft_left, coinft_right : np.ndarray | None
            Optional six-axis CoinFT wrench vectors. Each is sent independently
            when provided as a top-level request field.

        Returns
        -------
        dict
            Server response data, containing at least ``"actions"`` (np.ndarray).
        """
        msg = {
            "type": "infer",
            "images": images,
            "instruction": instruction,
            "state": state,
        }
        if coinft_left is not None:
            msg["coinft_left"] = coinft_left
        if coinft_right is not None:
            msg["coinft_right"] = coinft_right

        try:
            self._ensure_connected()
            self._ws.send(packb(msg))
            raw = self._ws.recv()
            resp = unpackb(raw)
        except Exception:
            logger.warning("Connection lost — attempting reconnect ...")
            self._ws = None
            self._connect()
            # Retry once after reconnect
            self._ws.send(packb(msg))
            raw = self._ws.recv()
            resp = unpackb(raw)

        if resp.get("status") == "error":
            err = resp.get("error", {})
            raise RuntimeError(
                f"Server error: {err.get('message', 'unknown')} — "
                f"{err.get('detail', '')}"
            )

        return resp.get("data", {})

    def ping(self) -> bool:
        """Send a ping and return True if the server responds."""
        try:
            self._ensure_connected()
            self._ws.send(packb({"type": "ping"}))
            raw = self._ws.recv()
            resp = unpackb(raw)
            return resp.get("status") == "ok"
        except Exception:
            return False

    def close(self):
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            logger.info("Connection closed.")
