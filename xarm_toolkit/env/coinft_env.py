"""CoinFT environment with continuous reception and latest-sample observations."""

from __future__ import annotations

import atexit
import math
import threading
import time
from typing import Any

import numpy as np

from xarm_toolkit.sensors.coinft.coinft_interface import (
    CalibrationFailedError,
    CoinFTInterface,
    CoinFTSample,
    ZeroCheckFailedError,
)
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("xarm_toolkit.env.coinft")


class CoinFTError(RuntimeError):
    """The CoinFT connection is closed or its receiver failed."""


class CoinFTNotReadyError(CoinFTError):
    """No valid sample from the current calibration is available."""


class CoinFTEnv:
    """Read the latest dual-sensor CoinFT observation without a consumer backlog.

    Construction opens the Teensy USB port; call ``reset()`` while unloaded to
    start streaming and wait for board-side calibration. One worker owns all
    subsequent serial operations. ``step()`` returns immediately and may return
    the same sample repeatedly, without comparing sequence IDs.

    Parameters
    ----------
    port:
        Teensy USB serial port, e.g. ``"COM3"`` or ``"/dev/ttyACM0"``.
    baud:
        Serial interface setting passed to the original CoinFT receiver.
    max_sample_age:
        Maximum time since the sample's host reception, in seconds. This is
        a reception-health check, not a sensor-to-host latency measurement.
    read_timeout:
        Finite serial-read timeout, allowing reset and cleanup requests to be
        handled even when the device stops sending frames.
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        *,
        max_sample_age: float = 0.1,
        read_timeout: float = 0.05,
    ) -> None:
        for name, value in (("max_sample_age", max_sample_age), ("read_timeout", read_timeout)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.max_sample_age = max_sample_age
        self._read_timeout = read_timeout
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._closed = False
        self._reader_error: Exception | None = None
        self._latest: CoinFTSample | None = None
        self._generation = 0
        self._reset_requested = False
        self._reset_in_progress = False
        self._stream_started = False
        self._awaiting_calibration = False
        self._received_frames = 0
        self._dropped_frames = 0
        self._parser_stats = (0, 0)
        self._first_receive_ns: int | None = None

        # Opening the port does not send commands or start a second calibration.
        self._interface = CoinFTInterface(
            port, baud, start_stream=False, read_timeout=read_timeout,
        )
        self._thread = threading.Thread(
            target=self._receive_loop, name="coinft-receiver", daemon=True,
        )
        try:
            self._thread.start()
        except Exception:
            self._interface.close()
            raise
        atexit.register(self.cleanup)
        logger.info("CoinFT connected at %s; call reset() while unloaded", port)

    # ------------------------------------------------------------------
    # Background reception: the sole owner of serial commands/read/close.
    # ------------------------------------------------------------------

    def _receive_loop(self) -> None:
        try:
            while not self._stop.is_set():
                with self._condition:
                    while not (
                        self._stop.is_set() or self._stream_started or self._reset_requested
                    ):
                        self._condition.wait()
                    if self._stop.is_set():
                        break
                    generation = self._generation
                    restart = self._reset_requested
                    self._reset_requested = False

                if restart:
                    self._interface.restart_calibration()
                    with self._condition:
                        self._stream_started = True
                    continue

                try:
                    sample = self._interface.read_sample(timeout=self._read_timeout)
                except TimeoutError:
                    with self._condition:
                        if generation == self._generation and not self._reset_requested:
                            self._parser_stats = self._interface.parser_stats
                    continue

                with self._condition:
                    # A reset may have been requested while read_sample blocked.
                    if generation != self._generation or self._reset_requested:
                        continue
                    previous = self._latest
                    self._latest = sample
                    self._received_frames += 1
                    self._dropped_frames += sample.dropped_before
                    self._parser_stats = self._interface.parser_stats
                    if self._first_receive_ns is None:
                        self._first_receive_ns = sample.host_receive_monotonic_ns
                    # Do not mistake a residual ready frame for the new tare.
                    # Current firmware emits non-ready frames during every tare.
                    if sample.calibration_status_id in range(7):
                        self._awaiting_calibration = False
                    self._condition.notify_all()

                if previous is None or previous.calibration_status != sample.calibration_status:
                    logger.info("CoinFT calibration: %s", sample.calibration_status)
                if sample.zero_check_warning and (
                    previous is None or not previous.zero_check_warning
                ):
                    logger.warning("CoinFT zero-check warning; six-axis output remains valid")
        except Exception as exc:
            with self._condition:
                self._reader_error = exc
                self._condition.notify_all()
            logger.error("CoinFT receiver failed: %s", exc)
        finally:
            try:
                self._interface.close()
            except Exception as exc:
                with self._condition:
                    if self._reader_error is None:
                        self._reader_error = exc
                logger.warning("CoinFT close failed: %s", exc)
            finally:
                with self._condition:
                    self._stream_started = False
                    self._condition.notify_all()

    # ------------------------------------------------------------------
    # Observation helpers (sample state is guarded by _condition).
    # ------------------------------------------------------------------

    def _check_connection(self) -> None:
        if self._closed:
            raise CoinFTError("CoinFT environment is closed")
        if self._reader_error is not None:
            raise CoinFTError(
                f"CoinFT receiver failed: {self._reader_error}"
            ) from self._reader_error

    @staticmethod
    def _check_calibration(sample: CoinFTSample) -> None:
        if sample.calibration_status == "zero_check_failed":
            raise ZeroCheckFailedError("CoinFT unloaded zero check failed; check load and reset")
        if sample.calibration_status == "error":
            raise CalibrationFailedError("CoinFT reported an inference or calibration error")

    @staticmethod
    def _sample_age(sample: CoinFTSample) -> float:
        return (time.monotonic_ns() - sample.host_receive_monotonic_ns) / 1e9

    @staticmethod
    def _observation(sample: CoinFTSample) -> dict[str, Any]:
        return {
            "left_wrench": np.array(sample.left_wrench, dtype=np.float32),
            "right_wrench": np.array(sample.right_wrench, dtype=np.float32),
            "sequence": sample.sequence,
            "teensy_time_us": sample.teensy_time_us,
            "host_receive_monotonic_ns": sample.host_receive_monotonic_ns,
            "calibration_status": sample.calibration_status,
            "wrench_valid": sample.wrench_valid,
            "zero_check_warning": sample.zero_check_warning,
        }

    def _get_observation(self) -> dict[str, Any]:
        with self._condition:
            self._check_connection()
            sample = self._latest
            if sample is not None:
                self._check_calibration(sample)
            if sample is None or self._awaiting_calibration or not sample.wrench_valid:
                raise CoinFTNotReadyError(
                    "CoinFT is not ready; call reset() and wait for calibration"
                )
            age = self._sample_age(sample)
            if age > self.max_sample_age:
                raise TimeoutError(
                    f"CoinFT sample is {age:.3f}s old (limit {self.max_sample_age:.3f}s)"
                )
        return self._observation(sample)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, timeout: float = 60.0) -> dict[str, Any]:
        """Request board-side tare and wait for a fresh, valid observation.

        Both sensors must remain unloaded and still. Old observations become
        unavailable immediately. Timing out stops this wait, not the board's
        calibration; call reset again to restart, or inspect get_status().
        """
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        deadline = time.monotonic() + timeout
        with self._condition:
            self._check_connection()
            if self._reset_in_progress:
                raise CoinFTError("CoinFT reset is already in progress")
            self._reset_in_progress = True
            self._generation += 1
            self._latest = None
            self._reset_requested = True
            self._awaiting_calibration = True
            self._received_frames = 0
            self._dropped_frames = 0
            self._parser_stats = (0, 0)
            self._first_receive_ns = None
            self._condition.notify_all()
            try:
                while True:
                    self._check_connection()
                    sample = self._latest
                    if sample is not None:
                        self._check_calibration(sample)
                        if (not self._awaiting_calibration and sample.wrench_valid
                                and self._sample_age(sample) <= self.max_sample_age):
                            break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("CoinFT calibration did not reach ready before timeout")
                    self._condition.wait(remaining)
            finally:
                self._reset_in_progress = False
        return self._observation(sample)

    def step(self, action: Any = None) -> dict[str, Any]:
        """Return the latest valid sample; repeated samples are allowed.

        No sequence comparison or wait for a new frame is performed. Raises
        TimeoutError if the host-reception age exceeds max_sample_age.
        """
        return self._get_observation()

    def get_status(self) -> dict[str, Any]:
        """Return reception/calibration diagnostics, even before ready or after failure.

        Counters and the average receive rate cover the current reset session.
        Host receive rate can temporarily exceed device rate when draining a
        buffered burst; neither it nor sample_age measures end-to-end latency.
        """
        with self._condition:
            sample = self._latest
            elapsed_ns = (
                sample.host_receive_monotonic_ns - self._first_receive_ns
                if sample is not None and self._first_receive_ns is not None else 0
            )
            return {
                "connected": not self._closed and self._reader_error is None,
                "streaming": self._stream_started and not self._closed,
                "calibrating": self._reset_requested or self._awaiting_calibration or (
                    sample is not None and sample.calibration_status_id in range(7)
                ),
                "calibration_status": sample.calibration_status if sample else None,
                "wrench_valid": bool(
                    sample and sample.wrench_valid and not self._awaiting_calibration
                    and not self._closed and self._reader_error is None
                    and self._sample_age(sample) <= self.max_sample_age
                ),
                "zero_check_warning": sample.zero_check_warning if sample else False,
                "sequence": sample.sequence if sample else None,
                "sample_age": self._sample_age(sample) if sample else None,
                "received_frames": self._received_frames,
                "receive_rate_hz": (self._received_frames - 1) * 1e9 / elapsed_ns
                if elapsed_ns > 0 else None,
                "dropped_frames": self._dropped_frames,
                "bad_frames": self._parser_stats[0],
                "discarded_bytes": self._parser_stats[1],
                "last_error": str(self._reader_error) if self._reader_error else None,
            }

    def cleanup(self) -> None:
        """Stop the sole serial worker and release the port; safe to repeat."""
        with self._condition:
            self._closed = True
            self._stop.set()
            self._condition.notify_all()
        self._thread.join(timeout=self._read_timeout + 3.0)
        if self._thread.is_alive():
            raise CoinFTError("CoinFT receiver did not stop before the cleanup timeout")
        atexit.unregister(self.cleanup)
