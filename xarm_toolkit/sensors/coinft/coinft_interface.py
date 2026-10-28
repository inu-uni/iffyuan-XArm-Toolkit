"""Small reusable serial API for Teensy-produced CoinFT wrench samples."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Any, Callable

try:
    import serial
except ImportError:  # pragma: no cover - environment dependent
    serial = None

from .protocol import StreamParser, WrenchFrame, sample_to_json, sequence_gap


class CalibrationFailedError(RuntimeError):
    """The Teensy reported an inference or calibration failure."""


class ZeroCheckFailedError(CalibrationFailedError):
    """The unloaded force zero check did not pass."""


@dataclass(frozen=True)
class CoinFTSample:
    sequence: int
    teensy_time_us: int
    calibration_status: str
    calibration_status_id: int
    calibration_ready: bool
    wrench_valid: bool
    zero_check_warning: bool
    left_wrench: tuple[float, ...] | None
    right_wrench: tuple[float, ...] | None
    dropped_before: int
    host_receive_monotonic_ns: int

    @classmethod
    def from_frame(
        cls, frame: WrenchFrame, host_receive_monotonic_ns: int, dropped_before: int
    ) -> "CoinFTSample":
        row = sample_to_json(frame, host_receive_monotonic_ns, dropped_before)
        return cls(
            sequence=frame.sequence,
            teensy_time_us=frame.teensy_time_us,
            calibration_status=row["calibration_status"],
            calibration_status_id=frame.calibration_status,
            calibration_ready=row["calibration_ready"],
            wrench_valid=frame.wrench_valid,
            zero_check_warning=frame.zero_check_warning,
            left_wrench=frame.left if frame.wrench_valid else None,
            right_wrench=frame.right if frame.wrench_valid else None,
            dropped_before=dropped_before,
            host_receive_monotonic_ns=host_receive_monotonic_ns,
        )

    def as_dict(self) -> dict[str, Any]:
        axes = ("fx", "fy", "fz", "mx", "my", "mz")
        left = dict(zip(axes, self.left_wrench)) if self.left_wrench is not None else None
        right = dict(zip(axes, self.right_wrench)) if self.right_wrench is not None else None
        return {
            "schema": "coinft-teensy-mlp-v2-sample",
            "host_receive_monotonic_ns": self.host_receive_monotonic_ns,
            "sequence_id": self.sequence,
            "teensy_time_us": self.teensy_time_us,
            "wire_format": "coinft-mlp-v2",
            "calibration_status": self.calibration_status,
            "calibration_status_id": self.calibration_status_id,
            "calibration_ready": self.calibration_ready,
            "calibration_error": (
                "unloaded zero check failed"
                if self.calibration_status == "zero_check_failed"
                else "inference or calibration error"
                if self.calibration_status == "error"
                else None
            ),
            "wrench_valid": self.wrench_valid,
            "zero_check_warning": self.zero_check_warning,
            "warning": (
                "zero check exceeded 0.15 N; wrench output remains valid"
                if self.zero_check_warning
                else None
            ),
            "side_order": ["left", "right"],
            "wrench_axes": ["fx", "fy", "fz", "mx", "my", "mz"],
            "dropped_before": self.dropped_before,
            "left_wrench": list(self.left_wrench) if self.left_wrench is not None else None,
            "right_wrench": list(self.right_wrench) if self.right_wrench is not None else None,
            "left": left,
            "right": right,
        }


class CoinFTInterface:
    """Read validated samples from one Teensy USB serial port.

    The default startup sends ``i`` then ``s``. This stops and restarts the
    sensor stream and triggers Teensy calibration for the new acquisition. The
    first calibration after board power-up runs the long stability sequence;
    later starts use the short tare sequence.
    """

    def __init__(
        self,
        port: str,
        baud: int = 115200,
        *,
        start_stream: bool = True,
        startup_delay: float = 1.0,
        read_timeout: float = 0.05,
        serial_factory: Callable[..., Any] | None = None,
    ) -> None:
        if serial_factory is None:
            if serial is None:
                raise RuntimeError("pyserial is required: python3 -m pip install pyserial")
            serial_factory = serial.Serial
        self._port = serial_factory(port, baud, timeout=read_timeout, write_timeout=1.0)
        self._parser = StreamParser()
        self._pending: deque[tuple[WrenchFrame, int]] = deque()
        self._previous_sequence: int | None = None
        self._started = False
        self._closed = False
        if start_stream:
            try:
                time.sleep(max(0.0, startup_delay))
                self._send(b"i")
                time.sleep(0.2)
                self._port.reset_input_buffer()
                self._send(b"s")
                self._started = True
            except Exception:
                self._port.close()
                raise

    def _send(self, command: bytes) -> None:
        self._port.write(command)
        self._port.flush()

    def read_sample(self, timeout: float | None = None) -> CoinFTSample:
        """Read a complete sample, optionally timing out while waiting for a frame."""
        if self._closed:
            raise RuntimeError("CoinFT interface is closed")
        deadline = time.monotonic() + timeout if timeout is not None else None
        while not self._pending:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("no complete CoinFT frame before timeout")
            chunk = self._port.read(max(1, self._port.in_waiting))
            if not chunk:
                continue
            received_ns = time.monotonic_ns()
            self._pending.extend((frame, received_ns) for frame in self._parser.feed(chunk))

        frame, received_ns = self._pending.popleft()
        dropped = sequence_gap(self._previous_sequence, frame.sequence)
        self._previous_sequence = frame.sequence
        return CoinFTSample.from_frame(frame, received_ns, dropped)

    def wait_until_ready(self, timeout: float | None = None) -> CoinFTSample:
        """Wait for the first valid wrench sample or raise on failure/timeout."""
        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0.0:
                raise TimeoutError("CoinFT calibration did not reach ready before timeout")
            sample = self.read_sample(timeout=remaining)
            if sample.calibration_status == "zero_check_failed":
                raise ZeroCheckFailedError(
                    "Teensy unloaded zero check failed; discard this acquisition and "
                    "check that both sensors are unloaded and still before retrying"
                )
            if sample.calibration_status == "error":
                raise CalibrationFailedError("Teensy reported an inference or calibration error")
            if sample.wrench_valid:
                return sample

    @property
    def parser_stats(self) -> tuple[int, int]:
        """Return the number of rejected frames and discarded stream bytes."""
        return self._parser.bad_frames, self._parser.discarded_bytes

    def restart_calibration(self, restart_delay: float = 0.2) -> None:
        """Restart sensor streams and trigger board-side tare for a new run."""
        if self._closed:
            raise RuntimeError("CoinFT interface is closed")
        self._send(b"i")
        time.sleep(max(0.0, restart_delay))
        self._port.reset_input_buffer()
        self._parser = StreamParser()
        self._pending.clear()
        self._previous_sequence = None
        self._send(b"s")
        self._started = True

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._started:
                self._send(b"i")
        finally:
            self._port.close()
            self._closed = True

    def __enter__(self) -> "CoinFTInterface":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
