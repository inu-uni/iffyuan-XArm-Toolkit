"""Decoder for Teensy-side CoinFT MLP output frames."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Any


MAGIC = b"CFTF"
VERSION = 2
FRAME_SIZE = 65
AXES = ("fx", "fy", "fz", "mx", "my", "mz")
CALIBRATION_STATES = {
    0: "warming_up",
    1: "delayed",
    2: "provisional_tare",
    3: "runtime_settling",
    4: "final_tare",
    5: "bias_calibration",
    6: "zero_check",
    7: "ready",
    8: "zero_check_failed",
    9: "error",
}


class ProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class WrenchFrame:
    sequence: int
    teensy_time_us: int
    calibration_status: int
    wrench_valid: bool
    zero_check_warning: bool
    left: tuple[float, ...]
    right: tuple[float, ...]
    raw: bytes


def crc16_ccitt_false(data: bytes) -> int:
    crc = 0xFFFF
    for value in data:
        crc ^= value << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def encode_frame(
    sequence: int,
    teensy_time_us: int,
    left: tuple[float, ...],
    right: tuple[float, ...],
    calibration_status: int = 7,
    wrench_valid: bool = True,
    zero_check_warning: bool = False,
) -> bytes:
    if len(left) != 6 or len(right) != 6:
        raise ProtocolError("left and right must each contain six axes")
    values = tuple(float(value) for value in left + right)
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("wrench values must be finite")
    if calibration_status not in CALIBRATION_STATES:
        raise ProtocolError(f"unsupported calibration status: {calibration_status}")
    flags = (0x01 if wrench_valid else 0x00) | (0x02 if zero_check_warning else 0x00)
    body = bytes((VERSION, calibration_status, flags)) + struct.pack(
        "<II12f", sequence & 0xFFFFFFFF, teensy_time_us & 0xFFFFFFFF, *values
    )
    crc = crc16_ccitt_false(body)
    return MAGIC + body + struct.pack("<H", crc)


def decode_frame(raw: bytes) -> WrenchFrame:
    if len(raw) != FRAME_SIZE:
        raise ProtocolError(f"expected {FRAME_SIZE} bytes, got {len(raw)}")
    if raw[:4] != MAGIC:
        raise ProtocolError("bad frame magic")
    if raw[4] != VERSION:
        raise ProtocolError(f"unsupported protocol version: {raw[4]}")
    expected_crc = struct.unpack_from("<H", raw, 63)[0]
    actual_crc = crc16_ccitt_false(raw[4:63])
    if expected_crc != actual_crc:
        raise ProtocolError("CRC mismatch")
    calibration_status = raw[5]
    if calibration_status not in CALIBRATION_STATES:
        raise ProtocolError(f"unsupported calibration status: {calibration_status}")
    flags = raw[6]
    if flags & ~0x03:
        raise ProtocolError(f"unsupported frame flags: 0x{flags:02x}")
    if bool(flags & 0x01) != (calibration_status == 7):
        raise ProtocolError("wrench validity must match the ready calibration state")
    if flags & 0x02 and not flags & 0x01:
        raise ProtocolError("zero-check warning requires a valid wrench")
    sequence, teensy_time_us, *values = struct.unpack_from("<II12f", raw, 7)
    if not all(math.isfinite(value) for value in values):
        raise ProtocolError("frame contains a non-finite wrench value")
    return WrenchFrame(
        sequence,
        teensy_time_us,
        calibration_status,
        bool(flags & 0x01),
        bool(flags & 0x02),
        tuple(values[:6]),
        tuple(values[6:]),
        raw,
    )


class StreamParser:
    """Incremental parser that recovers after garbage and damaged frames."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.discarded_bytes = 0
        self.bad_frames = 0

    def feed(self, data: bytes) -> list[WrenchFrame]:
        self.buffer.extend(data)
        frames = []
        while True:
            start = self.buffer.find(MAGIC)
            if start < 0:
                keep = min(len(self.buffer), len(MAGIC) - 1)
                discard = len(self.buffer) - keep
                if discard:
                    self.discarded_bytes += discard
                    del self.buffer[:discard]
                break
            if start:
                self.discarded_bytes += start
                del self.buffer[:start]
            if len(self.buffer) < FRAME_SIZE:
                break
            candidate = bytes(self.buffer[:FRAME_SIZE])
            try:
                frame = decode_frame(candidate)
            except ProtocolError:
                self.bad_frames += 1
                self.discarded_bytes += 1
                del self.buffer[0]
                continue
            frames.append(frame)
            del self.buffer[:FRAME_SIZE]
        return frames


def sequence_gap(previous: int | None, current: int) -> int:
    if previous is None:
        return 0
    gap = (current - previous - 1) & 0xFFFFFFFF
    return gap if gap < 1_000_000 else 0


def sample_to_json(frame: WrenchFrame, host_receive_monotonic_ns: int, dropped_before: int) -> dict[str, Any]:
    state = CALIBRATION_STATES[frame.calibration_status]
    ready = state == "ready"
    return {
        "schema": "coinft-teensy-mlp-v2-sample",
        "host_receive_monotonic_ns": host_receive_monotonic_ns,
        "sequence_id": frame.sequence,
        "teensy_time_us": frame.teensy_time_us,
        "wire_format": "coinft-mlp-v2",
        "calibration_status": state,
        "calibration_status_id": frame.calibration_status,
        "calibration_ready": ready,
        "calibration_error": (
            "unloaded zero check failed" if state == "zero_check_failed" else "inference or calibration error" if state == "error" else None
        ),
        "wrench_valid": frame.wrench_valid,
        "zero_check_warning": frame.zero_check_warning,
        "warning": "zero check exceeded 0.15 N; wrench output remains valid"
        if frame.zero_check_warning
        else None,
        "side_order": ["left", "right"],
        "wrench_axes": list(AXES),
        "dropped_before": dropped_before,
        "left_wrench": list(frame.left) if frame.wrench_valid else None,
        "right_wrench": list(frame.right) if frame.wrench_valid else None,
        "left": dict(zip(AXES, frame.left)) if frame.wrench_valid else None,
        "right": dict(zip(AXES, frame.right)) if frame.wrench_valid else None,
    }
