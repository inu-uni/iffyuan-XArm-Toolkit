# CoinFT sensor visualization, migrated from coinft/teensy_mlp_workspace/visualization.
"""Shared Matplotlib entry points for Teensy CoinFT wrench visualizers."""

from __future__ import annotations

import argparse
from collections import deque
import queue
import threading
import time
from typing import Any

from xarm_toolkit.sensors.coinft.coinft_interface import CoinFTInterface, CoinFTSample


AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
COLORS = ("red", "green", "blue")


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--history", type=float, default=5.0, help="seconds shown on the x axis")
    parser.add_argument("--no-start-command", action="store_true", help="attach without restarting calibration")
    return parser


def _sample_wrenches(sample: CoinFTSample):
    if not sample.wrench_valid or sample.left_wrench is None or sample.right_wrench is None:
        return None
    return {"left": sample.left_wrench, "right": sample.right_wrench}


class TeensyTimebase:
    """Convert the wrapping Teensy micros() counter into elapsed seconds."""

    def __init__(self) -> None:
        self.previous_us: int | None = None
        self.elapsed_s = 0.0

    def update(self, micros: int) -> float:
        micros &= 0xFFFFFFFF
        if self.previous_us is not None:
            delta_us = (micros - self.previous_us) & 0xFFFFFFFF
            self.elapsed_s += delta_us / 1_000_000.0
        self.previous_us = micros
        return self.elapsed_s


def _read_samples(
    args: argparse.Namespace,
    messages: queue.Queue,
    control: queue.Queue,
    stop: threading.Event,
) -> None:
    try:
        messages.put(("status", "正在连接 Teensy..."))
        with CoinFTInterface(
            args.port, args.baud, start_stream=not args.no_start_command
        ) as source:
            messages.put(("status", "串口已连接，等待数据帧..."))
            while not stop.is_set():
                try:
                    command = control.get_nowait()
                except queue.Empty:
                    command = None
                if command == "zero":
                    messages.put(("status", "正在重启传感器并执行板端归零..."))
                    source.restart_calibration()
                    messages.put(("reset_plot", None))
                try:
                    messages.put(("sample", source.read_sample(timeout=0.25)))
                except TimeoutError:
                    messages.put(("stats", source.parser_stats))
    except Exception as exc:
        messages.put(("error", f"串口读取异常: {type(exc).__name__}: {exc}"))


def run_dual(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    plt.ion()
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), squeeze=False, sharex=True)
    figure.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.18, wspace=0.30)
    side_plots = {}
    axis_colors = COLORS + COLORS
    for column, side in enumerate(("left", "right")):
        force_axis = axes[0][column]
        moment_axis = force_axis.twinx()
        force_axis.set_title(f"{side.title()} CoinFT")
        force_axis.set_ylabel("Force (N)")
        moment_axis.set_ylabel("Moment (Nm)")
        force_axis.set_xlabel("Time (s)")
        force_axis.grid(True, alpha=0.35)
        force_lines = [
            force_axis.plot([], [], color=axis_colors[index], linestyle="-", linewidth=0.9, label=name)[0]
            for index, name in enumerate(AXES[:3])
        ]
        moment_lines = [
            moment_axis.plot([], [], color=axis_colors[index + 3], linestyle="-", linewidth=0.9, label=name)[0]
            for index, name in enumerate(AXES[3:])
        ]
        force_axis.legend(force_lines + moment_lines, [line.get_label() for line in force_lines + moment_lines],
                          loc="upper left", ncol=3, fontsize="small")
        side_plots[side] = (force_axis, moment_axis, force_lines, moment_lines)
    status = figure.suptitle("正在连接 Teensy...", fontsize=12)
    control_queue: queue.Queue = queue.Queue()
    from matplotlib.widgets import Button

    button_axis = figure.add_axes((0.015, 0.035, 0.15, 0.065))
    zero_button = Button(button_axis, "重新归零", color="#f1f3f5", hovercolor="#dfe3e8")
    zero_state = {"pending": False}

    def request_zero(_event: Any) -> None:
        if zero_state["pending"]:
            return
        zero_state["pending"] = True
        zero_button.label.set_text("校准中...")
        zero_button.ax.set_facecolor("#e9ecef")
        control_queue.put("zero")
        status.set_text("正在执行板端归零；首次上电约 28-30 秒，后续约 3 秒；请保持空载")
        figure.canvas.draw_idle()

    zero_button.on_clicked(request_zero)
    plt.show(block=False)

    history = deque()
    messages: queue.Queue = queue.Queue()
    stop_reader = threading.Event()
    reader = threading.Thread(
        target=_read_samples,
        args=(args, messages, control_queue, stop_reader),
        daemon=True,
    )
    start_s = time.monotonic()
    last_draw_s = start_s
    samples_since_draw = 0
    last_frame_s: float | None = None
    last_status = "等待数据帧"
    parser_stats = (0, 0)
    reader_error: str | None = None
    valid_count = 0
    last_teensy_time_us = None
    measured_hz = 0.0
    sample = None
    device_time = TeensyTimebase()
    latest_sample_s = 0.0
    previous_valid_time_s: float | None = None
    reader.start()
    try:
        while plt.fignum_exists(figure.number):
            now_s = time.monotonic()
            while True:
                try:
                    kind, payload = messages.get_nowait()
                except queue.Empty:
                    break
                if kind == "sample":
                    sample = payload
                    last_frame_s = now_s
                    samples_since_draw += 1
                    last_status = sample.calibration_status
                    if sample.calibration_status in ("ready", "zero_check_failed", "error"):
                        zero_state["pending"] = False
                        zero_button.label.set_text("重新归零")
                        zero_button.ax.set_facecolor("#f1f3f5")
                    sample_s = device_time.update(sample.teensy_time_us)
                    latest_sample_s = sample_s
                    if sample.wrench_valid:
                        valid_count += 1
                        if previous_valid_time_s is not None and sample_s > previous_valid_time_s:
                            measured_hz = 1.0 / (sample_s - previous_valid_time_s)
                        last_teensy_time_us = sample.teensy_time_us
                        previous_valid_time_s = sample_s
                        values = _sample_wrenches(sample)
                        history.append((sample_s, values))
                        cutoff = sample_s - args.history
                        while history and history[0][0] < cutoff:
                            history.popleft()
                elif kind == "status":
                    last_status = payload
                elif kind == "stats":
                    parser_stats = payload
                elif kind == "reset_plot":
                    history.clear()
                    sample = None
                    last_frame_s = now_s
                    last_status = "等待 Teensy 重新校准"
                    valid_count = 0
                    measured_hz = 0.0
                    previous_valid_time_s = None
                    device_time = TeensyTimebase()
                    latest_sample_s = 0.0
                    samples_since_draw = 0
                    for force_axis, moment_axis, force_lines, moment_lines in side_plots.values():
                        for line in force_lines + moment_lines:
                            line.set_data([], [])
                        force_axis.set_xlim(0.0, args.history)
                elif kind == "error":
                    reader_error = payload
                    zero_state["pending"] = False
                    zero_button.label.set_text("重新归零")
                    zero_button.ax.set_facecolor("#f1f3f5")

            if last_frame_s is None:
                age_s = now_s - start_s
            else:
                age_s = now_s - last_frame_s

            if reader_error:
                message = reader_error
                status.set_color("crimson")
            elif age_s > 2.0:
                message = (
                    f"异常: {age_s:.1f}s 未收到完整数据帧 | CRC错误 {parser_stats[0]}，"
                    f"丢弃字节 {parser_stats[1]} | 检查固件发送、USB连接和波特率"
                )
                status.set_color("crimson")
            elif sample is None:
                message = last_status
                status.set_color("black")
            elif sample.calibration_status == "zero_check_failed":
                message = "异常: Teensy 零点校验未通过 | 输出无效，请检查传感器是否空载并重新归零"
                status.set_color("crimson")
            elif sample.calibration_status == "error":
                message = "异常: Teensy 推理或校准失败 | 请检查固件状态"
                status.set_color("crimson")
            elif sample.zero_check_warning:
                message = (
                    f"提示: 归零检查窗口受到扰动，六轴数据仍有效并持续采集 | "
                    f"有效帧 {valid_count} | 约 {measured_hz:.0f} Hz | 丢帧 {sample.dropped_before}"
                )
                status.set_color("darkorange")
            else:
                message = (
                    f"状态: {sample.calibration_status} | 有效帧 {valid_count} | "
                    f"约 {measured_hz:.0f} Hz | 丢帧 {sample.dropped_before}"
                )
                status.set_color("black")

            if samples_since_draw >= 40 or now_s - last_draw_s >= 0.25:
                if history:
                    stamps = [row[0] for row in history]
                    newest = latest_sample_s
                    for side in ("left", "right"):
                        force_axis, moment_axis, force_lines, moment_lines = side_plots[side]
                        for index, line in enumerate(force_lines):
                            line.set_data(stamps, [row[1][side][index] for row in history])
                        for index, line in enumerate(moment_lines, start=3):
                            line.set_data(stamps, [row[1][side][index] for row in history])
                        force_values = [row[1][side][index] for row in history for index in range(3)]
                        moment_values = [row[1][side][index] for row in history for index in range(3, 6)]
                        for axis, values, floor_span in (
                            (force_axis, force_values, 0.1),
                            (moment_axis, moment_values, 0.01),
                        ):
                            low, high = min(values), max(values)
                            span = max(high - low, floor_span)
                            padding = max(span * 0.15, floor_span * 0.5)
                            desired = (low - padding, high + padding)
                            current = axis.get_ylim()
                            if desired[0] < current[0] or desired[1] > current[1]:
                                axis.set_ylim(min(desired[0], current[0]), max(desired[1], current[1]))
                            elif (current[1] - current[0]) > (desired[1] - desired[0]) * 1.4:
                                axis.set_ylim(desired)
                        force_axis.set_xlim(max(0.0, newest - args.history), max(args.history, newest))
                status.set_text(message)
                figure.canvas.draw_idle()
                plt.pause(0.001)
                last_draw_s = now_s
                samples_since_draw = 0
    except KeyboardInterrupt:
        pass
    finally:
        stop_reader.set()
        plt.close(figure)
        reader.join(timeout=1.0)
    return 0


def run_axis_compare(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    plt.ion()
    figure, axes = plt.subplots(3, 2, figsize=(12, 8), squeeze=False, sharex=True)
    axes_flat = list(axes.flat)
    lines = {}
    for index, name in enumerate(AXES):
        axis = axes_flat[index]
        axis.set_title(name)
        axis.set_ylabel("N" if index < 3 else "Nm")
        axis.grid(True, alpha=0.3)
        lines[index] = {
            side: axis.plot([], [], color=COLORS[index % 3], linestyle=style, label=side.title())[0]
            for side, style in (("left", "-"), ("right", "--"))
        }
        axis.legend(loc="upper right")
    for axis in axes_flat[-2:]:
        axis.set_xlabel("Time [s]")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    status = figure.suptitle("Waiting for CoinFT calibration")
    plt.show(block=False)

    history = deque()
    start_s = time.monotonic()
    last_draw_s = start_s
    valid_count = 0
    last_teensy_time_us = None
    measured_hz = 0.0
    try:
        with CoinFTInterface(args.port, args.baud, start_stream=not args.no_start_command) as source:
            while plt.fignum_exists(figure.number):
                sample = source.read_sample()
                now_s = time.monotonic()
                values = _sample_wrenches(sample)
                if values is not None:
                    valid_count += 1
                    if last_teensy_time_us is not None:
                        interval_us = (sample.teensy_time_us - last_teensy_time_us) & 0xFFFFFFFF
                        if interval_us > 0:
                            measured_hz = 1_000_000.0 / interval_us
                    last_teensy_time_us = sample.teensy_time_us
                    stamp = now_s - start_s
                    history.append((stamp, values))
                    cutoff = stamp - args.history
                    while history and history[0][0] < cutoff:
                        history.popleft()

                if now_s - last_draw_s >= 0.05:
                    if history:
                        stamps = [row[0] for row in history]
                        for axis_index, axis_lines in lines.items():
                            for side in ("left", "right"):
                                axis_lines[side].set_data(
                                    stamps, [row[1][side][axis_index] for row in history]
                                )
                            axes_flat[axis_index].relim()
                            axes_flat[axis_index].autoscale_view()
                    status.set_text(
                        f"{sample.calibration_status} | valid={valid_count} | ~{measured_hz:.0f} Hz | "
                        f"dropped={sample.dropped_before}"
                    )
                    figure.canvas.draw_idle()
                    figure.canvas.flush_events()
                    plt.pause(0.001)
                    last_draw_s = now_s
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(figure)
    return 0
