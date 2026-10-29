#!/usr/bin/env python3
"""交互式回放 xArm Toolkit 采集的 Zarr 数据集。

用法::

    python scripts/data/replay_zarr.py --input datasets/demo.zarr --addr 192.168.1.XXX

脚本以只读方式打开数据集。回放前请确认机械臂周围安全、末端工具和采集时一致。

每个要回放的 episode 必须包含
``meta/episode_config/episode_XXXXXX``，且其 ``start_bias`` attribute 为
有限的 6 维偏置向量。回放会先复位到 Home，再按记录偏置移动，之后按序执行
该 episode 的动作和夹爪数据。缺少偏置元数据时，该 episode 不会回退到其他开局移动方式。
夹爪原始开合范围从同组的 ``gripper_range`` attribute 读取；缺失或无效时跳过该 episode。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from xarm_toolkit.env.xarm_env import XArmEnv


@dataclass(frozen=True)
class Episode:
    """连续存储的一个 episode 区间。"""

    label: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


class KeyboardListener:
    """跨平台读取单键；仅回放期间读取单键；POSIX 使用 cbreak，保留 Ctrl+C 和正常换行。"""

    def __init__(self):
        self._windows = os.name == "nt"
        self._old_settings = None

    def start(self):
        if self._windows:
            import msvcrt  # noqa: F401
            return
        import termios
        import tty

        if not sys.stdin.isatty():
            raise RuntimeError("键盘交互需要在终端中运行")
        self._termios = termios
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

    def stop(self):
        if not self._windows and self._old_settings is not None:
            self._termios.tcsetattr(
                sys.stdin, self._termios.TCSADRAIN, self._old_settings
            )
            self._old_settings = None

    def get_key(self) -> str | None:
        if self._windows:
            import msvcrt

            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key in ("\x00", "\xe0"):
                    msvcrt.getwch()
                    return None
                return key.lower()
            return None

        import select

        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1).lower()
        return None


def _episode_ranges(data: zarr.Group) -> list[Episode]:
    """从逐帧 episode 标签恢复连续区间，不依赖或修改 episode_ends。"""
    if "episode" not in data:
        raise ValueError("数据集缺少 data/episode")
    labels = np.asarray(data["episode"][:]).reshape(-1)
    if labels.size == 0:
        raise ValueError("数据集没有任何帧")

    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(labels)]
    episodes = [Episode(int(labels[s]), int(s), int(e)) for s, e in zip(starts, ends)]
    for episode in episodes:
        if np.any(labels[episode.start:episode.end] != episode.label):
            raise ValueError("episode 标签不是连续分段，无法安全回放")
    return episodes


def _validate_dataset(data: zarr.Group, episodes: list[Episode]):
    required = ("action", "pos", "gripper_action")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"缺少回放所需数组: {', '.join(missing)}")

    lengths = {name: len(data[name]) for name in data.keys()}
    if any(length != len(data["episode"]) for length in lengths.values()):
        mismatched = ", ".join(f"{k}={v}" for k, v in lengths.items())
        raise ValueError(f"数据数组长度不一致: {mismatched}")
    if data["action"].ndim != 2 or data["action"].shape[1] != 6:
        raise ValueError(f"action 应为 (N, 6)，实际为 {data['action'].shape}")
    if data["pos"].ndim != 2 or data["pos"].shape[1] != 6:
        raise ValueError(f"pos 应为 (N, 6)，实际为 {data['pos'].shape}")

    for episode in episodes:
        actions = np.asarray(data["action"][episode.start:episode.end], dtype=np.float64)
        positions = np.asarray(data["pos"][episode.start:episode.end], dtype=np.float64)
        if not np.isfinite(actions).all() or not np.isfinite(positions).all():
            raise ValueError(f"Episode {episode.label} 含 NaN/Inf，拒绝回放")
        if episode.length > 1:
            errors = positions[1:] - positions[:-1] - actions[1:]
            max_error = float(np.max(np.abs(errors)))
            if max_error > 2.0:
                print(
                    f"[WARN] Episode {episode.label} 的 pos/action 增量最大偏差 "
                    f"{max_error:.3f}，回放仍按 action 增量执行。"
                )


def _show_dataset(path: Path, root: zarr.Group, data: zarr.Group,
                  episodes: list[Episode]):
    print("\n========== Zarr 数据集 ==========")
    print(f"路径: {path.resolve()}")
    print(f"数组: {', '.join(data.keys())}")
    print(f"总帧数: {len(data['episode']):,}")
    if "episode_ends" in root.get("meta", {}):
        ends = np.asarray(root["meta"]["episode_ends"][:]).reshape(-1)
        print(f"meta/episode_ends: {ends.tolist()}")
    else:
        print("meta/episode_ends: 缺失（已从 data/episode 只读推算区间）")
    print("\nEpisode 列表（输入序号从 1 开始）:")
    for index, episode in enumerate(episodes, 1):
        print(
            f"  {index:>3}. 标签 {episode.label:<5} "
            f"帧 [{episode.start}, {episode.end})  {episode.length} steps"
        )
    print("=================================\n")


def _select_episode(episodes: list[Episode]) -> Episode | None:
    while True:
        try:
            value = input("输入回放序号（q 退出）: ").strip().lower()
        except EOFError:
            return None
        if value in ("q", "quit", "exit"):
            return None
        try:
            index = int(value)
        except ValueError:
            print("请输入列表中的数字序号，或 q 退出。")
            continue
        if 1 <= index <= len(episodes):
            return episodes[index - 1]
        print(f"序号范围为 1–{len(episodes)}。")


def _episode_start_bias(meta: zarr.Group | None, episode: Episode) -> np.ndarray:
    """Read the recorded effective start bias for one episode."""
    if meta is None or "episode_config" not in meta:
        raise ValueError("数据集缺少 meta/episode_config；请使用包含偏置元数据的采集数据")

    group_name = f"episode_{episode.label:06d}"
    config_group = meta["episode_config"]
    if group_name not in config_group:
        raise ValueError(f"Episode {episode.label} 缺少 meta/episode_config/{group_name}")

    attrs = config_group[group_name].attrs
    if "start_bias" not in attrs:
        raise ValueError(f"Episode {episode.label} 的配置缺少 start_bias")
    if int(attrs.get("episode_index", episode.label)) != episode.label:
        raise ValueError(f"Episode {episode.label} 的配置编号不匹配")

    bias = np.asarray(attrs["start_bias"], dtype=np.float64)
    if bias.shape != (6,) or not np.isfinite(bias).all():
        raise ValueError(
            f"Episode {episode.label} 的 start_bias 应为有限的 6 维向量，实际为 {bias}"
        )
    return bias


def _episode_gripper_range(meta: zarr.Group | None, episode: Episode) -> tuple[int, int]:
    """Read the raw closed/open positions recorded for one episode."""
    if meta is None or "episode_config" not in meta:
        raise ValueError("数据集缺少 meta/episode_config，无法读取夹爪范围")
    group_name = f"episode_{episode.label:06d}"
    config_group = meta["episode_config"]
    if group_name not in config_group or "gripper_range" not in config_group[group_name].attrs:
        raise ValueError(f"Episode {episode.label} 的配置缺少 gripper_range")
    bounds = np.asarray(config_group[group_name].attrs["gripper_range"], dtype=np.float64)
    if (bounds.shape != (2,) or not np.isfinite(bounds).all()
            or not 0 <= bounds[0] < bounds[1] <= 840
            or not np.equal(bounds, np.floor(bounds)).all()):
        raise ValueError(f"Episode {episode.label} 的 gripper_range 应为满足 0 <= min < max <= 840 的两个整数")
    return int(bounds[0]), int(bounds[1])


def _stop_arm(env: XArmEnv):
    """尽力停止正在执行的运动；失败时再尝试 SDK 状态停止。"""
    try:
        code = env.arm.emergency_stop()
        print(f"[INFO] emergency_stop 返回: {code}")
    except Exception as exc:
        print(f"[WARN] emergency_stop 不可用: {exc}")
        try:
            code = env.arm.set_state(4)
            print(f"[INFO] set_state(4) 返回: {code}")
        except Exception as stop_exc:
            print(f"[ERROR] 停止命令失败: {stop_exc}")


def _replay(env: XArmEnv, data: zarr.Group, episode: Episode,
            start_bias: np.ndarray,
            keyboard: KeyboardListener, hz: float, speed: float) -> bool:
    """按记录偏置定位开局，再回放一段数据；返回是否由 q 中断。

    ``start_bias`` 是采集时保存的相对 Home 偏置。这里先复位并直接应用该偏置，
    然后才从 episode 的第一条 action 开始回放；开局位置不从 pos/action 重新推算。
    """
    actions = np.asarray(data["action"][episode.start:episode.end], dtype=np.float64)
    grippers = np.asarray(
        data["gripper_action"][episode.start:episode.end], dtype=np.float64
    ).reshape(-1)

    start_pose = env.reset_pose + start_bias
    print(f"Episode {episode.label} 起始位姿: {np.round(start_pose, 3).tolist()}")
    print(f"夹爪原始范围（闭合、张开）: {list(env.gripper_range)}；回放指令使用 0/1。")
    print("将先复位到 home，再移动至该起始位姿。按 q 中断当前回放。")
    confirm = input("确认机械臂周围安全并开始？[y/N]: ").strip().lower()
    if confirm != "y":
        print("已取消。")
        return False

    env.reset(close_gripper=True)
    env.step(start_bias, gripper_action=0, speed=speed)

    period = 1.0 / hz
    previous_gripper = None
    interrupted = False
    print(f"开始回放，共 {episode.length} steps，目标频率 {hz:g} Hz。")
    # 单键模式仅覆盖动作回放；所有 input() 均在正常终端模式下执行。
    try:
        keyboard.start()
        for index, action in enumerate(actions):
            key = keyboard.get_key()
            if key == "q":
                interrupted = True
                _stop_arm(env)
                break
            if key == "\x03":
                raise KeyboardInterrupt

            # 仅向 Env 发送归一化二值指令，由 Env 映射到本轮原始范围。
            gripper_value = 0 if grippers[index] <= 0.5 else 1
            gripper_command = (
                gripper_value if previous_gripper is None or gripper_value != previous_gripper
                else None
            )
            frame_start = time.monotonic()
            env.step(action, gripper_action=gripper_command, speed=speed)
            previous_gripper = gripper_value

            if (index + 1) % 50 == 0 or index + 1 == episode.length:
                print(f"已执行 {index + 1}/{episode.length} steps", flush=True)
            remaining = period - (time.monotonic() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        keyboard.stop()
        print()

    if interrupted:
        print("回放已中断；当前 SDK step 若正在同步执行，会在返回后响应 q。")
    else:
        print(f"Episode {episode.label} 回放完成。")
    return interrupted


def main():
    parser = argparse.ArgumentParser(description="交互式回放 xArm Toolkit Zarr 数据")
    parser.add_argument("--input", "-i", required=True, help="Zarr 数据集路径")
    parser.add_argument("--addr", default="192.168.31.232", help="xArm 控制器 IP")
    parser.add_argument("--hz", type=float, default=10.0, help="回放目标频率（默认 10 Hz）")
    parser.add_argument("--speed", type=float, default=100.0, help="笛卡尔速度（默认 100）")
    args = parser.parse_args()

    if not math.isfinite(args.hz) or args.hz <= 0:
        parser.error("--hz 必须是正数")
    if not math.isfinite(args.speed) or not 0 < args.speed <= 2000:
        parser.error("--speed 必须在 (0, 2000] 范围内")
    path = Path(args.input).expanduser()
    if not path.exists():
        parser.error(f"数据集路径不存在: {path}")

    # mode='r' 保证脚本不会写入或修复用户数据集。
    root = zarr.open(str(path), mode="r")
    if "data" not in root:
        parser.error("Zarr 中缺少 data group")
    data = root["data"]
    meta = root["meta"] if "meta" in root else None
    episodes = _episode_ranges(data)
    _validate_dataset(data, episodes)
    _show_dataset(path, root, data, episodes)

    print("注意：回放会实际驱动机械臂。请确认急停可用、工作空间安全。")
    env = None
    keyboard = KeyboardListener()
    try:
        while True:
            episode = _select_episode(episodes)
            if episode is None:
                break
            try:
                start_bias = _episode_start_bias(meta, episode)
                gripper_range = _episode_gripper_range(meta, episode)
            except ValueError as exc:
                print(f"\n[ERROR] {exc}")
                continue
            try:
                if env is None:
                    env = XArmEnv(
                        addr=args.addr,
                        action_mode="delta_eef",
                        use_force=False,
                        initial_gripper_position=0,
                        gripper_range=gripper_range,
                    )
                else:
                    env.gripper_range = gripper_range
                interrupted = _replay(
                    env, data, episode, start_bias, keyboard, args.hz, args.speed
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"\n[ERROR] 回放失败: {exc}")
                _stop_arm(env) if env is not None else None
                interrupted = True

            if interrupted:
                choice = input("回放中断。[c] 复位并继续选择 episode / [q] 退出: ").strip().lower()
                if choice != "c":
                    break
            else:
                choice = input("[c] 复位后继续 / [q] 结束: ").strip().lower()
                if choice != "c":
                    break
                if env is not None:
                    env.reset(close_gripper=True)
    except KeyboardInterrupt:
        print("\n检测到 Ctrl+C，正在停止机械臂并清理环境……")
        if env is not None:
            _stop_arm(env)
    finally:
        keyboard.stop()
        if env is not None:
            env.cleanup()
        print("已退出回放程序。")


if __name__ == "__main__":
    main()

