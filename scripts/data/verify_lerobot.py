#!/usr/bin/env python3
"""验证转换后的 LeRobot 数据集 + 输出统计信息.

加载 LeRobotDataset 做端到端检查：帧数、episode、维度、图像 shape，
并输出 state/action 各维度统计（min/max/mean/std）、episode 长度分布、
异常帧检测（NaN、全零、极端值）。

用法::

    python scripts/data/verify_lerobot.py \
        --path lerobot_datasets/demo \
        --repo-id demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# ── state / action 各维度含义标签 ──────────────────────────
STATE_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
ACTION_LABELS = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
WRENCH_LABELS = ["fx", "fy", "fz", "mx", "my", "mz"]


def _print_separator(char: str = "─", width: int = 64):
    print(char * width)


def _print_array_stats(name: str, arr: np.ndarray, labels: list[str]):
    """打印 (N, D) 数组每个维度的统计."""
    print(f"\n[STATS] {name}  shape={arr.shape}  dtype={arr.dtype}")
    header = f"  {'维度':<10s} {'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s}"
    print(header)
    print(f"  {'─' * 50}")
    for d in range(arr.shape[1]):
        col = arr[:, d]
        label = labels[d] if d < len(labels) else f"dim{d}"
        print(
            f"  {label:<10s} {col.min():>10.4f} {col.max():>10.4f} "
            f"{col.mean():>10.4f} {col.std():>10.4f}"
        )


def _check_anomalies(name: str, arr: np.ndarray) -> list[str]:
    """检测 NaN、全零帧、极端值，返回警告列表."""
    warnings = []

    # NaN
    nan_count = int(np.isnan(arr).sum())
    if nan_count > 0:
        nan_frames = int(np.any(np.isnan(arr), axis=1).sum())
        warnings.append(f"{name}: 发现 {nan_count} 个 NaN 值 ({nan_frames} 帧)")

    # Inf
    inf_count = int(np.isinf(arr).sum())
    if inf_count > 0:
        warnings.append(f"{name}: 发现 {inf_count} 个 Inf 值")

    # 全零帧
    zero_frames = int(np.all(arr == 0, axis=1).sum())
    if zero_frames > 0:
        ratio = zero_frames / len(arr) * 100
        warnings.append(f"{name}: {zero_frames} 帧全零 ({ratio:.1f}%)")

    # 极端值 (超过 5σ)
    if not np.isnan(arr).any():
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        # 只检查 std > 0 的维度
        mask = std > 1e-8
        if mask.any():
            z_scores = np.abs((arr[:, mask] - mean[mask]) / std[mask])
            extreme_frames = int((z_scores > 5).any(axis=1).sum())
            if extreme_frames > 0:
                ratio = extreme_frames / len(arr) * 100
                warnings.append(
                    f"{name}: {extreme_frames} 帧有极端值 (>5σ, {ratio:.1f}%)"
                )

    return warnings


def verify(lerobot_path: str, repo_id: str):
    """验证 LeRobot 数据集 + 输出统计."""
    # 兼容 lerobot 新旧版本导入
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    resolved = Path(lerobot_path).resolve()
    if not resolved.exists():
        print(f"[FAIL] 路径不存在: {resolved}")
        sys.exit(1)

    # ── 加载 ──────────────────────────────────────────────
    print(f"[INFO] 加载数据集: {resolved}")
    dataset = LeRobotDataset(repo_id, root=resolved)

    n_frames = len(dataset)
    n_episodes = dataset.num_episodes
    n_tasks = len(dataset.meta.tasks)

    _print_separator("═")
    print("  数据集概览")
    _print_separator("═")
    print(f"  repo_id:    {dataset.repo_id}")
    print(f"  fps:        {dataset.fps}")
    print(f"  robot_type: {dataset.meta.robot_type}")
    print(f"  frames:     {n_frames}")
    print(f"  episodes:   {n_episodes}")
    print(f"  tasks:      {n_tasks}")
    if dataset.fps > 0:
        total_sec = n_frames / dataset.fps
        print(f"  总时长:     {total_sec:.1f}s ({total_sec / 60:.1f}min)")

    # tasks is a DataFrame indexed by task text. Iterating over .items() yields
    # columns (Series), which pandas formats as a multi-line table and misaligns
    # the overview. Print each task as a single row instead.
    for task, row in dataset.meta.tasks.iterrows():
        print(f"    [{int(row['task_index'])}] {task}")

    ok = True
    all_warnings: list[str] = []

    # ── 抽样第一帧，字段完整性检查 ────────────────────────
    _print_separator()
    print("[CHECK] 字段完整性 (frame[0])")
    sample = dataset[0]

    data_keys = [
        "observation.image",
        "observation.wrist_image",
        "observation.state",
        "action",
    ]
    # 传感器为可选字段，仅检查数据集中实际存在的项。
    if "observation.force" in dataset.features:
        data_keys.append("observation.force")
    if "observation.coinft_left" in dataset.features:
        data_keys.append("observation.coinft_left")
    if "observation.coinft_right" in dataset.features:
        data_keys.append("observation.coinft_right")
    meta_keys = {"timestamp", "frame_index", "episode_index", "index", "task_index", "task"}

    for key in data_keys:
        if key not in sample:
            print(f"  [FAIL] 缺少 {key}")
            ok = False
            continue
        value = sample[key]
        if hasattr(value, "shape"):
            shape_str = "×".join(str(d) for d in value.shape)
            print(f"  ✓ {key:30s} [{shape_str}] {value.dtype}")

    # state / action 维度检查
    if "observation.state" in sample:
        dim = sample["observation.state"].shape[-1]
        if dim != 7:
            print(f"  [WARN] observation.state dim={dim}, 期望 7")
            ok = False

    if "action" in sample:
        dim = sample["action"].shape[-1]
        if dim != 7:
            print(f"  [WARN] action dim={dim}, 期望 7")
            ok = False

    # 力传感器与 CoinFT 均应为六维力/力矩。
    if "observation.force" in sample:
        if sample["observation.force"].shape != (6,):
            print(f"  [WARN] observation.force shape={sample['observation.force'].shape}, 期望 (6,)")
            ok = False
    if "observation.coinft_left" in sample:
        if sample["observation.coinft_left"].shape != (6,):
            print(f"  [WARN] observation.coinft_left shape={sample['observation.coinft_left'].shape}, 期望 (6,)")
            ok = False
    if "observation.coinft_right" in sample:
        if sample["observation.coinft_right"].shape != (6,):
            print(f"  [WARN] observation.coinft_right shape={sample['observation.coinft_right'].shape}, 期望 (6,)")
            ok = False

    # task 描述
    if "task" in sample:
        print(f"  ✓ {'task':30s} = {sample['task']}")

    skipped = [k for k in sample if k in meta_keys and k != "task"]
    if skipped:
        print(f"  (元数据字段省略: {', '.join(skipped)})")

    # ── 遍历全量数据，收集统计 ────────────────────────────
    _print_separator()
    print(f"[INFO] 遍历全量数据 ({n_frames} frames)...")

    all_states = []
    all_actions = []
    all_forces = []
    all_coinft_left = []
    all_coinft_right = []
    episode_indices = []

    for i in range(n_frames):
        frame = dataset[i]
        if "observation.state" in frame:
            all_states.append(frame["observation.state"].numpy())
        if "action" in frame:
            all_actions.append(frame["action"].numpy())
        # 与 state/action 一样，收集所有帧的传感器读数。
        if "observation.force" in frame:
            all_forces.append(frame["observation.force"].numpy())
        if "observation.coinft_left" in frame:
            all_coinft_left.append(frame["observation.coinft_left"].numpy())
        if "observation.coinft_right" in frame:
            all_coinft_right.append(frame["observation.coinft_right"].numpy())
        if "episode_index" in frame:
            episode_indices.append(int(frame["episode_index"]))

        # 进度提示
        if (i + 1) % 2000 == 0 or i == n_frames - 1:
            print(f"  ... {i + 1}/{n_frames}", end="\r")

    print()

    # ── Episode 长度统计 ──────────────────────────────────
    if episode_indices:
        ep_arr = np.array(episode_indices)
        unique_eps = np.unique(ep_arr)

        ep_lengths = []
        for ep in unique_eps:
            ep_lengths.append(int(np.sum(ep_arr == ep)))
        ep_lengths = np.array(ep_lengths)

        _print_separator()
        print(f"[STATS] Episode 长度分布 (共 {len(ep_lengths)} 个)")
        print(f"  min:    {ep_lengths.min():>6d} frames", end="")
        if dataset.fps > 0:
            print(f"  ({ep_lengths.min() / dataset.fps:.1f}s)")
        else:
            print()
        print(f"  max:    {ep_lengths.max():>6d} frames", end="")
        if dataset.fps > 0:
            print(f"  ({ep_lengths.max() / dataset.fps:.1f}s)")
        else:
            print()
        print(f"  mean:   {ep_lengths.mean():>9.1f} frames", end="")
        if dataset.fps > 0:
            print(f"  ({ep_lengths.mean() / dataset.fps:.1f}s)")
        else:
            print()
        print(f"  std:    {ep_lengths.std():>9.1f} frames")
        print(f"  median: {np.median(ep_lengths):>9.1f} frames")

        # 逐 episode 列表
        print(f"\n  {'Episode':>8s}  {'Frames':>8s}", end="")
        if dataset.fps > 0:
            print(f"  {'Duration':>8s}")
        else:
            print()
        print(f"  {'─' * 30}")
        for i, length in enumerate(ep_lengths):
            line = f"  {i:>8d}  {length:>8d}"
            if dataset.fps > 0:
                line += f"  {length / dataset.fps:>7.1f}s"
            print(line)

        # 检测异常短 episode (< 10 frames)
        short_eps = np.where(ep_lengths < 10)[0]
        if len(short_eps) > 0:
            all_warnings.append(
                f"有 {len(short_eps)} 个异常短 episode (<10帧): {short_eps.tolist()}"
            )

    # ── State 统计 ────────────────────────────────────────
    if all_states:
        states = np.stack(all_states)  # (N, 7)
        _print_array_stats("observation.state", states, STATE_LABELS)
        all_warnings.extend(_check_anomalies("state", states))

    # ── Action 统计 ───────────────────────────────────────
    if all_actions:
        actions = np.stack(all_actions)  # (N, 7)
        _print_array_stats("action", actions, ACTION_LABELS)
        all_warnings.extend(_check_anomalies("action", actions))

        # action gripper 分布
        gripper_col = actions[:, -1]
        n_open = int(np.sum(gripper_col == 1))
        n_close = int(np.sum(gripper_col == 0))
        n_other = len(gripper_col) - n_open - n_close
        print(f"\n  gripper_action 分布: close(0)={n_close}, open(1)={n_open}", end="")
        if n_other > 0:
            print(f", other={n_other}")
        else:
            print()

    # ── 可选传感器统计：沿用 state/action 的统计和异常检测 ──
    if all_forces:
        forces = np.stack(all_forces)
        _print_array_stats("observation.force", forces, WRENCH_LABELS)
        all_warnings.extend(_check_anomalies("observation.force", forces))
    if all_coinft_left:
        coinft_left = np.stack(all_coinft_left)
        _print_array_stats("observation.coinft_left", coinft_left, WRENCH_LABELS)
        all_warnings.extend(_check_anomalies("observation.coinft_left", coinft_left))
    if all_coinft_right:
        coinft_right = np.stack(all_coinft_right)
        _print_array_stats("observation.coinft_right", coinft_right, WRENCH_LABELS)
        all_warnings.extend(_check_anomalies("observation.coinft_right", coinft_right))

    # ── 异常汇总 ──────────────────────────────────────────
    _print_separator()
    if all_warnings:
        print(f"[WARN] 发现 {len(all_warnings)} 个潜在问题:")
        for w in all_warnings:
            print(f"  ⚠ {w}")
        ok = False
    else:
        print("[INFO] 未发现异常")

    # ── 最终结果 ──────────────────────────────────────────
    _print_separator("═")
    print(f"  [{'PASS ✓' if ok else 'FAIL ✗'}] 验证完成")
    _print_separator("═")


def main():
    parser = argparse.ArgumentParser(description="验证 LeRobot 数据集 + 统计信息")
    parser.add_argument("--path", "-p", required=True, help="LeRobot 数据集路径")
    parser.add_argument("--repo-id", default=None, help="repo ID (默认使用目录名)")

    args = parser.parse_args()
    repo_id = args.repo_id or Path(args.path).name
    verify(args.path, repo_id)


if __name__ == "__main__":
    main()
