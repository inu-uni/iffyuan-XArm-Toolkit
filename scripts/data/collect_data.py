#!/usr/bin/env python3
"""XArm6 数据采集入口脚本。

Usage
-----
# 默认采集 rgbd，目标 3 个 episode（新建数据集）
python scripts/data/collect_data.py --dataset datasets/demo.zarr

# 指定目标 30 个 episode（断点续采：已有 N 个则只采 30-N 个）
python scripts/data/collect_data.py --dataset datasets/plug.zarr --episodes 30

# 指定任务初始偏移 + 力控
python scripts/data/collect_data.py --start-bias 0 0 -200 --force --dataset datasets/plug.zarr

# 带随机偏移 + 夹爪始终闭合（stamp 类任务）
python scripts/data/collect_data.py --start-bias 0 0 -270 --gripper-closed --dataset datasets/stamp.zarr

# 采集同时保存回放视频
python scripts/data/collect_data.py --save-video --dataset datasets/demo.zarr

# 启用 CoinFT，只保存 coinft_left / coinft_right
python -m scripts.data.collect_data --coinft --coinft.port COM3 --dataset datasets/coinft.zarr

断点续采
--------
``--episodes`` 含义是 **目标总数** 而非"再采 N 个"。
例如 ``--episodes 30`` 表示这个数据集总共需要 30 个 episode：
  - 数据集不存在 → 新建，采 30 个
  - 已有 12 个    → 自动算出还差 18 个，只采 18 个
  - 已有 ≥ 30 个  → 打印提示直接退出
同一个命令反复跑就行，中途 Ctrl+C 也不会丢失已保存的数据。

键盘控制
--------
  Space   — 开始录制当前 episode
  Enter   — 结束当前 episode
  Ctrl+C  — 优雅退出（已保存的 episode 完好）

SpaceMouse
----------
  6D 移动    — 控制机械臂末端
  左键(btn0) — 张开夹爪
  右键(btn1) — 关闭夹爪
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import zarr

from xarm_toolkit.env.xarm_env import XArmEnv
from xarm_toolkit.env.realsense_env import RealsenseEnv
from xarm_toolkit.teleop.spacemouse import SpacemouseAgent, SpacemouseConfig
from xarm_toolkit.collect.collector import Collector
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("collect_data")

# 硬件常量
XARM_IP = "192.168.31.232"
CAM_ARM_SERIAL = "327122075644"  # D435i
CAM_FIX_SERIAL = "f1271506"      # L515


def parse_args():
    p = argparse.ArgumentParser(description="XArm6 data collection")

    # 数据集
    p.add_argument("--dataset", type=str, default="datasets/demo.zarr",
                   help="Zarr dataset path")
    p.add_argument("--episodes", type=int, default=3,
                   help="目标 episode 总数（断点续采：已有 N 个则只采 episodes-N 个）")

    # 任务参数
    p.add_argument("--start-bias", type=float, nargs=3, default=[0, 0, 0],
                   metavar=("X", "Y", "Z"),
                   help="Initial position offset from Home [x, y, z] in mm (default: 0 0 0)")
    p.add_argument("--random-bias-x", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Random X offset range per episode, e.g. --random-bias-x -50 70")
    p.add_argument("--random-bias-y", type=float, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="Random Y offset range per episode, e.g. --random-bias-y -20 10")
    p.add_argument("--gripper-closed", action="store_true",
                   help="Gripper always closed (stamp-like tasks)")

    # 硬件
    p.add_argument("--ip", type=str, default=XARM_IP,
                   help="XArm IP address")
    p.add_argument("--force", action="store_true",
                   help="Enable force sensor (default: off)")

    # CoinFT 开关与 USB 串口，默认关闭。
    p.add_argument("--coinft", action="store_true", help="Enable CoinFT recording")
    p.add_argument("--coinft.port", dest="coinft_port", type=str, default=None,
                   help="Teensy USB serial port")

    # 相机
    p.add_argument("--cam-mode", type=str, default="rgbd",
                   choices=["rgb", "rgbd", "pcd"],
                   help="Camera mode: rgb, rgbd (default), pcd")
    p.add_argument("--image-size", type=int, nargs=2, default=[320, 240],
                   metavar=("W", "H"),
                   help="Image size for saved data (default: 320 240)")

    # 视频
    p.add_argument("--save-video", action="store_true",
                   help="Save per-episode MP4 videos for review")
    p.add_argument("--video-fps", type=float, default=15.0,
                   help="FPS for saved videos (default 15)")

    # SpaceMouse
    p.add_argument("--trans-scale", type=float, default=5.0,
                   help="SpaceMouse translation sensitivity")
    p.add_argument("--rot-scale", type=float, default=0.004,
                   help="SpaceMouse rotation sensitivity")

    args = p.parse_args()
    if args.coinft and not args.coinft_port:
        p.error("--coinft 需要同时指定 --coinft.port")
    return args


def main():
    args = parse_args()

    # --- 断点续采：计算剩余 episode 数 ---
    existing_episodes = 0
    if os.path.exists(args.dataset):
        try:
            ds = zarr.open(args.dataset, mode="r")
            if "data" in ds and "episode" in ds["data"]:
                existing_episodes = len(np.unique(ds["data"]["episode"][:]))
        except Exception:
            pass  # 无法读取，当作新建

    remaining = args.episodes - existing_episodes

    if remaining <= 0:
        print(
            f"\n✓ 数据集已有 {existing_episodes} episodes，"
            f"已达到目标 {args.episodes}，无需继续采集。\n"
            f"  路径: {args.dataset}\n"
            f"  如需采集更多，请增大 --episodes 的值。\n"
        )
        return

    if existing_episodes > 0:
        logger.info(
            "断点续采: 已有 %d episodes, 目标 %d, 本次采集 %d",
            existing_episodes, args.episodes, remaining,
        )

    # --- Build task config from CLI args ---
    task_cfg = {
        "start_bias": args.start_bias,
        "gripper_always_closed": args.gripper_closed,
    }
    if args.random_bias_x is not None:
        task_cfg.setdefault("random_bias", {})["x"] = args.random_bias_x
    if args.random_bias_y is not None:
        task_cfg.setdefault("random_bias", {})["y"] = args.random_bias_y

    # CoinFT 按需连接；Collector.run() 启动时仅初始化一次。
    coinft_env = None
    if args.coinft:
        from xarm_toolkit.env.coinft_env import CoinFTEnv
        coinft_env = CoinFTEnv(port=args.coinft_port)

    # --- Init env ---
    env = XArmEnv(
        addr=args.ip,
        use_force=args.force,
        action_mode="delta_eef",
        initial_gripper_position=840,
    )

    # --- Init cameras ---
    cam_mode = args.cam_mode
    cam_arm = RealsenseEnv(serial=CAM_ARM_SERIAL, mode=cam_mode)
    cam_fix = RealsenseEnv(serial=CAM_FIX_SERIAL, mode=cam_mode)

    # --- Init SpaceMouse ---
    sm_cfg = SpacemouseConfig(
        translation_scale=args.trans_scale,
        rotation_scale=args.rot_scale,
    )
    agent = SpacemouseAgent(config=sm_cfg)

    # --- Run collector ---
    collector = Collector(
        env=env,
        cam_arm=cam_arm,
        cam_fix=cam_fix,
        agent=agent,
        dataset_path=args.dataset,
        task_config=task_cfg,
        num_episodes=remaining,
        cam_mode=cam_mode,
        image_size=tuple(args.image_size),
        save_video=args.save_video,
        video_fps=args.video_fps,
        use_coinft=args.coinft,
        coinft_env=coinft_env,
    )

    collector.run()


if __name__ == "__main__":
    main()
