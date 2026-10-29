#!/usr/bin/env python3
"""单机串行部署：模型加载 + 环境初始化 + 推理循环，不走 Server/Client。

把 GPU 推理和机械臂控制放在同一台机器上，适用于：
- GPU 机器直接连着 XArm 和相机
- 调试时不想折腾两台机器

Usage::

    python scripts/deploy_local.py \
        --config-file /path/to/pi05_xarm_v2_stage1.py \
        --ckpt-dir /path/to/checkpoint \
        --instruction "pick up the orange toy" \
        --ip 192.168.31.232 \
        --hz 10 \
        --max-steps 800
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from xarm_toolkit.deploy.openpi_server import Pi05XArmPolicy
from xarm_toolkit.env.realsense_env import RealsenseEnv
from xarm_toolkit.env.xarm_env import XArmEnv
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("deploy_local")

# ── 硬件常量 ──────────────────────────────────────────────
CAM_ARM_SERIAL = "327122075644"   # D435i (wrist)
CAM_FIX_SERIAL = "f1271506"       # L515  (base)
GRIPPER_MAX = 1
GRIPPER_THRESHOLD = 0.5           # ≤0.5 → 0 闭合, >0.5 → 1 张开


# ── 工具函数 ──────────────────────────────────────────────

def build_state(obs: dict) -> np.ndarray:
    """构建 7 维状态: goal_pos(6) + gripper_binary(1)."""
    goal_pos = np.asarray(obs["goal_pos"], dtype=np.float32)
    gripper_raw = float(obs.get("gripper_position", 0))
    gripper_binary = 0.0 if gripper_raw <= GRIPPER_THRESHOLD else 1.0
    return np.concatenate([goal_pos, [gripper_binary]], dtype=np.float32)


def ensure_hwc(rgb) -> np.ndarray:
    """确保图像为 HWC uint8（兼容 Open3D Image 和 numpy array）."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb.astype(np.uint8)


# ── 主流程 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="单机串行 VLA 部署 (模型 + 环境)")

    # 模型相关
    parser.add_argument("--config-file", type=str, required=True,
                        help="openpi config 文件路径 (须定义 build_config())")
    parser.add_argument("--ckpt-dir", type=str, required=True,
                        help="checkpoint 目录")

    # 任务指令
    parser.add_argument("--instruction", type=str,
                        default="pick up the orange toy and place it to the plate",
                        help="语言指令")

    # 机械臂
    parser.add_argument("--ip", type=str, default="192.168.31.232",
                        help="XArm IP")
    parser.add_argument("--action-mode", type=str, default="delta_eef",
                        choices=["delta_eef", "absolute_eef", "absolute_joint"],
                        help="动作模式")
    parser.add_argument("--use-force", action="store_true",
                        help="启用力控")

    # 控制参数
    parser.add_argument("--max-steps", type=int, default=800,
                        help="最大步数")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="控制频率")

    args = parser.parse_args()
    dt = 1.0 / args.hz

    # ── 1) 加载模型 ──────────────────────────────────────
    logger.info("加载 Pi0.5 模型 ...")
    logger.info("  config: %s", args.config_file)
    logger.info("  ckpt:   %s", args.ckpt_dir)
    policy = Pi05XArmPolicy(
        config_file=args.config_file,
        ckpt_dir=args.ckpt_dir,
    )
    logger.info("模型加载完成 ✓")

    # ── 2) 初始化环境 ────────────────────────────────────
    logger.info("初始化 XArm 环境 ...")
    env = XArmEnv(
        addr=args.ip,
        action_mode=args.action_mode,
        use_force=args.use_force,
    )

    logger.info("初始化相机 ...")
    cam_arm = RealsenseEnv(serial=CAM_ARM_SERIAL, mode="rgb")
    cam_fix = RealsenseEnv(serial=CAM_FIX_SERIAL, mode="rgb")
    logger.info("硬件初始化完成 ✓")

    # ── 3) 推理循环 ──────────────────────────────────────
    obs = env.reset(close_gripper=False)
    logger.info("机械臂已复位，开始部署循环")
    logger.info("  指令: %s", args.instruction)
    logger.info("  模式: %s | 频率: %.1f Hz | 最大步数: %d",
                args.action_mode, args.hz, args.max_steps)

    try:
        for step_i in range(args.max_steps):
            t0 = time.monotonic()

            # 采图
            rgb_arm = ensure_hwc(cam_arm.step()["rgb"])   # wrist
            rgb_fix = ensure_hwc(cam_fix.step()["rgb"])   # base

            # 构建状态
            state = build_state(obs)

            # 本地推理（直接调 policy，不走网络）
            result = policy.predict_action(
                images=[rgb_arm, rgb_fix],
                instruction=args.instruction,
                state=state,
            )
            print(result)

            actions = result.get("actions")
            if actions is None:
                logger.warning("Step %d: 模型未返回 actions，跳过", step_i)
                continue

            # 取第一个 action
            action = actions[0] if actions.ndim > 1 else actions

            # 前 6 维: delta EEF;  第 7 维: 夹爪
            arm_action = action[:6]
            gripper_action = None
            if len(action) > 6:
                gripper_action = GRIPPER_MAX if float(action[6]) >= 0.5 else 0

            # 执行
            obs = env.step(action=arm_action, gripper_action=gripper_action)

            # 频率控制
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

            # 日志
            if step_i % 10 == 0:
                logger.info(
                    "Step %d/%d  %.3fs  pos=%s  gripper=%.0f",
                    step_i, args.max_steps, elapsed,
                    np.array2string(obs["cart_pos"][:3], precision=1),
                    state[-1],
                )

    except KeyboardInterrupt:
        logger.info("用户中断")
    finally:
        logger.info("清理资源 ...")
        env.cleanup()
        cam_arm.cleanup()
        cam_fix.cleanup()
        logger.info("完成 ✓")


if __name__ == "__main__":
    main()
