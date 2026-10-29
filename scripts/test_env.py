#!/usr/bin/env python3
"""验证 XArm6 机械臂环境是否正常工作.

逐步测试: 连接 → 读状态 → 复位 → 小幅移动 → 夹爪 → 清理.
每一步都会暂停等确认，确保安全。

Usage:
    python scripts/test_env.py                      # 默认 IP
    python scripts/test_env.py --ip 192.168.31.232   # 指定 IP
    python scripts/test_env.py --force               # 测试力传感器
    python scripts/test_env.py --skip-move           # 跳过移动测试
"""

from __future__ import annotations

import argparse
import traceback

import numpy as np

from xarm_toolkit.env.xarm_env import XArmEnv
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("test_env")


def wait_confirm(prompt: str = "按 Enter 继续，输入 q 退出: "):
    resp = input(prompt).strip().lower()
    if resp == "q":
        raise KeyboardInterrupt("User quit")


def main():
    parser = argparse.ArgumentParser(description="XArm6 environment verification")
    parser.add_argument("--ip", default="192.168.31.232", help="XArm IP")
    parser.add_argument("--force", action="store_true", help="Enable FT sensor")
    parser.add_argument("--skip-move", action="store_true", help="Skip movement test")
    args = parser.parse_args()

    env = None

    try:
        # ---- Step 1: Connect ----
        print("\n" + "=" * 60)
        print("  步骤 1/5: 连接 XArm6")
        print("=" * 60)
        logger.info("Connecting to XArm6 at %s (force=%s) ...", args.ip, args.force)

        env = XArmEnv(
            addr=args.ip,
            use_force=args.force,
            action_mode="delta_eef",
            initial_gripper_position=1,
        )
        logger.info("✓ 连接成功")

        # ---- Step 2: Read state ----
        print("\n" + "=" * 60)
        print("  步骤 2/5: 读取当前状态")
        print("=" * 60)

        obs = env._get_observation()
        print(f"  笛卡尔位姿 (cart_pos):   {np.array2string(obs['cart_pos'], precision=2)}")
        print(f"  关节角 (servo_angle):     {np.array2string(obs['servo_angle'], precision=3)}")
        print(f"  夹爪位置 (gripper):       {obs['gripper_position']:.1f}")
        print(f"  目标位姿 (goal_pos):      {np.array2string(obs['goal_pos'], precision=2)}")
        if obs.get("ext_force") is not None:
            print(f"  外力 (ext_force):         {np.array2string(obs['ext_force'], precision=2)}")
            print(f"  原始力 (raw_force):       {np.array2string(obs['raw_force'], precision=2)}")
        else:
            print("  外力: disabled (use --force to enable)")

        logger.info("✓ 状态读取正常")

        # ---- Step 3: Reset ----
        print("\n" + "=" * 60)
        print("  步骤 3/5: 复位到 Home")
        print("=" * 60)
        print(f"  Home 位姿: {env.RESET_POSE}")
        wait_confirm("  确认复位? Enter 继续 / q 退出: ")

        obs = env.reset(close_gripper=False)
        print(f"  复位后位姿: {np.array2string(obs['cart_pos'], precision=2)}")
        logger.info("✓ 复位成功")

        if args.force:
            env.reset_force_sensor_zero()
            logger.info("✓ 力传感器已调零")

        # ---- Step 4: Small movement ----
        if not args.skip_move:
            print("\n" + "=" * 60)
            print("  步骤 4/5: 小幅移动测试")
            print("=" * 60)

            # Test X+10mm
            delta = np.array([10, 0, 0, 0, 0, 0], dtype=np.float64)
            print(f"  即将执行 delta = {delta} (X+10mm)")
            wait_confirm("  确认移动? Enter 继续 / q 退出: ")

            obs = env.step(action=delta, speed=200)
            print(f"  移动后位姿: {np.array2string(obs['cart_pos'], precision=2)}")

            # Move back
            delta_back = np.array([-10, 0, 0, 0, 0, 0], dtype=np.float64)
            obs = env.step(action=delta_back, speed=200)
            print(f"  回到原位:   {np.array2string(obs['cart_pos'], precision=2)}")
            logger.info("✓ 移动测试通过")
        else:
            print("\n  [跳过移动测试]")

        # ---- Step 5: Gripper ----
        print("\n" + "=" * 60)
        print("  步骤 5/5: 夹爪测试")
        print("=" * 60)
        wait_confirm("  即将关闭夹爪, Enter 继续 / q 退出: ")

        obs = env.step(action=[0, 0, 0, 0, 0, 0], gripper_action=0)
        print(f"  夹爪关闭后位置: {obs['gripper_position']:.1f}")

        wait_confirm("  即将打开夹爪, Enter 继续 / q 退出: ")

        obs = env.step(action=[0, 0, 0, 0, 0, 0], gripper_action=1)
        print(f"  夹爪打开后位置: {obs['gripper_position']:.1f}")
        logger.info("✓ 夹爪测试通过")

        # ---- Summary ----
        print("\n" + "=" * 60)
        print("  ✅ 所有测试通过!")
        print("=" * 60 + "\n")

    except KeyboardInterrupt:
        print("\n  用户退出")
    except Exception:
        logger.error("Test failed:\n%s", traceback.format_exc())
    finally:
        if env is not None:
            logger.info("Cleanup ...")
            env.cleanup()
            logger.info("Done.")


if __name__ == "__main__":
    main()
