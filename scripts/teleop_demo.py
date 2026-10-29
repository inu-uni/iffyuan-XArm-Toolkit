#!/usr/bin/env python3
"""SpaceMouse teleoperation demo for XArm6.

Usage:
    python scripts/teleop_demo.py                   # 默认: 无力控
    python scripts/teleop_demo.py --force            # 启用力传感器
    python scripts/teleop_demo.py --speed 500        # 自定义速度
    python scripts/teleop_demo.py --trans-scale 3    # 调整灵敏度

键盘控制:
    q / Ctrl+C  — 退出
    r           — 复位到初始位姿
    o           — 打开夹爪
    c           — 关闭夹爪

SpaceMouse:
    6D 移动    — 控制机械臂末端
    左键(btn0) — 切换夹爪开/关
"""

from __future__ import annotations

import argparse
import sys
import time

from xarm_toolkit.env.xarm_env import XArmEnv
from xarm_toolkit.teleop.spacemouse import SpacemouseAgent, SpacemouseConfig
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("teleop_demo")


def parse_args():
    p = argparse.ArgumentParser(description="SpaceMouse teleop for XArm6")
    p.add_argument("--ip", default="192.168.31.232", help="XArm IP")
    p.add_argument("--force", action="store_true", help="Enable FT sensor")
    p.add_argument("--speed", type=float, default=400, help="Cartesian speed mm/s (default 400)")
    p.add_argument("--hz", type=float, default=50, help="Control loop frequency")
    p.add_argument("--trans-scale", type=float, default=5.0, help="Translation sensitivity")
    p.add_argument("--rot-scale", type=float, default=0.004, help="Rotation sensitivity")
    p.add_argument("--deadzone", type=float, default=0.05, help="SpaceMouse deadzone")
    return p.parse_args()


def main():
    args = parse_args()

    # --- Init env ---
    logger.info("Connecting to XArm6 at %s ...", args.ip)
    env = XArmEnv(
        addr=args.ip,
        use_force=args.force,
        action_mode="delta_eef",
        initial_gripper_position=1,  # start open
    )

    # --- Init SpaceMouse ---
    sm_cfg = SpacemouseConfig(
        translation_scale=args.trans_scale,
        rotation_scale=args.rot_scale,
        deadzone=args.deadzone,
    )
    agent = SpacemouseAgent(config=sm_cfg)

    # --- Reset ---
    logger.info("Resetting arm ...")
    obs = env.reset(close_gripper=False)
    logger.info("Ready! Cart pos: %s", obs["cart_pos"])
    if args.force:
        env.reset_force_sensor_zero()
        logger.info("Force sensor zeroed.")

    print("\n" + "=" * 50)
    print("  SpaceMouse Teleop — XArm6")
    print("  q: quit  |  r: reset  |  o: open  |  c: close")
    print("  Left button: toggle gripper")
    print("=" * 50 + "\n")

    # --- Keyboard listener (non-blocking) ---
    import termios
    import tty
    import select

    old_settings = termios.tcgetattr(sys.stdin)

    def get_key():
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1).lower()
        return None

    dt = 1.0 / args.hz
    step_count = 0

    try:
        tty.setraw(sys.stdin.fileno())

        while True:
            t0 = time.time()

            # Check keyboard
            key = get_key()
            if key == "q" or key == "\x03":  # q or Ctrl+C
                print("\r\nExiting ...")
                break
            elif key == "r":
                print("\r\nResetting ...")
                obs = env.reset(close_gripper=False)
                if args.force:
                    env.reset_force_sensor_zero()
                print("\r\nReset done.")
                continue
            elif key == "o":
                env.step(action=[0, 0, 0, 0, 0, 0], gripper_action=1)
                print("\r\nGripper opened.")
                continue
            elif key == "c":
                env.step(action=[0, 0, 0, 0, 0, 0], gripper_action=0)
                print("\r\nGripper closed.")
                continue

            # Read SpaceMouse & step
            action, gripper = agent.act(obs)
            obs = env.step(action, gripper_action=gripper, speed=args.speed)

            step_count += 1
            if step_count % 50 == 0:
                pos = obs["cart_pos"]
                grip = obs["gripper_position"]
                force_str = ""
                if obs.get("ext_force") is not None:
                    f = obs["ext_force"]
                    force_str = f" | force=[{f[0]:.1f},{f[1]:.1f},{f[2]:.1f}]"
                # \r\n for raw terminal mode
                print(
                    f"\r\nStep {step_count:5d} | "
                    f"pos=[{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f}] | "
                    f"gripper={grip:.0f}{force_str}"
                )

            # Rate limiting
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\r\nInterrupted.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("Cleanup done.")


if __name__ == "__main__":
    main()
