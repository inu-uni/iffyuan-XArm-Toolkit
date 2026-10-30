#!/usr/bin/env python3
"""Initialize XArm hardware and run the robot-side VLA deployment loop.

Usage::

    python scripts/deploy_vla.py \\
        --server-host 192.168.1.100 \\
        --instruction "pick up the orange toy and place it to the plate"
"""

from __future__ import annotations

import argparse

from xarm_toolkit.deploy.robot_deploy import XArmDeployer
from xarm_toolkit.env.realsense_env import RealsenseEnv
from xarm_toolkit.env.xarm_env import XArmEnv

CAM_ARM_SERIAL = "327122075644"   # D435i (arm-mounted / wrist)
CAM_FIX_SERIAL = "f1271506"       # L515 (fixed / base)


def main():
    parser = argparse.ArgumentParser(description="VLA robot-side deployment (Pi0.5 + XArm6)")
    parser.add_argument("--server-host", type=str, default="localhost",
                        help="VLA server hostname / IP")
    parser.add_argument("--server-port", type=int, default=10093,
                        help="VLA server port")
    parser.add_argument("--instruction", type=str, required=True,
                        help="Language instruction for the task")
    parser.add_argument("--ip", type=str, default="192.168.31.232",
                        help="XArm IP address")
    parser.add_argument("--action-mode", type=str, default="delta_eef",
                        choices=["absolute_joint", "absolute_eef", "delta_eef"],
                        help="Action mode (default: delta_eef)")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Maximum steps per episode")
    parser.add_argument("--hz", type=float, default=10.0,
                        help="Control frequency (Hz)")
    parser.add_argument("--use-force", action="store_true",
                        help="Enable force control mode")
    parser.add_argument("--coinft", action="store_true",
                        help="Read and send dual CoinFT wrench values to the inference server")
    parser.add_argument("--coinft.port", dest="coinft_port", type=str, default=None,
                        help="Teensy USB serial port for CoinFT (required with --coinft)")
    parser.add_argument("--gripper-range", type=int, nargs=2, default=[0, 840],
                        metavar=("MIN", "MAX"),
                        help="夹爪原始位置范围：闭合位置与张开位置（默认：0 840）")
    parser.add_argument("--debug", action="store_true",
                        help="Enable deployment debug output and recording")
    parser.add_argument("--debug.print-actions", dest="debug_print_actions", action="store_true",
                        help="Print the per-step server action log (requires --debug)")
    parser.add_argument("--debug.print-eef-pose", dest="debug_print_eef_pose", action="store_true",
                        help="Print the pre-action end-effector pose (requires --debug)")
    args = parser.parse_args()
    if args.coinft and not args.coinft_port:
        parser.error("--coinft requires --coinft.port")

    env = None
    cam_arm = None
    cam_fix = None
    coinft_env = None
    try:
        # Initialize hardware in the same order as the data collection entry point.
        if args.coinft:
            from xarm_toolkit.env.coinft_env import CoinFTEnv
            coinft_env = CoinFTEnv(port=args.coinft_port)

        env = XArmEnv(
            addr=args.ip,
            use_force=args.use_force,
            action_mode=args.action_mode,
            gripper_range=tuple(args.gripper_range),
        )
        cam_arm = RealsenseEnv(serial=CAM_ARM_SERIAL, mode="rgb")
        cam_fix = RealsenseEnv(serial=CAM_FIX_SERIAL, mode="rgb")

        deployer = XArmDeployer(
            env=env,
            cam_arm=cam_arm,
            cam_fix=cam_fix,
            coinft_env=coinft_env,
            server_host=args.server_host,
            server_port=args.server_port,
            instruction=args.instruction,
            action_mode=args.action_mode,
            max_steps=args.max_steps,
            hz=args.hz,
            debug=args.debug,
            debug_print_actions=args.debug_print_actions,
            debug_print_eef_pose=args.debug_print_eef_pose,
        )
        deployer.run()
    finally:
        # Also release any devices initialized before a later initialization failed.
        for device in (coinft_env, cam_fix, cam_arm, env):
            if device is not None:
                device.cleanup()


if __name__ == "__main__":
    main()

