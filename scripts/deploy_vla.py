#!/usr/bin/env python3
"""Robot-side VLA deployment script for Pi0.5 + XArm6.

Thin entry point — all logic lives in :mod:`xarm_toolkit.deploy.robot_deploy`.

Usage::

    python scripts/deploy_vla.py \\
        --server-host 192.168.1.100 \\
        --instruction "pick up the orange toy and place it to the plate"
"""

from __future__ import annotations

import argparse

from xarm_toolkit.deploy.robot_deploy import XArmDeployer


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
    parser.add_argument("--gripper-range", type=int, nargs=2, default=[0, 840],
                        metavar=("MIN", "MAX"),
                        help="夹爪原始位置范围：闭合位置与张开位置（默认：0 840）")
    parser.add_argument("--debug", action="store_true",
                        help="Enable deployment debug output and recording")
    parser.add_argument("--debug.print-actions", dest="debug_print_actions", action="store_true",
                        help="Print the per-step server action log (requires --debug)")
    parser.add_argument("--debug.print-eef-pose", dest="debug_print_eef_pose", action="store_true",
                        help="Print the pre-action end-effector pose (requires --debug)")
    parser.add_argument("--debug.file", dest="debug_file", type=str, default=None,
                        help="Append selected per-step debug data to this JSONL file (requires --debug)")
    args = parser.parse_args()

    deployer = XArmDeployer(
        server_host=args.server_host,
        server_port=args.server_port,
        instruction=args.instruction,
        arm_ip=args.ip,
        action_mode=args.action_mode,
        max_steps=args.max_steps,
        hz=args.hz,
        use_force=args.use_force,
        gripper_range=tuple(args.gripper_range),
        debug=args.debug,
        debug_print_actions=args.debug_print_actions,
        debug_print_eef_pose=args.debug_print_eef_pose,
        debug_file=args.debug_file,
    )
    deployer.run()


if __name__ == "__main__":
    main()

