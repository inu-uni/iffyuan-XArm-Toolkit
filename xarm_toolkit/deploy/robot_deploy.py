"""Robot-side VLA deployment loop for XArm6.

Wraps XArmEnv + dual RealSense cameras + VLAClient into a single class that
runs the deploy control loop.

Usage::

    from xarm_toolkit.deploy.robot_deploy import XArmDeployer

    deployer = XArmDeployer(
        server_host="192.168.1.100",
        instruction="pick up the orange toy and place it to the plate",
    )
    deployer.run()
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from xarm_toolkit.deploy.client import VLAClient
from xarm_toolkit.env.realsense_env import RealsenseEnv
from xarm_toolkit.env.xarm_env import XArmEnv, ActionMode
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("xarm_toolkit.deploy.robot_deploy")

# Hardware defaults
CAM_ARM_SERIAL = "327122075644"   # D435i (arm-mounted / wrist)
CAM_FIX_SERIAL = "f1271506"       # L515 (fixed / base)
GRIPPER_MAX = 1
GRIPPER_THRESHOLD = 0.5  # 与 Collector 一致的二值阈值


def build_state(obs: dict) -> np.ndarray:
    """Build 7-dim state: goal_pos(6) + gripper_binary(1).

    与 Collector 采集时完全一致：
    - pos = ``goal_pos``（commanded 目标位姿，非 cart_pos 反馈）
    - gripper = 二值 0/1（≤0.5 → 0 闭合, >0.5 → 1 张开）

    Parameters
    ----------
    obs : dict
        XArmEnv observation containing ``goal_pos`` (6,) and
        ``gripper_position`` (normalized 0-1).

    Returns
    -------
    np.ndarray
        Shape (7,), float32.
    """
    goal_pos = np.asarray(obs["goal_pos"], dtype=np.float32)
    gripper_position = float(obs.get("gripper_position", 0))
    gripper_binary = 0.0 if gripper_position <= GRIPPER_THRESHOLD else 1.0
    return np.concatenate([goal_pos, [gripper_binary]], dtype=np.float32)


def _ensure_hwc(rgb) -> np.ndarray:
    """Convert an Open3D image or array to HWC uint8."""
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] == 3:
        rgb = np.transpose(rgb, (1, 2, 0))
    return rgb.astype(np.uint8)


class XArmDeployer:
    """End-to-end deployment: VLAClient + XArmEnv + cameras.

    Parameters
    ----------
    server_host : str
        VLA inference server hostname / IP.
    server_port : int
        VLA inference server port.
    instruction : str
        Language instruction for the task.
    arm_ip : str
        XArm controller IP.
    action_mode : ActionMode
        How to interpret the model's action output.
    max_steps : int
        Maximum steps per episode.
    hz : float
        Control frequency.
    use_force : bool
        Enable force control mode on XArmEnv.
    gripper_range : tuple[int, int]
        Raw closed/open positions passed to XArmEnv for normalized control.
    cam_arm_serial : str
        Arm-mounted camera serial.
    cam_fix_serial : str
        Fixed camera serial.
    """

    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 10093,
        instruction: str = "",
        arm_ip: str = "192.168.31.232",
        action_mode: ActionMode = "delta_eef",
        max_steps: int = 800,
        hz: float = 10.0,
        use_force: bool = False,
        debug: bool = False,
        debug_print_actions: bool = False,
        debug_print_eef_pose: bool = False,
        debug_file: str | None = None,
        cam_arm_serial: str = CAM_ARM_SERIAL,
        cam_fix_serial: str = CAM_FIX_SERIAL,
        gripper_range: tuple[int, int] = (0, 840),
    ):
        self.instruction = instruction
        self.max_steps = max_steps
        self.dt = 1.0 / hz
        self.hz = hz
        self.action_mode = action_mode
        self.debug = debug
        self.debug_print_actions = debug_print_actions
        self.debug_print_eef_pose = debug_print_eef_pose
        self.debug_file = debug_file

        # ---- Hardware ----
        logger.info("Initializing XArm environment ...")
        self.env = XArmEnv(
            addr=arm_ip,
            action_mode=action_mode,
            use_force=use_force,
            gripper_range=gripper_range,
        )

        logger.info("Initializing cameras ...")
        self.cam_arm = RealsenseEnv(serial=cam_arm_serial, mode="rgb")
        self.cam_fix = RealsenseEnv(serial=cam_fix_serial, mode="rgb")

        # ---- VLA client ----
        logger.info("Connecting to VLA server at %s:%d ...", server_host, server_port)
        self.client = VLAClient(host=server_host, port=server_port)

        if not self.client.ping():
            raise ConnectionError("VLA server ping failed!")
        logger.info("Server connected. Metadata: %s", self.client.metadata)

    def run(self):
        """Run the deployment control loop (blocking)."""
        # Reset
        obs = self.env.reset(close_gripper=False)
        logger.info("Robot reset. Starting deployment loop ...")
        logger.info("Instruction: %s", self.instruction)
        logger.info(
            "Action mode: %s, Hz: %.1f, Max steps: %d",
            self.action_mode, self.hz, self.max_steps,
        )

        debug_file = None
        try:
            if self.debug and self.debug_file:
                debug_path = Path(self.debug_file).expanduser()
                debug_path.parent.mkdir(parents=True, exist_ok=True)
                debug_file = debug_path.open("a", encoding="utf-8", buffering=1)

            for step_i in range(self.max_steps):
                t0 = time.monotonic()

                # ---- Capture images (HWC uint8) ----
                rgb_arm = _ensure_hwc(self.cam_arm.step()["rgb"])   # wrist
                rgb_fix = _ensure_hwc(self.cam_fix.step()["rgb"])   # base

                # ---- Build state: cart_pos(6) + gripper(1) ----
                state = build_state(obs)

                # ---- Inference ----
                result = self.client.predict(
                    images=[rgb_arm, rgb_fix],
                    instruction=self.instruction,
                    state=state,
                )

                actions = result.get("actions")
                if actions is None:
                    logger.warning("No actions returned at step %d", step_i)
                    continue

                # ---- Execute action ----
                # actions shape: (chunk_size, 7) — 取第一个 action
                action = actions[0] if actions.ndim > 1 else actions

                debug_record = {"step": step_i} if self.debug else None
                if self.debug and self.debug_print_eef_pose:
                    eef_pose = np.asarray(obs["cart_pos"], dtype=float).copy()
                    logger.info(
                        "Step %d/%d eef_pose=%s",
                        step_i, self.max_steps,
                        np.array2string(eef_pose, precision=4, separator=",", max_line_width=1000),
                    )
                    debug_record["eef_pose"] = eef_pose.tolist()

                # 前 6 维: delta EEF [dx, dy, dz, dr, dp, dy]
                arm_action = action[:6]
                # 第 7 维：归一化夹爪动作，>0.5 → 1 张开，≤0.5 → 0 闭合。
                gripper_input = None
                gripper_action = None
                if len(action) > 6:
                    gripper_input = float(action[6])
                    gripper_action = GRIPPER_MAX if gripper_input > GRIPPER_THRESHOLD else 0

                if self.debug and self.debug_print_actions:
                    logger.info(
                        "Step %d/%d action_mode=%s arm_action=%s gripper_input=%s gripper_command=%s",
                        step_i, self.max_steps, self.action_mode,
                        np.array2string(arm_action, precision=4, separator=",", max_line_width=1000),
                        gripper_input, gripper_action,
                    )
                    debug_record["action"] = np.asarray(action, dtype=float).tolist()

                if debug_file is not None and debug_record is not None and len(debug_record) > 1:
                    debug_file.write(json.dumps(debug_record, ensure_ascii=False) + "\n")

                try:
                    obs = self.env.step(
                        action=arm_action,
                        gripper_action=gripper_action,
                    )
                except Exception as exc:
                    if debug_file is not None:
                        try:
                            debug_file.write(json.dumps({
                                "step": step_i,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }, ensure_ascii=False) + "\n")
                        except Exception as log_exc:
                            logger.error("Failed to record control error: %s", log_exc)
                    raise

                # ---- Rate control ----
                elapsed = time.monotonic() - t0
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)

                if step_i % 10 == 0:
                    logger.info(
                        "Step %d/%d  elapsed=%.3fs  pos=%s  gripper=%.2f",
                        step_i, self.max_steps, elapsed,
                        np.array2string(obs["cart_pos"][:3], precision=1),
                        state[-1],
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            if debug_file is not None:
                debug_file.close()
            self.cleanup()

    def cleanup(self):
        """Release all resources."""
        self.client.close()
        self.env.cleanup()
        self.cam_arm.cleanup()
        self.cam_fix.cleanup()
        logger.info("Deployer cleaned up.")

