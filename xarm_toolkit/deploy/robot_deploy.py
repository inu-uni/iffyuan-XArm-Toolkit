"""Robot-side VLA deployment loop for XArm6.

Wraps XArmEnv + dual RealSense cameras + VLAClient into a single class that
runs the deploy control loop.

Usage::

    from xarm_toolkit.deploy.robot_deploy import XArmDeployer
    from xarm_toolkit.env.realsense_env import RealsenseEnv
    from xarm_toolkit.env.xarm_env import XArmEnv

    env = XArmEnv(addr="192.168.31.232", action_mode="delta_eef")
    cam_arm = RealsenseEnv(serial="327122075644", mode="rgb")
    cam_fix = RealsenseEnv(serial="f1271506", mode="rgb")
    deployer = XArmDeployer(env, cam_arm, cam_fix, instruction="pick up the cup")
    deployer.run()
"""

from __future__ import annotations

import time

import numpy as np

from xarm_toolkit.deploy.client import VLAClient
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("xarm_toolkit.deploy.robot_deploy")

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
    """Run the deployment loop using initialized hardware environments.

    Parameters
    ----------
    server_host : str
        VLA inference server hostname / IP.
    server_port : int
        VLA inference server port.
    instruction : str
        Language instruction for the task.
    env, cam_arm, cam_fix : object
        Initialized robot and camera environments.
    coinft_env : object | None
        Initialized CoinFT environment, if enabled.
    max_steps : int
        Maximum steps per episode.
    hz : float
        Control frequency.
    action_mode : str
        How to interpret the model's action output.
    """

    def __init__(
        self,
        env,
        cam_arm,
        cam_fix,
        coinft_env=None,
        *,
        server_host: str = "localhost",
        server_port: int = 10093,
        instruction: str = "",
        action_mode: str = "delta_eef",
        max_steps: int = 800,
        hz: float = 10.0,
        debug: bool = False,
    ):
        self.instruction = instruction
        self.max_steps = max_steps
        self.dt = 1.0 / hz
        self.hz = hz
        self.action_mode = action_mode
        self.env = env
        self.cam_arm = cam_arm
        self.cam_fix = cam_fix
        self.coinft_env = coinft_env
        self.debug = debug

        # ---- VLA client ----
        logger.info("Connecting to VLA server at %s:%d ...", server_host, server_port)
        self.client = VLAClient(host=server_host, port=server_port)

        if not self.client.ping():
            raise ConnectionError("VLA server ping failed!")
        logger.info("Server connected. Metadata: %s", self.client.metadata)

    def run(self):
        """Run the deployment control loop (blocking)."""
        try:
            # Start a fresh CoinFT tare before moving the arm; keep both sensors unloaded.
            if self.coinft_env is not None:
                self.coinft_env.reset(timeout=60.0)

            obs = self.env.reset(close_gripper=False)
            logger.info("Robot reset. Starting deployment loop ...")
            logger.info("Instruction: %s", self.instruction)
            logger.info(
                "Action mode: %s, Hz: %.1f, Max steps: %d",
                self.action_mode, self.hz, self.max_steps,
            )

            for step_i in range(self.max_steps):
                t0 = time.monotonic()

                # ---- Capture images (HWC uint8) ----
                rgb_arm = _ensure_hwc(self.cam_arm.step()["rgb"])   # wrist
                rgb_fix = _ensure_hwc(self.cam_fix.step()["rgb"])   # base

                # ---- Build state: cart_pos(6) + gripper(1) ----
                state = build_state(obs)

                # ---- Read optional dual CoinFT wrench ----
                coinft_left = None
                coinft_right = None
                if self.coinft_env is not None:
                    coinft_obs = self.coinft_env.step()
                    if "left_wrench" in coinft_obs:
                        coinft_left = coinft_obs["left_wrench"]
                    if "right_wrench" in coinft_obs:
                        coinft_right = coinft_obs["right_wrench"]

                # ---- Inference ----
                result = self.client.predict(
                    images=[rgb_arm, rgb_fix],
                    instruction=self.instruction,
                    state=state,
                    coinft_left=coinft_left,
                    coinft_right=coinft_right,
                )

                actions = result.get("actions")
                if actions is None:
                    logger.warning("No actions returned at step %d", step_i)
                    continue

                # ---- Execute action ----
                # actions shape: (chunk_size, 7) — 取第一个 action
                action = actions[0] if actions.ndim > 1 else actions

                if self.debug:
                    eef_pose = np.asarray(obs["cart_pos"], dtype=float).copy()
                    logger.info(
                        "Step %d/%d eef_pose=%s",
                        step_i, self.max_steps,
                        np.array2string(eef_pose, precision=4, separator=",", max_line_width=1000),
                    )

                # 前 6 维: delta EEF [dx, dy, dz, dr, dp, dy]
                arm_action = action[:6]
                # 第 7 维：归一化夹爪动作，>0.5 → 1 张开，≤0.5 → 0 闭合。
                gripper_input = None
                gripper_action = None
                if len(action) > 6:
                    gripper_input = float(action[6])
                    gripper_action = GRIPPER_MAX if gripper_input > GRIPPER_THRESHOLD else 0

                if self.debug:
                    logger.info(
                        "Step %d/%d action_mode=%s arm_action=%s gripper_input=%s gripper_command=%s",
                        step_i, self.max_steps, self.action_mode,
                        np.array2string(arm_action, precision=4, separator=",", max_line_width=1000),
                        gripper_input, gripper_action,
                    )
                obs = self.env.step(
                    action=arm_action,
                    gripper_action=gripper_action,
                )

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
            self.cleanup()

    def cleanup(self):
        """Release the inference client; hardware is owned by the entry point."""
        self.client.close()
        logger.info("VLA client closed.")

