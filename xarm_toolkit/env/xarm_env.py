"""Unified XArm robot environment.

Consolidates xarm_env.py, xarm_env_force.py and xarm_env_no_force.py into a
single class with configurable force sensing and action modes.
"""

from __future__ import annotations

import atexit
import math
import time
from typing import Literal

import numpy as np
from pytransform3d import rotations as pr
from xarm.wrapper import XArmAPI

from xarm_toolkit.utils.logger import get_logger

logger = get_logger("xarm_toolkit.env")

# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

class XArmCommandError(RuntimeError):
    """Raised when an xArm SDK call fails after exhausting retries."""


def _retry(fn, *, clear_fn=None, max_retries: int = 5, label: str = ""):
    """Call *fn* until it returns ``(0, ...)``.  On non-zero return code,
    optionally call *clear_fn* then retry up to *max_retries* times.

    Returns the result of *fn* (code stripped when it is a tuple).
    Raises :class:`XArmCommandError` if retries are exhausted.
    """
    for attempt in range(1, max_retries + 2):  # 1 initial + max_retries
        result = fn()
        # SDK returns either a plain int code or (code, payload)
        code = result if isinstance(result, int) else result[0]
        if code == 0:
            return result
        logger.warning(
            "Error code %d in %s (attempt %d/%d)",
            code, label, attempt, max_retries + 1,
        )
        if attempt <= max_retries and clear_fn is not None:
            clear_fn()
    raise XArmCommandError(
        f"{label} failed with code {code} after {max_retries + 1} attempts"
    )


# ---------------------------------------------------------------------------
# Action modes
# ---------------------------------------------------------------------------

ActionMode = Literal["delta_eef", "absolute_eef", "absolute_joint"]

_VALID_ACTION_MODES: set[str] = {"delta_eef", "absolute_eef", "absolute_joint"}

# ---------------------------------------------------------------------------
# XArmEnv
# ---------------------------------------------------------------------------

class XArmEnv:
    """Unified environment wrapper for the xArm robot.

    Parameters
    ----------
    addr : str
        IP address of the xArm controller.
    use_force : bool
        If True, enable the FT sensor for force readings (requires hardware).
        If False, skip all FT-related calls; force fields in obs will be None.
    action_mode : ActionMode
        One of ``"delta_eef"``, ``"absolute_eef"``, ``"absolute_joint"``.
    initial_gripper_position : float
        Normalized gripper position on init (0 = closed, 1 = open).
    gripper_range : tuple[int, int]
        Raw closed/open positions. Observations report normalized positions.
    max_retries : int
        Max retries for SDK calls before raising an exception.
    """

    # 复位位姿（笛卡尔）
    RESET_POSE = np.array([470, 0, 530, math.pi, 0, -math.pi / 2])
    # XArm6: 6 关节 + 1 占位（SDK 要求 7 个值，最后一位为占位 0）
    RESET_JOINT_ANGLES = [0, 0, -math.pi / 2, 0, math.pi / 2, math.pi / 2, 0]

    def __init__(
        self,
        addr: str = "192.168.31.232",
        use_force: bool = False,
        action_mode: ActionMode = "delta_eef",
        initial_gripper_position: float = 1.0,
        max_retries: int = 10,
        gripper_range: tuple[int, int] = (0, 840),
    ):
        if action_mode not in _VALID_ACTION_MODES:
            raise ValueError(
                f"Invalid action_mode={action_mode!r}. "
                f"Choose from {_VALID_ACTION_MODES}"
            )

        self.use_force = use_force
        self.action_mode: ActionMode = action_mode
        self.max_retries = max_retries
        self.gripper_range = tuple(gripper_range)
        if not 0 <= self.gripper_range[0] < self.gripper_range[1] <= 840:
            raise ValueError("gripper_range must satisfy 0 <= min < max <= 840")

        self.reset_pose = self.RESET_POSE.copy()
        self.reset_joint_angles = list(self.RESET_JOINT_ANGLES)

        # goal_pos tracks the desired EEF pose (used by delta/absolute eef)
        self.goal_pos = self.reset_pose.copy()

        # ---- Connect ----
        logger.info("Connecting to xArm at %s ...", addr)
        self.arm = XArmAPI(
            addr,
            report_type="real",  # fast force feedback
        )
        if self.arm is None:
            raise RuntimeError(f"Failed to connect to xArm at {addr}")
        logger.info("Connected.")

        # ---- Initialise ----
        self._clear_error_states()
        self._set_gripper()
        self.arm.set_gripper_position(
            self._gripper_to_raw(initial_gripper_position), wait=True, speed=5000
        )

        if self.use_force:
            self._enable_force_sensor()

        atexit.register(self.cleanup)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clear_error_states(self, mode: int = 7):
        """Clean errors, enable motion, set SDK mode."""
        if self.arm is None:
            raise RuntimeError("arm is not connected")
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        self.arm.set_mode(mode)
        self.arm.set_state(state=0)

    def _set_gripper(self):
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(5000)

    # ---- Force sensor ----

    def _enable_force_sensor(self):
        """Enable FT sensor communication and set zero."""
        self._clear_error_states()
        time.sleep(0.1)

        code = self.arm.ft_sensor_enable(1)
        if code != 0:
            logger.warning("ft_sensor_enable() returned code %d", code)
        time.sleep(0.2)

        self._clear_error_states()
        time.sleep(0.1)

        code = self.arm.ft_sensor_set_zero()
        if code != 0:
            logger.warning("ft_sensor_set_zero() returned code %d", code)
        time.sleep(0.2)

        self._clear_error_states()

    def _disable_force_sensor(self):
        """Disable FT sensor communication."""
        self.arm.ft_sensor_enable(0)

    def reset_force_sensor_zero(self) -> bool:
        """Re-zero the FT sensor (e.g. before a collection episode).

        Returns True on success.
        """
        if not self.use_force:
            logger.warning("reset_force_sensor_zero() called but use_force=False")
            return False
        code = self.arm.ft_sensor_set_zero()
        if code != 0:
            logger.warning("ft_sensor_set_zero() returned code %d", code)
        time.sleep(0.1)
        self._clear_error_states()
        return code == 0

    # ---- Movement primitives ----

    def _move_cartesian(self, pose: np.ndarray, speed: float = 1000):
        """Send a Cartesian set_position command with retry.

        Args:
            pose: [x, y, z, roll, pitch, yaw] in radian.
            speed: 笛卡尔运动速度，相对值，范围 0–2000。
        """
        def _cmd():
            return self.arm.set_position(
                x=pose[0], y=pose[1], z=pose[2],
                roll=pose[3], pitch=pose[4], yaw=pose[5],
                speed=speed, wait=False, is_radian=True,
            )
        _retry(
            _cmd,
            clear_fn=self._clear_error_states,
            max_retries=self.max_retries,
            label="set_position",
        )

    def _move_joint(self, angles, speed: float = 1):
        """Send a joint set_servo_angle command with retry (Mode 6).

        Args:
            angles: 目标关节角 [j1..j6]，单位 rad。
            speed: 关节角速度，单位 rad/s（1.0 ≈ 57°/s）。
        """
        if self.arm.mode != 6:
            self._clear_error_states(6)

        target = np.array(angles).tolist()

        def _cmd():
            return self.arm.set_servo_angle(
                angle=target, speed=speed, is_radian=True,
                relative=False, wait=False,
            )
        _retry(
            _cmd,
            clear_fn=lambda: self._clear_error_states(6),
            max_retries=self.max_retries,
            label="set_servo_angle",
        )

    def _gripper_to_raw(self, position: float) -> float:
        """Map normalized opening (0–1) to the configured raw position."""
        minimum, maximum = self.gripper_range
        return minimum + position * (maximum - minimum)

    def _move_gripper(self, position):
        """Send a normalized gripper command with retry."""
        raw_position = self._gripper_to_raw(position)
        def _cmd():
            return self.arm.set_gripper_position(raw_position, wait=False)
        _retry(
            _cmd,
            clear_fn=self._clear_error_states,
            max_retries=self.max_retries,
            label="set_gripper_position",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ---- Observation ----

    def _get_observation(self) -> dict[str, np.ndarray | None]:
        """Read and return the current observation dict."""
        # Cart pos
        result = _retry(
            lambda: self.arm.get_position(is_radian=True),
            clear_fn=self._clear_error_states,
            max_retries=self.max_retries,
            label="get_position",
        )
        cart_pos = result[1]

        # Servo angles
        result = _retry(
            lambda: self.arm.get_servo_angle(is_radian=True),
            clear_fn=self._clear_error_states,
            max_retries=self.max_retries,
            label="get_servo_angle",
        )
        servo_angle = result[1][:6]

        # Gripper
        result = _retry(
            lambda: self.arm.get_gripper_position(),
            clear_fn=self._clear_error_states,
            max_retries=self.max_retries,
            label="get_gripper_position",
        )
        minimum, maximum = self.gripper_range
        gripper_position = np.clip((result[1] - minimum) / (maximum - minimum), 0.0, 1.0)

        # Force
        if self.use_force:
            ext_force = np.array(self.arm.ft_ext_force)
            raw_force = np.array(self.arm.ft_raw_force)
        else:
            ext_force = None
            raw_force = None

        return {
            "cart_pos": np.array(cart_pos),
            "servo_angle": np.array(servo_angle),
            "ext_force": ext_force,
            "raw_force": raw_force,
            "goal_pos": np.array(self.goal_pos),
            "gripper_position": np.array(gripper_position),
        }

    # ---- Reset ----

    def reset(self, close_gripper: bool = True) -> dict[str, np.ndarray | None]:
        """Reset the arm to its home pose.

        Args:
            close_gripper: If True close gripper first; otherwise open it.

        Returns:
            Observation dict after reset.
        """
        logger.info("Resetting ...")

        if close_gripper:
            self.arm.set_gripper_position(self._gripper_to_raw(0.0), wait=False, speed=8000)
            time.sleep(1)
        else:
            self.arm.set_gripper_position(self._gripper_to_raw(1.0), wait=False, speed=8000)

        # Disable force sensor for the reset motion
        if self.use_force:
            self._disable_force_sensor()

        self._clear_error_states(6)  # mode 6 for joint-level reset
        time.sleep(0.1)

        self.goal_pos = self.reset_pose.copy()

        # Move to reset joint angles
        while True:
            self.arm.set_servo_angle(
                angle=self.reset_joint_angles, speed=0.5, is_radian=True
            )
            code, curr_pos = self.arm.get_position(is_radian=True)
            time.sleep(0.02)

            pos_dist = math.dist(self.goal_pos[:3], curr_pos[:3])
            rot_dist = pr.quaternion_dist(
                pr.quaternion_from_extrinsic_euler_xyz(self.goal_pos[3:6]),
                pr.quaternion_from_extrinsic_euler_xyz(curr_pos[3:6]),
            )
            logger.debug(
                "Resetting — pos_dist=%.1f rot_dist=%.4f curr_pos=%s",
                pos_dist, rot_dist, curr_pos,
            )

            if pos_dist < 20 and rot_dist < 0.02:
                break

        # Restore
        self._clear_error_states()
        time.sleep(0.2)

        if self.use_force:
            self._enable_force_sensor()

        logger.info("Reset done.")
        return self._get_observation()

    # ---- Step ----

    def step(
        self,
        action,
        gripper_action=None,
        speed: float = 1000,
        joint_speed: float = 1.0,
    ) -> dict[str, np.ndarray | None]:
        """Execute one action step.

        Args:
            action: Action array whose semantics depend on ``action_mode``.
                - ``"delta_eef"``:  6D delta ``[dx,dy,dz,dr,dp,dy]``
                - ``"absolute_eef"``: 6D absolute ``[x,y,z,r,p,y]``
                - ``"absolute_joint"``: 6D joint angles ``[j1..j6]`` (rad)
            gripper_action: Normalized opening (0–1), or None to skip.
            speed: EEF 模式下的笛卡尔运动速度，相对值，范围 0–2000。
            joint_speed: 关节模式下的角速度，单位 rad/s（1.0 ≈ 57°/s）。

        Returns:
            Observation dict.
        """
        action = np.asarray(action, dtype=np.float64)

        if self.action_mode == "delta_eef":
            self.goal_pos = self.goal_pos + action
            self._move_cartesian(self.goal_pos, speed)
        elif self.action_mode == "absolute_eef":
            self.goal_pos = action.copy()
            self._move_cartesian(self.goal_pos, speed)
        elif self.action_mode == "absolute_joint":
            self._move_joint(action, speed=joint_speed)
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode!r}")

        if gripper_action is not None:
            self._move_gripper(gripper_action)

        return self._get_observation()

    # ---- Cleanup ----

    def cleanup(self):
        """Disconnect and disable force sensor (registered via atexit)."""
        logger.info("Cleaning up ...")
        if self.use_force:
            try:
                self._disable_force_sensor()
            except Exception:
                pass
        try:
            self.arm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XArmEnv demo")
    parser.add_argument(
        "--mode",
        choices=["delta_eef", "absolute_eef", "absolute_joint"],
        default="delta_eef",
    )
    args = parser.parse_args()

    env = XArmEnv(
        action_mode=args.mode,
        use_force=False,
        initial_gripper_position=1.0,
    )
    obs = env.reset()
    print("Reset obs keys:", list(obs.keys()))
    print("  cart_pos:", obs["cart_pos"])
    print("  servo_angle:", obs["servo_angle"])
    print("  gripper:", obs["gripper_position"])

    # --- delta_eef: 遥操作 / spacemouse 采集 ---
    if args.mode == "delta_eef":
        for i in range(50):
            obs = env.step(
                action=[0, 0, 0, 0, 0, 0],   # 6D 增量 [dx,dy,dz,dr,dp,dy]
                gripper_action=None,
                speed=1000,
            )
            time.sleep(0.02)

    # --- absolute_eef: VLA 部署（笛卡尔） ---
    elif args.mode == "absolute_eef":
        target = obs["cart_pos"].copy()
        target[2] += 50  # z 方向抬高 50mm
        for i in range(5):
            obs = env.step(
                action=target,                # 6D 绝对位姿 [x,y,z,r,p,y]
                gripper_action=None,
                speed=1000,
            )
            time.sleep(0.02)

    # --- absolute_joint: Pi0 关节部署 ---
    elif args.mode == "absolute_joint":
        target_joints = obs["servo_angle"].copy()
        target_joints[1] += 0.1  # 第2个关节偏移 0.1 rad
        for i in range(5):
            obs = env.step(
                action=target_joints,         # 6D 关节角 [j1..j6] (rad)
                gripper_action=None,
                joint_speed=0.5,              # rad/s
            )
            time.sleep(0.02)

    print("Done. Final cart_pos:", obs["cart_pos"])
