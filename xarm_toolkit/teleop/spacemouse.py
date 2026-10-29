"""SpaceMouse teleoperation agent for xArm6.

Uses SpaceMouseExpert (spacemouse_expert.py, from SERL) as the low-level
threaded HID reader, and adds axis scaling / rotation remapping / gripper
toggle logic on top.

Reference:
    - spacemouse_expert.py: SERL SpaceMouseExpert (threaded HID reader, DO NOT MODIFY)
    - reference/spacemouse_slow.py: axis mapping & scaling for xArm6
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from xarm_toolkit.teleop.spacemouse_expert import SpaceMouseExpert
from xarm_toolkit.utils.logger import get_logger

logger = get_logger("xarm_toolkit.teleop")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpacemouseConfig:
    """SpaceMouse mapping & scaling parameters.

    Attributes
    ----------
    translation_scale : float
        Multiplier for x/y translation (mm per raw unit).
    z_scale : float
        Multiplier for z translation; defaults to same as translation_scale.
    rotation_scale : float
        Multiplier converting raw rotation to rad delta.
    deadzone : float
        Ignore raw values whose absolute value is below this threshold.
    gripper_open_pos : int
        Normalized gripper opening when open (0–1, default 1).
    gripper_close_pos : int
        Normalized gripper opening when closed (0–1, default 0).
    """

    translation_scale: float = 5.0
    z_scale: float | None = None  # None → use translation_scale
    rotation_scale: float = 0.004
    deadzone: float = 0.0
    gripper_open_pos: int = 1
    gripper_close_pos: int = 0


# ---------------------------------------------------------------------------
# SpacemouseAgent — public API
# ---------------------------------------------------------------------------

class SpacemouseAgent:
    """Teleoperation agent: SpaceMouse -> delta EEF action for XArmEnv.

    Usage
    -----
    >>> agent = SpacemouseAgent()
    >>> action, gripper = agent.act(obs)   # obs from XArmEnv
    >>> obs = env.step(action, gripper_action=gripper)

    The returned ``action`` is a 6D array ``[dx, dy, dz, droll, dpitch, dyaw]``
    ready for ``XArmEnv.step()`` in ``delta_eef`` mode.

    Button mapping (default):
        - button[0] (left)  → toggle gripper open/close
        - button[1] (right) → (reserved, unused)
    """

    def __init__(self, config: SpacemouseConfig | None = None):
        self.cfg = config or SpacemouseConfig()
        self._expert = SpaceMouseExpert()

        # Gripper state: starts open
        self._gripper_open = True
        self._prev_btn0 = False  # edge detection for toggle

        z_scale = self.cfg.z_scale if self.cfg.z_scale is not None else self.cfg.translation_scale

        # Pre-build per-axis scale vector for fast multiply
        self._scale = np.array([
            self.cfg.translation_scale,   # dx
            self.cfg.translation_scale,   # dy
            z_scale,                      # dz
            self.cfg.rotation_scale,      # droll
            self.cfg.rotation_scale,      # dpitch
            self.cfg.rotation_scale,      # dyaw
        ], dtype=np.float64)

        logger.info(
            "SpacemouseAgent ready — trans_scale=%.1f rot_scale=%.4f deadzone=%.3f",
            self.cfg.translation_scale, self.cfg.rotation_scale, self.cfg.deadzone,
        )

    # ------------------------------------------------------------------

    def act(self, obs=None) -> Tuple[np.ndarray, int]:
        """Read SpaceMouse and return (delta_action, normalized_gripper_opening).

        Parameters
        ----------
        obs : dict | None
            Current observation from XArmEnv (reserved for future use,
            e.g. force-feedback scaling).  Currently unused.

        Returns
        -------
        action : np.ndarray, shape (6,)
            ``[dx, dy, dz, droll, dpitch, dyaw]`` — ready for
            ``XArmEnv.step(action)`` in ``delta_eef`` mode.
        gripper_action : int
            Normalized gripper target (0 = closed, 1 = open by default).
        """
        raw, buttons = self._expert.get_action()

        # --- Deadzone ---
        if self.cfg.deadzone > 0:
            raw = raw.copy()
            mask = np.abs(raw) < self.cfg.deadzone
            raw[mask] = 0.0

        # --- Scale ---
        action = raw.copy()
        # Translation
        action[:3] *= self._scale[:3]

        # Rotation: remap for 90-degree gripper rotation
        #   reference/spacemouse_slow.py 中的映射:
        #     roll  = -raw_pitch
        #     pitch =  raw_roll
        #     yaw   =  raw_yaw
        original_rot = raw[3:] * self._scale[3:]
        action[3] = -original_rot[1]   # roll  ← -pitch
        action[4] = original_rot[0]    # pitch ←  roll
        action[5] = original_rot[2]    # yaw   ←  yaw

        # --- Gripper toggle (rising edge on button 0) ---
        btn0 = bool(buttons[0])
        if btn0 and not self._prev_btn0:
            self._gripper_open = not self._gripper_open
            logger.debug("Gripper toggled → %s", "OPEN" if self._gripper_open else "CLOSE")
        self._prev_btn0 = btn0

        gripper_action = (
            self.cfg.gripper_open_pos if self._gripper_open
            else self.cfg.gripper_close_pos
        )

        return action, gripper_action

    @property
    def gripper_is_open(self) -> bool:
        return self._gripper_open
