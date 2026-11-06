"""SpaceMouse teleoperation agent for xArm6.

The module separates input acquisition, agent behavior, and gripper button
policy into layers:

1. ``SpaceMouseExpert`` reads raw axes and button states from the device.
2. ``SpacemouseAgent`` is the stable public entry point. It scales/remaps the
   six motion axes and delegates button handling to the selected controller.
3. ``BinaryGripperController`` and ``ContinuousGripperController`` implement
   independent button policies and keep their own state.

``SpacemouseAgent.act()`` keeps the same return shape for both policies, so
callers do not need to know which controller handles the buttons. The button
policy is independent of how a collector stores gripper data.

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

# Normalized opening change per SpacemouseAgent.act() call while a button is held.
GRIPPER_CONTINUOUS_STEP = 0.01

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
    button_mode : str
        Selects the button policy, independently of dataset encoding:
        ``"binary"`` toggles between endpoints; ``"continuous"`` changes the
        normalized target by ``GRIPPER_CONTINUOUS_STEP`` while held.
    gripper_open_pos : int
        Normalized gripper opening when open (0–1, default 1).
    gripper_close_pos : int
        Normalized gripper opening when closed (0–1, default 0).
    """

    translation_scale: float = 5.0
    z_scale: float | None = None  # None → use translation_scale
    rotation_scale: float = 0.004
    deadzone: float = 0.0
    button_mode: str = "binary"
    gripper_open_pos: int = 1
    gripper_close_pos: int = 0


# ---------------------------------------------------------------------------
# Gripper button controllers
# ---------------------------------------------------------------------------

class BinaryGripperController:
    """Binary button policy with state isolated from continuous control.

    Button 0 toggles the target on its rising edge. The controller remembers
    only the binary open/closed state and returns one configured endpoint.
    """

    def __init__(self, config: SpacemouseConfig):
        self.cfg = config
        self._gripper_open = True
        self._prev_btn0 = False

    def act(self, buttons) -> int:
        btn0 = bool(buttons[0])
        if btn0 and not self._prev_btn0:
            self._gripper_open = not self._gripper_open
            logger.debug("Gripper toggled → %s", "OPEN" if self._gripper_open else "CLOSE")
        self._prev_btn0 = btn0
        return (
            self.cfg.gripper_open_pos if self._gripper_open
            else self.cfg.gripper_close_pos
        )

    @property
    def gripper_is_open(self) -> bool:
        return self._gripper_open


class ContinuousGripperController:
    """Continuous button policy with an independent normalized target.

    The target starts at the configured open endpoint. Holding button 0 lowers
    it; holding button 1 raises it. Releasing both buttons returns the current
    target unchanged. Pressing both also leaves the target unchanged.
    """

    def __init__(self, config: SpacemouseConfig):
        self.cfg = config
        self._position = float(self.cfg.gripper_open_pos)

    def act(self, buttons) -> float:
        btn0 = bool(buttons[0])
        btn1 = bool(buttons[1])
        if btn0 and not btn1:
            self._position = float(np.clip(
                self._position - GRIPPER_CONTINUOUS_STEP,
                self.cfg.gripper_close_pos,
                self.cfg.gripper_open_pos,
            ))
        elif btn1 and not btn0:
            self._position = float(np.clip(
                self._position + GRIPPER_CONTINUOUS_STEP,
                self.cfg.gripper_close_pos,
                self.cfg.gripper_open_pos,
            ))
        return self._position

    @property
    def gripper_is_open(self) -> bool:
        """A continuous target has no binary open/closed state."""
        raise NotImplementedError("gripper_is_open is only available in binary button mode")


# ---------------------------------------------------------------------------
# SpacemouseAgent — public API
# ---------------------------------------------------------------------------

class SpacemouseAgent:
    """Shared SpaceMouse agent and public facade for button controllers.

    The agent owns device reading and six-axis motion conversion. It creates
    exactly one gripper controller from ``SpacemouseConfig.button_mode`` and
    delegates button-state handling to it. Binary toggle state never enters
    the continuous controller, and the continuous target never enters the
    binary controller.

    The external call remains ``action, gripper = agent.act()`` regardless of
    the selected mode. Dataset binary/continuous storage is configured by the
    collector separately and is not selected here.

    Usage
    -----
    >>> agent = SpacemouseAgent()
    >>> action, gripper = agent.act(obs)   # obs from XArmEnv
    >>> obs = env.step(action, gripper_action=gripper)

    The returned ``action`` is a 6D array ``[dx, dy, dz, droll, dpitch, dyaw]``
    ready for ``XArmEnv.step()`` in ``delta_eef`` mode.

    Button mapping in binary mode:
        - button[0] (left)  → toggle gripper open/close
        - button[1] (right) → (reserved, unused)

    Button mapping in continuous mode:
        - button[0] (left)  → decrease opening while held (close)
        - button[1] (right) → increase opening while held (open)
        - neither or both   → hold the current target
    """

    def __init__(self, config: SpacemouseConfig | None = None):
        self.cfg = config or SpacemouseConfig()
        self._expert = SpaceMouseExpert()
        # Strategy selection stays inside the facade; callers keep using the
        # same SpacemouseAgent and act() interface for either button policy.
        gripper_controllers = {
            "binary": BinaryGripperController,
            "continuous": ContinuousGripperController,
        }
        self._gripper_controller = gripper_controllers[self.cfg.button_mode](self.cfg)

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

    def act(self, obs=None) -> Tuple[np.ndarray, float | int]:
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
        gripper_action : float | int
            Normalized gripper target. Binary mode returns the configured
            open/close endpoint; continuous mode returns the current target.
        """
        # The low-level reader supplies raw device values; this agent transforms
        # only the motion axes and passes buttons to the selected policy.
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

        gripper_action = self._gripper_controller.act(buttons)

        return action, gripper_action

    @property
    def gripper_is_open(self) -> bool:
        return self._gripper_controller.gripper_is_open
