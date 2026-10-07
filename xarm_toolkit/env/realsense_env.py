"""RealSense camera environment based on Open3D RealSenseSensor API."""

from __future__ import annotations

import atexit
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import open3d as o3d

from xarm_toolkit.utils.logger import get_logger

logger = get_logger("xarm_toolkit.env.realsense")

# ---------------------------------------------------------------------------
# Camera configuration registry
# Add new cameras here — one dict per serial number.
# ---------------------------------------------------------------------------
CAMERA_CONFIGS: dict[str, dict[str, str]] = {
    "327122075644": {  # D435i (arm-mounted)
        "color_format": "RS2_FORMAT_RGB8",
        "color_resolution": "640,480",
        "depth_format": "RS2_FORMAT_Z16",
        "depth_resolution": "640,480",
        "fps": "30",
    },
    "f1271506": {  # L515 (fixed)
        "color_format": "RS2_FORMAT_RGB8",
        "color_resolution": "640,480",
        "depth_format": "RS2_FORMAT_Z16",
        "depth_resolution": "640,480",
        "fps": "30",
        "visual_preset": "RS2_L500_VISUAL_PRESET_MAX_RANGE",
    },
}

# Human-readable names used in automatically generated recording filenames.
CAMERA_NAMES: dict[str, str] = {
    "327122075644": "arm",
    "f1271506": "fix",
}

# Fallback for unknown serials (D435-series defaults).
_DEFAULT_CONFIG: dict[str, str] = {
    "color_format": "RS2_FORMAT_RGB8",
    "color_resolution": "640,480",
    "depth_format": "RS2_FORMAT_Z16",
    "depth_resolution": "640,480",
    "fps": "30",
}

# Valid observation modes.
VALID_MODES = ("rgb", "rgbd", "pcd")


class RealsenseEnv:
    """Thin wrapper around :class:`open3d.t.io.RealSenseSensor`.

    Parameters
    ----------
    serial:
        Camera serial number (required — avoids accidentally opening the
        wrong device).
    mode:
        Observation mode.

        * ``"rgb"``  — return only the colour image.
        * ``"rgbd"`` — return the RGBD image (colour + depth).
        * ``"pcd"``  — return RGBD image **and** the derived point cloud.
    record:
        If ``True``, capture is recorded to a ``.bag`` file.
    record_path:
        Recording directory. Relative paths are resolved against the current
        working directory. Defaults to ``debug``.
    record_file:
        Recording filename. Defaults to
        ``YYYYMMDD_HHMMSS_<camera>_debug.bag``.
    """

    def __init__(
        self,
        serial: str,
        mode: Literal["rgb", "rgbd", "pcd"] = "rgbd",
        record: bool = False,
        record_path: str | Path | None = None,
        record_file: str | None = None,
    ) -> None:
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}, expected one of {VALID_MODES}")
        self.mode = mode

        logger.info("Available RealSense devices: %s", o3d.t.io.RealSenseSensor.list_devices())

        # Build per-camera config dict.
        if serial in CAMERA_CONFIGS:
            cfg_dict = {**CAMERA_CONFIGS[serial], "serial": serial}
        else:
            logger.warning(
                "Unknown serial %s — falling back to default D435 config. "
                "Consider adding it to CAMERA_CONFIGS.",
                serial,
            )
            cfg_dict = {**_DEFAULT_CONFIG, "serial": serial}

        config = o3d.t.io.RealSenseSensorConfig(cfg_dict)

        self.rs = o3d.t.io.RealSenseSensor()
        self.record_file: Path | None = None
        if record:
            output_dir = Path(record_path) if record_path is not None else Path("debug")
            output_dir = output_dir.expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)

            if record_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                camera_name = CAMERA_NAMES.get(serial, "unknown")
                record_file = f"{timestamp}_{camera_name}_debug.bag"

            if Path(record_file).name != record_file:
                raise ValueError(
                    "record_file must be a filename without directory components"
                )
            if not record_file.lower().endswith(".bag"):
                record_file = f"{record_file}.bag"

            self.record_file = output_dir / record_file
            logger.info("Recording RealSense capture to %s", self.record_file.resolve())
            self.rs.init_sensor(config, 0, str(self.record_file))
            self.rs.start_capture(True)  # start recording with capture
        else:
            self.rs.init_sensor(config, 0)
            self.rs.start_capture()

        self.intrinsic_matrix: np.ndarray = self.rs.get_metadata().intrinsics.intrinsic_matrix
        self.depth_scale: float = self.rs.get_metadata().depth_scale

        atexit.register(self.cleanup)
        logger.info(
            "RealSense %s initialised (mode=%s, depth_scale=%.4f)",
            serial,
            mode,
            self.depth_scale,
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_observation(self) -> dict[str, Any]:
        im_rgbd: o3d.t.geometry.RGBDImage = self.rs.capture_frame(True, True)

        obs: dict[str, Any] = {
            "intrinsic_matrix": self.intrinsic_matrix,
            "depth_scale": self.depth_scale,
        }

        if self.mode == "rgb":
            obs["rgb"] = im_rgbd.color
        elif self.mode == "rgbd":
            obs["rgbd"] = im_rgbd
        elif self.mode == "pcd":
            obs["rgbd"] = im_rgbd
            obs["pcd"] = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                im_rgbd,
                intrinsics=o3d.core.Tensor(
                    self.intrinsic_matrix, dtype=o3d.core.Dtype.Float32
                ),
                depth_scale=self.depth_scale,
            )

        return obs

    def reset(self, action: Any = None) -> dict[str, Any]:
        """Return current observation (camera has no state to reset)."""
        return self._get_observation()

    def step(self, action: Any = None) -> dict[str, Any]:
        """Capture one frame and return the observation dict."""
        return self._get_observation()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Stop the capture stream."""
        logger.info("Stopping RealSense capture …")
        self.rs.stop_capture()


# ======================================================================
# Quick demo — dual-camera preview
# ======================================================================
if __name__ == "__main__":
    import traceback

    import cv2

    logger.info("Starting dual-camera preview …")

    rs_arm = RealsenseEnv(serial="327122075644", mode="rgb")
    rs_fix = RealsenseEnv(serial="f1271506", mode="rgb")

    try:
        while True:
            arm_obs = rs_arm.step()
            fix_obs = rs_fix.step()

            color_arm = np.asarray(arm_obs["rgb"])
            color_fix = np.asarray(fix_obs["rgb"])

            cv2.imshow("rgb_arm", cv2.cvtColor(color_arm, cv2.COLOR_RGB2BGR))
            cv2.imshow("rgb_fix", cv2.cvtColor(color_fix, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    except KeyboardInterrupt:
        logger.info("Preview interrupted by user.")
    except Exception:
        logger.error("Unexpected error:\n%s", traceback.format_exc())
    finally:
        cv2.destroyAllWindows()

