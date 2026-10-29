#!/usr/bin/env python3
"""Diffusion policy deployment script using xarm_toolkit.

Uses xarm_toolkit (XArmEnv + RealsenseEnv + SpacemouseAgent) for environment,
and loads a CleanDiffuser checkpoint (train11_force_vec_ext) for inference.

Usage::

    python scripts/deploy_diffusion.py --config path/to/config.yaml

Typical workflow:
    1. Script initialises arm, cameras, SpaceMouse.
    2. Moves arm to task-specific init offset (based on --task).
    3. SpaceMouse lets you manually adjust pose + gripper; press Enter to start.
    4. Loops: infer → execute Ta steps at CONTROL_FREQ → repeat.
    5. Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("DISPLAY", ":1")

from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from termcolor import cprint
from torchvision.transforms.v2 import Normalize, Resize

# ---- xarm_toolkit (our framework) ----
from xarm_toolkit.env.xarm_env import XArmEnv
from xarm_toolkit.env.realsense_env import RealsenseEnv
from xarm_toolkit.teleop.spacemouse import SpacemouseAgent, SpacemouseConfig
from xarm_toolkit.utils.logger import get_logger

# ---- CleanDiffuser (only library-level imports, no pipeline dependency) ----
from cleandiffuser.diffusion import ContinuousRectifiedFlow
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_condition.resnet import ResNet18
from cleandiffuser.nn_diffusion import DiT1d
from cleandiffuser.utils import MinMaxNormalizer


# ---------------------------------------------------------------------------
# Config loader (inlined from pipeline, no more pipeline dependency)
# ---------------------------------------------------------------------------
def load_config() -> dict:
    """Load YAML/JSON config via --config argument."""
    parser = argparse.ArgumentParser(description="Diffusion policy deployment")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/ubuntu/zsh/configs/dp_xarm_11_vec_ext.yaml",
        help="Path to config file (.json or .yaml/.yml)",
    )
    args = parser.parse_args()
    config_path = args.config

    if config_path.endswith((".yaml", ".yml")):
        try:
            from omegaconf import OmegaConf
            cfg_ = OmegaConf.load(config_path)
            return OmegaConf.to_container(cfg_, resolve=True)
        except Exception:
            try:
                import yaml
            except ImportError as e:
                raise ImportError(
                    "YAML config requested but neither omegaconf nor PyYAML is available.\n"
                    "pip install hydra-core  or  pip install pyyaml"
                ) from e
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Condition network (inlined from pipeline.train11_force_vec_ext)
# ---------------------------------------------------------------------------
class MultiViewResnetWithLowdimObsSeqCondition(IdentityCondition):
    """Multi-view ResNet + lowdim + force history condition encoder.

    Produces:
        vec_condition:  (b, To*lowdim_emb + force_emb)  = (b, 256)
        seq_condition:  (b, To, image_emb*2)             = (b, To, 512)
    """

    def __init__(
        self,
        image_sz: int = 224,
        in_channel: int = 3,
        lowdim: int = 7,
        force_dim: int = 6,
        T_force: int = 10,
        image_emb_dim: int = 256,
        lowdim_emb_dim: int = 64,
        force_emb_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__(dropout)
        self.T_force = T_force
        self.force_dim = force_dim

        self.resnet18_arm = ResNet18(image_sz=image_sz, in_channel=in_channel, emb_dim=image_emb_dim)
        self.resnet18_fix = ResNet18(image_sz=image_sz, in_channel=in_channel, emb_dim=image_emb_dim)
        self.lowdim_mlp = nn.Sequential(
            nn.Linear(lowdim, lowdim_emb_dim), nn.SiLU(), nn.Linear(lowdim_emb_dim, lowdim_emb_dim),
        )
        self.force_mlp = nn.Sequential(
            nn.Linear(T_force * force_dim, force_emb_dim), nn.SiLU(), nn.Linear(force_emb_dim, force_emb_dim),
        )

    def forward(self, condition: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None):
        image_arm = condition["image_arm"]  # (b, To, C, H, W)
        image_fix = condition["image_fix"]  # (b, To, C, H, W)
        lowdim = condition["lowdim"]        # (b, To, 7)
        force = condition["force"]          # (b, T_force, 6)

        b, To = image_arm.shape[0], image_arm.shape[1]

        # Image encode per frame
        image_feat_arm = self.resnet18_arm(image_arm.reshape(b * To, *image_arm.shape[-3:]))
        image_feat_fix = self.resnet18_fix(image_fix.reshape(b * To, *image_fix.shape[-3:]))
        image_feat_arm = image_feat_arm.reshape(b, To, -1)  # (b, To, 256)
        image_feat_fix = image_feat_fix.reshape(b, To, -1)  # (b, To, 256)

        # Force history → flatten → MLP
        force_feat = self.force_mlp(force.reshape(b, -1))  # (b, force_emb_dim)

        # Lowdim → MLP → flatten
        lowdim_feat = self.lowdim_mlp(lowdim).reshape(b, -1)  # (b, To*64)

        return {
            "vec_condition": torch.cat([lowdim_feat, force_feat], dim=-1),  # (b, 256)
            "seq_condition": torch.cat([image_feat_arm, image_feat_fix], dim=-1),  # (b, To, 512)
            "seq_condition_mask": None,
        }

logger = get_logger("deploy_diffusion")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CAM_ARM_SERIAL = "327122075644"  # D435i (arm-mounted)
CAM_FIX_SERIAL = "f1271506"  # L515 (fixed)

CONTROL_FREQ = 30.0
RECORD_VIDEO = True

# ---------------------------------------------------------------------------
# Non-blocking keyboard (for pre-inference adjustment)
# ---------------------------------------------------------------------------
class _KeyboardListener:
    """Non-blocking keyboard reader in raw terminal mode."""

    def __init__(self):
        import termios
        self._termios = termios
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        import tty
        tty.setraw(self._fd)

    def get_key(self) -> str | None:
        import select
        if select.select([sys.stdin], [], [], 0.0)[0]:
            return sys.stdin.read(1)
        return None

    def stop(self):
        self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._old)


# ---------------------------------------------------------------------------
# Video recorder (dual view)
# ---------------------------------------------------------------------------
class VideoRecorder:
    """Record two camera views simultaneously."""

    def __init__(self, filename_arm: str, filename_fix: str):
        self.writer_arm = cv2.VideoWriter(
            filename_arm, cv2.VideoWriter_fourcc(*"MP4V"), 30, (640, 480),
        )
        self.writer_fix = cv2.VideoWriter(
            filename_fix, cv2.VideoWriter_fourcc(*"MP4V"), 30, (640, 480),
        )

    def render(self, rgb_arm: np.ndarray | None, rgb_fix: np.ndarray | None):
        """Write one frame (expects HWC uint8 RGB)."""
        if rgb_arm is not None:
            self.writer_arm.write(cv2.cvtColor(rgb_arm, cv2.COLOR_RGB2BGR))
        if rgb_fix is not None:
            self.writer_fix.write(cv2.cvtColor(rgb_fix, cv2.COLOR_RGB2BGR))

    def finish(self):
        for w in (self.writer_arm, self.writer_fix):
            if w is not None:
                w.release()
        self.writer_arm = None
        self.writer_fix = None


# ---------------------------------------------------------------------------
# Helper: extract RGB from RealsenseEnv observation
# ---------------------------------------------------------------------------
def _get_rgb(cam_obs: dict, mode: str = "rgbd") -> np.ndarray:
    """Return (H, W, 3) uint8 numpy array from a RealsenseEnv observation.

    Handles both 'rgb' and 'rgbd' mode observations.
    """
    if "rgb" in cam_obs:
        return np.asarray(cam_obs["rgb"])  # (H, W, 3)
    elif "rgbd" in cam_obs:
        return np.asarray(cam_obs["rgbd"].color)  # (H, W, 3)
    else:
        raise KeyError(f"Camera obs has no 'rgb' or 'rgbd' key, got: {list(cam_obs.keys())}")


# ---------------------------------------------------------------------------
# Task-specific init offsets
# ---------------------------------------------------------------------------
TASK_OFFSETS = {
    "plug": lambda: (np.random.uniform(-50, 70), np.random.uniform(-20, 10), -200),
    "insert": lambda: (np.random.uniform(-50, 70), np.random.uniform(-20, 10), -200),
    "whiteboard": lambda: (-230, -40, -280),
    "stamp": lambda: (0, 0, -270),
    "press_button": lambda: (0, 0, 0),
    "vase": lambda: (-117, 53, 0),
    "push": lambda: (-120, 0, -170),
}


def _get_task_offset(task: str) -> tuple[float, float, float]:
    """Return (x_bias, y_bias, z_bias) for the given task name."""
    for key, fn in TASK_OFFSETS.items():
        if key in task:
            return fn() if callable(fn) else fn
    logger.warning("Unknown task %r — using zero offset.", task)
    return (0, 0, 0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ================================================================
    # 1. Load model config
    # ================================================================
    cfg = load_config()
    # 夹爪原始闭合/张开位置，仅由 Env 执行归一化映射。
    gripper_range = tuple(cfg.get("gripper_range", [0, 840]))

    devices = cfg.get("devices", [0])
    ckpt_path = cfg.get("inference_ckpt_path", "") or None
    model_name = cfg.get("model", "dit")
    image_size = int(cfg.get("image_size", 224))
    task = cfg.get("task", "pick_strawberry")
    horizon = int(cfg.get("horizon", cfg.get("Ta", 64)))
    Ta = int(cfg.get("Ta", horizon))
    To = int(cfg.get("To", 2))
    T_force = int(cfg.get("T_force", 10))
    device = f"cuda:{devices[0]}"

    logger.info(
        "Config: horizon=%d, Ta=%d, To=%d, T_force=%d, image=%d, task=%s",
        horizon, Ta, To, T_force, image_size, task,
    )

    # ================================================================
    # 2. Load normalizer
    # ================================================================
    normalizer_path = cfg.get("normalizer_path", None)
    if not normalizer_path or not os.path.exists(normalizer_path):
        raise ValueError(f"Normalizer file not found: {normalizer_path}")

    with open(normalizer_path, "r") as f:
        ninfo = json.load(f)

    pos_normalizer = MinMaxNormalizer(
        X_max=np.array(ninfo["pos"]["max"], dtype=np.float32),
        X_min=np.array(ninfo["pos"]["min"], dtype=np.float32),
    )
    action_normalizer = MinMaxNormalizer(
        X_max=np.array(ninfo["action"]["max"], dtype=np.float32),
        X_min=np.array(ninfo["action"]["min"], dtype=np.float32),
    )
    force_normalizer = MinMaxNormalizer(
        X_max=np.array(ninfo["force"]["max"], dtype=np.float32),
        X_min=np.array(ninfo["force"]["min"], dtype=np.float32),
    )
    cprint(f"  Normalizer loaded: {normalizer_path}", "green")

    # ================================================================
    # 3. Build & load model (keep original architecture)
    # ================================================================
    nn_diffusion = DiT1d(
        x_dim=13,  # 6 pos + 1 gripper + 6 ext_force
        x_seq_len=horizon,
        vec_emb_dim=256,
        seq_emb_dim=512,
        d_model=384,
        n_heads=6,
        depth=12,
        head_type="mlp",
        use_cross_attn=True,
        adaLN_on_cross_attn=True,
        timestep_emb_type="untrainable_fourier",
        timestep_emb_params={"scale": 0.2},
    )

    nn_condition = MultiViewResnetWithLowdimObsSeqCondition(
        image_sz=image_size,
        in_channel=3,
        lowdim=7,  # 6 pos + 1 gripper state
        force_dim=6,
        T_force=T_force,
        image_emb_dim=256,
        lowdim_emb_dim=64,
        force_emb_dim=128,
        dropout=0.0,
    )

    policy = ContinuousRectifiedFlow(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition,
    )

    policy.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    )
    policy = policy.to(device).eval()

    total_params = sum(p.numel() for p in policy.parameters())
    logger.info(
        "Model loaded: %s  |  %.2f M params  |  %s",
        ckpt_path,
        total_params / 1e6,
        next(policy.parameters()).dtype,
    )

    # Image transforms
    NORM_PARAMS = (0.5, 0.5, 0.5)
    normalize = Normalize(NORM_PARAMS, NORM_PARAMS)
    resize = Resize((image_size, image_size))

    # ================================================================
    # 4. Initialize environment — xarm_toolkit
    # ================================================================
    logger.info("Initialising XArmEnv (delta_eef, force=True) ...")
    env = XArmEnv(action_mode="delta_eef", use_force=True, gripper_range=gripper_range)

    logger.info("Initialising cameras ...")
    cam_arm = RealsenseEnv(serial=CAM_ARM_SERIAL, mode="rgbd")
    cam_fix = RealsenseEnv(serial=CAM_FIX_SERIAL, mode="rgbd")

    logger.info("Initialising SpaceMouse ...")
    agent = SpacemouseAgent(SpacemouseConfig(
        translation_scale=5.0,
        rotation_scale=0.004,
    ))

    # ================================================================
    # 5. Video & force logger setup
    # ================================================================
    video_dir = f"videos/{task}_11_vec_ext/{model_name}_horizon_{horizon}"
    os.makedirs(video_dir, exist_ok=True)
    recorder = None
    if RECORD_VIDEO:
        ts = time.strftime("%Y%m%d_%H%M%S")
        recorder = VideoRecorder(
            f"{video_dir}/{ts}_arm.mp4",
            f"{video_dir}/{ts}_fix.mp4",
        )
        logger.info("Recording → %s/%s_*.mp4", video_dir, ts)

    force_dir = f"force_data/{task}_11_vec_ext"
    os.makedirs(force_dir, exist_ok=True)
    existing = [
        int(f.replace("force_data_", "").replace(".txt", ""))
        for f in os.listdir(force_dir)
        if f.startswith("force_data_") and f.endswith(".txt")
        and f.replace("force_data_", "").replace(".txt", "").isdigit()
    ]
    force_idx = (max(existing) + 1) if existing else 1
    force_file = f"{force_dir}/force_data_{force_idx}.txt"
    logger.info("Force data → %s", force_file)

    # ================================================================
    # 6. Reset & move to task init offset
    # ================================================================
    obs = env.reset(close_gripper=False)
    cam_arm_obs = cam_arm.reset()
    cam_fix_obs = cam_fix.reset()
    time.sleep(1)

    x_bias, y_bias, z_bias = _get_task_offset(task)
    logger.info("Task %r init offset: (%.0f, %.0f, %.0f)", task, x_bias, y_bias, z_bias)
    obs = env.step([x_bias, y_bias, z_bias, 0, 0, 0], gripper_action=0, speed=100)
    cam_arm_obs = cam_arm.step()
    cam_fix_obs = cam_fix.step()
    time.sleep(3)

    if recorder:
        recorder.render(_get_rgb(cam_arm_obs), _get_rgb(cam_fix_obs))

    # ================================================================
    # 7. Pre-inference: SpaceMouse adjustment + Enter to start
    # ================================================================
    kb = _KeyboardListener()
    goal_gripper = 0
    logger.info("SpaceMouse adjustment phase. Press Enter to start inference, Ctrl+C to abort.")

    try:
        while True:
            sm_action, sm_gripper = agent.act(obs)
            # Use SpaceMouse gripper toggle result directly
            goal_gripper = sm_gripper
            obs = env.step(sm_action, gripper_action=goal_gripper, speed=100)
            cam_arm_obs = cam_arm.step()
            cam_fix_obs = cam_fix.step()

            key = kb.get_key()
            if key == "\n" or key == "\r":
                cprint("Zeroing force sensor ...", "yellow")
                if env.reset_force_sensor_zero():
                    cprint("Force sensor zeroed!", "green")
                else:
                    cprint("Warning: force zero failed, continuing ...", "yellow")
                time.sleep(0.2)
                break
            elif key == "\x03":
                cprint("Aborted.", "red")
                kb.stop()
                sys.exit(0)
            time.sleep(0.1)
    except KeyboardInterrupt:
        kb.stop()
        sys.exit(0)
    finally:
        kb.stop()

    # One more obs after zero
    obs = env.step([0, 0, 0, 0, 0, 0], gripper_action=0)
    cam_arm_obs = cam_arm.step()
    cam_fix_obs = cam_fix.step()
    if recorder:
        recorder.render(_get_rgb(cam_arm_obs), _get_rgb(cam_fix_obs))

    # ================================================================
    # 8. Build initial observation tensors
    # ================================================================
    def _gripper_state(gripper_pos) -> float:
        """Binary gripper state: >0.5 → 1.0 (open), else 0.0 (closed)."""
        g = float(np.asarray(gripper_pos).item())
        return 1.0 if g > 0.5 else 0.0

    def _make_state_tensor(obs_dict: dict) -> torch.Tensor:
        """7-dim state: normalised 6D pos + binary gripper."""
        pos = np.asarray(obs_dict["goal_pos"], dtype=np.float32)
        pos = pos_normalizer.normalize(pos)
        gs = _gripper_state(obs_dict["gripper_position"])
        return torch.tensor(
            np.concatenate([pos, [gs]]), device=device, dtype=torch.float32,
        )

    def _make_force_tensor(obs_dict: dict) -> torch.Tensor:
        """6-dim normalised force."""
        f = np.asarray(obs_dict["ext_force"], dtype=np.float32)
        f = force_normalizer.normalize(f)
        return torch.tensor(f, device=device, dtype=torch.float32)

    def _make_image_tensor(rgb_np: np.ndarray) -> torch.Tensor:
        """(C, H', W') normalised image tensor."""
        img = rgb_np.astype(np.float32).transpose(2, 0, 1) / 255.0
        img = torch.tensor(img, device=device, dtype=torch.float32)
        img = resize(img)
        img = normalize(img)
        return img

    # Allocate buffers
    prior = torch.zeros((1, horizon, 13), device=device)

    state0 = _make_state_tensor(obs)
    states = state0.unsqueeze(0).unsqueeze(0).repeat(1, To, 1)  # (1, To, 7)

    force0 = _make_force_tensor(obs)
    forces = force0.unsqueeze(0).unsqueeze(0).repeat(1, T_force, 1)  # (1, T_force, 6)

    img_arm0 = _make_image_tensor(_get_rgb(cam_arm_obs))
    rgb_arm = img_arm0.unsqueeze(0).repeat(To, 1, 1, 1).unsqueeze(0)  # (1, To, C, H, W)

    img_fix0 = _make_image_tensor(_get_rgb(cam_fix_obs))
    rgb_fix = img_fix0.unsqueeze(0).repeat(To, 1, 1, 1).unsqueeze(0)  # (1, To, C, H, W)

    # ================================================================
    # 9. Inference loop
    # ================================================================
    dt = 1.0 / CONTROL_FREQ

    try:
        chunk_count = 0
        while True:
            chunk_count += 1
            logger.info("=== Inference round %d ===", chunk_count)

            # --- Diffusion inference ---
            condition_cfg = {
                "image_arm": rgb_arm,
                "image_fix": rgb_fix,
                "lowdim": states,
                "force": forces,
            }

            goal_states, _log = policy.sample(
                prior,
                solver="euler",
                sample_steps=20,
                condition_cfg=condition_cfg,
                use_ema=False,
                w_cfg=1.0,
            )

            act = goal_states[0].cpu().numpy()  # (horizon, 13)

            # Split: 6 delta_pose + 1 gripper + 6 ext_force
            act_6d = action_normalizer.unnormalize(act[:, :6])
            act_gripper_binary = (act[:, 6:7] > 0.5).astype(np.float32)
            act_gripper_exec = np.where(act_gripper_binary > 0.5, 1, 0).astype(int).ravel()
            act_ext_force_raw = force_normalizer.unnormalize(act[:, 7:])

            logger.info("Executing %d / %d steps ...", Ta, horizon)

            # --- Execute Ta steps ---
            for i in range(Ta):
                t0 = time.time()

                obs = env.step(act_6d[i], gripper_action=int(act_gripper_exec[i]))
                cam_arm_obs = cam_arm.step()
                cam_fix_obs = cam_fix.step()

                real_fz = obs["ext_force"][2] if obs["ext_force"] is not None else float("nan")
                pred_fz = act_ext_force_raw[i][2]
                logger.info(
                    "  step %d/%d  real_fz=%.3f  pred_fz=%.3f",
                    i, Ta, real_fz, pred_fz,
                )

                with open(force_file, "a") as f:
                    f.write(
                        f"Chunk {chunk_count}, Step {i}: "
                        f"real_ext_force_z={real_fz:.6f}, "
                        f"predicted_ext_force_z={pred_fz:.6f}\n"
                    )

                rgb_arm_np = _get_rgb(cam_arm_obs)
                rgb_fix_np = _get_rgb(cam_fix_obs)

                if recorder:
                    recorder.render(rgb_arm_np, rgb_fix_np)

                # Update force history (last T_force steps only)
                if i >= max(0, Ta - T_force):
                    fidx = i - max(0, Ta - T_force)
                    forces[0, fidx] = _make_force_tensor(obs)

                # Update observation history (last To steps only)
                if i >= max(0, Ta - To):
                    oidx = i - max(0, Ta - To)
                    states[0, oidx] = _make_state_tensor(obs)
                    rgb_arm[0, oidx] = _make_image_tensor(rgb_arm_np)
                    rgb_fix[0, oidx] = _make_image_tensor(rgb_fix_np)

                # Frequency control
                elapsed = time.time() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception:
        logger.error("Unexpected error:\n%s", traceback.format_exc())
    finally:
        if recorder:
            recorder.finish()
        logger.info("Done.")


if __name__ == "__main__":
    main()
