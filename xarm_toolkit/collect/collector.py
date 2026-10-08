"""Data collector: env + cameras + teleop → Zarr dataset.

Manages the full collect loop:
  1. Init env / cameras / SpaceMouse
  2. Per-episode: reset → move to task start pos → wait for start signal
     → record steps → confirm/save buffer → repeat
  3. Compute episode_ends metadata

Features:
  - Episode 崩溃恢复：单个 episode 报错不会杀死整个采集
  - 优雅 Ctrl+C：中断后仍输出最终总结、已保存数据完好
  - 启动横幅 & Episode 摘要：清晰展示配置和采集进度
  - 断点续采：基于已有 episode 数自动续采

Reference: reference/collect.py
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
import zarr

from xarm_toolkit.utils.logger import get_logger

logger = get_logger("xarm_toolkit.collect")


# ---------------------------------------------------------------------------
# EpisodeStats
# ---------------------------------------------------------------------------

@dataclass
class EpisodeStats:
    """Statistics for a single completed episode."""
    steps: int
    duration: float
    fps: float


# ---------------------------------------------------------------------------
# Keyboard listener (non-blocking, raw terminal)
# ---------------------------------------------------------------------------

class _KeyboardListener:
    """Non-blocking keyboard reader in raw terminal mode."""

    def __init__(self):
        import termios
        self._termios = termios
        self._old_settings = termios.tcgetattr(sys.stdin)

    def start(self):
        import tty
        tty.setraw(sys.stdin.fileno())

    def stop(self):
        self._termios.tcsetattr(sys.stdin, self._termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> str | None:
        import select
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1).lower()
        return None


# ---------------------------------------------------------------------------
# Zarr dataset helper
# ---------------------------------------------------------------------------

def _open_or_create_zarr(
    path: str,
    image_shape: tuple[int, ...] = (3, 240, 320),
    cam_mode: str = "rgbd",
) -> tuple[zarr.Group, zarr.Group, int]:
    """Open existing or create new Zarr dataset.

    Returns (data_group, meta_group, start_episode).
    """
    dataset_path = str(path)
    exists = os.path.exists(dataset_path)

    if exists:
        logger.info("Opening existing dataset: %s", dataset_path)
        ds = zarr.open(dataset_path, mode="a")
        data = ds["data"]
        meta = ds["meta"]
        if "episode" in data:
            start_ep = len(np.unique(data["episode"][:]))
            logger.info("Found %d existing episodes", start_ep)
        else:
            start_ep = 0
        return data, meta, start_ep

    logger.info("Creating new dataset: %s (cam_mode=%s)", dataset_path, cam_mode)
    ds = zarr.open(dataset_path, mode="w")
    data = ds.create_group("data")
    meta = ds.create_group("meta")

    # Image compressor: Blosc zstd
    try:
        from numcodecs import Blosc
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except ImportError:
        compressor = None

    # RGB — always stored
    data.require_dataset(
        "rgb_arm", shape=(0, *image_shape), dtype=np.uint8,
        chunks=(1, *image_shape), compressor=compressor,
    )
    data.require_dataset(
        "rgb_fix", shape=(0, *image_shape), dtype=np.uint8,
        chunks=(1, *image_shape), compressor=compressor,
    )

    # Depth — stored for rgbd / pcd modes
    if cam_mode in ("rgbd", "pcd"):
        depth_shape = (1, image_shape[1], image_shape[2])  # (1, H, W)
        data.require_dataset(
            "depth_arm", shape=(0, *depth_shape), dtype=np.uint16,
            chunks=(1, *depth_shape), compressor=compressor,
        )
        data.require_dataset(
            "depth_fix", shape=(0, *depth_shape), dtype=np.uint16,
            chunks=(1, *depth_shape), compressor=compressor,
        )

    data.require_dataset("pos", shape=(0, 6), dtype=np.float32)
    data.require_dataset("force", shape=(0, 6), dtype=np.float32)
    data.require_dataset("action", shape=(0, 6), dtype=np.float32)
    data.require_dataset("gripper_state", shape=(0, 1), dtype=np.float32)
    data.require_dataset("gripper_action", shape=(0, 1), dtype=np.float32)
    data.require_dataset("episode", shape=(0,), dtype=np.uint16)

    return data, meta, 0


def _compute_episode_ends(data: zarr.Group, meta: zarr.Group):
    """Recompute ``meta/episode_ends`` from ``data/episode``."""
    all_ep = data["episode"][:]
    unique_eps = np.unique(all_ep)
    ends = []
    running = 0
    for ep in unique_eps:
        running += int(np.sum(all_ep == ep))
        ends.append(running)
    if "episode_ends" in meta:
        del meta["episode_ends"]
    meta.require_dataset("episode_ends", shape=(len(ends),), dtype=np.uint32)
    meta["episode_ends"][:] = np.array(ends, dtype=np.uint32)
    logger.info("episode_ends: %s (total %d episodes)", ends, len(ends))


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------

class Collector:
    """Orchestrates data collection: env + cameras + teleop → Zarr.

    Parameters
    ----------
    env : XArmEnv
        Robot environment (delta_eef mode, use_force=True recommended).
    cam_arm : RealsenseEnv
        Arm-mounted camera.
    cam_fix : RealsenseEnv
        Fixed camera.
    agent : SpacemouseAgent
        Teleoperation agent.
    dataset_path : str
        Path to Zarr dataset (created if not exists).
    task_config : dict | None
        Task-specific config with keys:
        - ``start_bias``: [x, y, z] offset from reset pose (default [0,0,0])
        - ``random_bias``: {x: [lo, hi], y: [lo, hi]} for randomised start (optional)
        - ``gripper_always_closed``: bool (for stamp-like tasks)
    num_episodes : int
        Number of episodes to collect in this session.
    cam_mode : str
        Camera observation mode: ``"rgb"``, ``"rgbd"`` (default), ``"pcd"``.
        - ``"rgb"``:  only save RGB images
        - ``"rgbd"``: save RGB + depth separately
        - ``"pcd"``:  save RGB + depth (point cloud computed from RealsenseEnv but not stored)
    image_size : tuple
        Target (W, H) for saved images (default 320×240).
    save_video : bool
        If True, save per-episode MP4 videos for arm and fix cameras
        alongside the dataset (e.g. ``datasets/demo_ep0_arm.mp4``).
    video_fps : float
        FPS for saved videos (default 15).
    warmup_time : float
        Seconds to wait after init before first episode.
    """

    def __init__(
        self,
        env,
        cam_arm,
        cam_fix,
        agent,
        dataset_path: str = "datasets/demo.zarr",
        task_config: dict | None = None,
        num_episodes: int = 3,
        cam_mode: str = "rgbd",
        image_size: tuple[int, int] = (320, 240),
        save_video: bool = False,
        video_fps: float = 15.0,
        warmup_time: float = 1.0,
    ):
        self.env = env
        self.cam_arm = cam_arm
        self.cam_fix = cam_fix
        self.agent = agent

        self.dataset_path = dataset_path
        self.task_cfg = task_config or {}
        self.num_episodes = num_episodes
        self.cam_mode = cam_mode
        self.image_w, self.image_h = image_size
        self.save_video = save_video
        self.video_fps = video_fps
        self.warmup_time = warmup_time

        self._save_depth = cam_mode in ("rgbd", "pcd")
        self._gripper_always_closed = self.task_cfg.get("gripper_always_closed", False)

    # ------------------------------------------------------------------
    # Task start position
    # ------------------------------------------------------------------

    def _get_start_bias(self) -> np.ndarray:
        """Compute [x, y, z, 0, 0, 0] bias for task start position."""
        base = np.array(self.task_cfg.get("start_bias", [0, 0, 0]), dtype=np.float64)

        # Optional random offset
        rand_cfg = self.task_cfg.get("random_bias", {})
        for i, axis in enumerate(["x", "y", "z"]):
            if axis in rand_cfg:
                lo, hi = rand_cfg[axis]
                base[i] += np.random.uniform(lo, hi)

        return np.array([base[0], base[1], base[2], 0, 0, 0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Helpers: disk usage, banner, summaries
    # ------------------------------------------------------------------

    @staticmethod
    def _get_disk_usage(path: str) -> str:
        """Recursively compute directory size, return human-readable string."""
        total = 0
        for dirpath, _dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total += os.path.getsize(fp)
                except OSError:
                    pass
        if total < 1024:
            return f"{total} B"
        elif total < 1024 ** 2:
            return f"{total / 1024:.1f} KB"
        elif total < 1024 ** 3:
            return f"{total / 1024 ** 2:.1f} MB"
        else:
            return f"{total / 1024 ** 3:.2f} GB"

    def _print_banner(self, start_ep: int, total_steps: int):
        """Print startup banner with configuration and dataset status."""
        disk = self._get_disk_usage(self.dataset_path) if os.path.exists(self.dataset_path) else "0 B"
        bias = self.task_cfg.get("start_bias", [0, 0, 0])
        rand_bias = self.task_cfg.get("random_bias", {})

        lines = [
            "",
            "┌─────────────────────────────────────────────────┐",
            "│            XArm Toolkit — Data Collector         │",
            "├─────────────────────────────────────────────────┤",
            f"│  Dataset    : {self.dataset_path}",
            f"│  Cam mode   : {self.cam_mode}    Image: {self.image_w}×{self.image_h}",
            f"│  Force      : {'ON' if self.env.use_force else 'OFF'}    Video: {'ON' if self.save_video else 'OFF'}",
            f"│  Bias       : {bias}",
        ]
        if rand_bias:
            lines.append(f"│  Rand bias  : {rand_bias}")
        if self._gripper_always_closed:
            lines.append("│  Gripper    : always closed")

        lines.append("├─────────────────────────────────────────────────┤")

        if start_ep > 0:
            lines.append(f"│  续采模式   : 已有 {start_ep} episodes, {total_steps:,} steps")
            lines.append(f"│  磁盘占用   : {disk}")
        else:
            lines.append("│  新建数据集")

        lines.append(f"│  本次采集   : {self.num_episodes} episodes")
        lines.append("├─────────────────────────────────────────────────┤")
        lines.append("│  Space: 开始录制 | Enter: 结束 | Ctrl+C: 退出   │")
        lines.append("└─────────────────────────────────────────────────┘")
        lines.append("")

        # Use \r\n for raw terminal mode
        print("\r\n".join(lines) + "\r\n")

    def _print_episode_summary(
        self,
        current_ep: int,
        episodes_saved: int,
        data: zarr.Group,
        stats: EpisodeStats,
    ):
        """Print summary box after a successful episode save."""
        total_episodes = current_ep + 1
        total_steps = len(data["episode"]) if "episode" in data else 0
        disk = self._get_disk_usage(self.dataset_path)
        remain = self.num_episodes - episodes_saved

        lines = [
            "",
            f"  ┌─ Episode {current_ep} ────────────────────────",
            f"  │ Steps   : {stats.steps} ({stats.duration:.1f}s, avg {stats.fps:.1f} FPS)",
            f"  │ Dataset : {total_episodes} episodes, {total_steps:,} total steps",
            f"  │ Disk    : {disk}",
            f"  │ Remain  : {remain} episodes",
            f"  └─────────────────────────────────────",
            "",
        ]
        print("\r\n".join(lines) + "\r\n")

    def _print_final_summary(self, episodes_saved: int, data: zarr.Group, start_ep: int):
        """Print final summary on exit."""
        total_episodes = start_ep + episodes_saved
        total_steps = len(data["episode"]) if "episode" in data else 0
        disk = self._get_disk_usage(self.dataset_path) if os.path.exists(self.dataset_path) else "0 B"

        lines = [
            "",
            "╔═════════════════════════════════════════════════╗",
            "║              采集结束 — 最终总结                 ║",
            "╠═════════════════════════════════════════════════╣",
            f"║  本次新增   : {episodes_saved} episodes",
            f"║  数据集总量 : {total_episodes} episodes, {total_steps:,} steps",
            f"║  磁盘占用   : {disk}",
            f"║  路径       : {self.dataset_path}",
            "╚═════════════════════════════════════════════════╝",
            "",
        ]
        print("\r\n".join(lines) + "\r\n")

    def _reset_for_next_episode(self):
        """Reset arm and cameras, move to task start position."""
        obs = self.env.reset(close_gripper=True)
        self.cam_arm.step()
        self.cam_fix.step()
        time.sleep(0.5)

        bias = self._get_start_bias()
        obs = self.env.step(bias, gripper_action=0, speed=100)
        self.cam_arm.step()
        self.cam_fix.step()
        time.sleep(1)
        return obs

    def _confirm_episode_and_save(
        self,
        kb: _KeyboardListener,
        current_ep: int,
        steps: int,
        buffer: dict[str, list],
        data: zarr.Group,
    ) -> bool:
        """Ask whether to save a completed episode, then write it to Zarr."""
        print(
            f"\r\n  是否保存 Episode {current_ep}？"
            " [y] 保存 | [n] 丢弃 | Ctrl+C 退出\r\n"
        )

        while True:
            key = kb.get_key()

            if key == "y":
                logger.info("Episode %d: saving %d steps...", current_ep, steps)
                for name, values in buffer.items():
                    data[name].append(np.concatenate(values, axis=0))
                logger.info("Episode %d saved (%d steps).", current_ep, steps)
                return True

            if key == "n":
                logger.info(
                    "Episode %d discarded by user (%d steps). "
                    "Recorded video is retained.",
                    current_ep,
                    steps,
                )
                return False

            if key == "\x03":
                raise KeyboardInterrupt

            time.sleep(0.05)

    # ------------------------------------------------------------------
    # Main collection loop
    # ------------------------------------------------------------------

    def run(self):
        """Run the collection session (blocking)."""
        # Open dataset
        image_shape = (3, self.image_h, self.image_w)
        data, meta, start_ep = _open_or_create_zarr(
            self.dataset_path, image_shape=image_shape, cam_mode=self.cam_mode,
        )

        # Compute existing total steps for banner
        total_steps = len(data["episode"]) if "episode" in data and len(data["episode"]) > 0 else 0

        # Print startup banner
        self._print_banner(start_ep, total_steps)

        # Initial reset
        obs = self.env.reset(close_gripper=True)
        self.cam_arm.step()
        self.cam_fix.step()
        time.sleep(0.5)

        # Move to task start position
        bias = self._get_start_bias()
        obs = self.env.step(bias, gripper_action=0, speed=100)
        self.cam_arm.step()
        self.cam_fix.step()
        logger.info("Initial position: %s", obs["goal_pos"][:3])

        logger.info("Warming up... (%.1fs)", self.warmup_time)
        time.sleep(self.warmup_time)

        kb = _KeyboardListener()
        episodes_saved = 0

        try:
            kb.start()

            for ep_idx in range(self.num_episodes):
                current_ep = start_ep + episodes_saved

                try:
                    stats = self._run_episode(current_ep, ep_idx, data, kb)

                    if stats is None:
                        # Empty or explicitly discarded episode
                        print(f"\r\n  Episode {current_ep}: 未保存\r\n")
                    else:
                        # Successfully saved — update metadata & print summary
                        episodes_saved += 1
                        _compute_episode_ends(data, meta)
                        self._print_episode_summary(current_ep, episodes_saved, data, stats)

                except KeyboardInterrupt:
                    # Ctrl+C — break out to finally block
                    print("\r\n\r\n  ⚠ Ctrl+C 检测到，正在退出...\r\n")
                    break

                except Exception:
                    # Episode failed — print error, don't pollute zarr
                    err_msg = traceback.format_exc()
                    print(
                        f"\r\n  ✗ Episode {current_ep} 出错，数据已丢弃:\r\n"
                        f"  {err_msg}\r\n"
                    )

                    # Try auto-reset
                    try:
                        print("  尝试自动复位机械臂...\r\n")
                        self._reset_for_next_episode()
                        print("  ✓ 复位成功\r\n")
                    except Exception:
                        reset_err = traceback.format_exc()
                        print(
                            f"  ✗ 自动复位失败，请手动处理后按 Space 继续:\r\n"
                            f"  {reset_err}\r\n"
                        )

                    # Wait for user to press Space to continue or Ctrl+C to quit
                    print("  按 Space 继续下一个 episode | Ctrl+C 退出\r\n")
                    while True:
                        key = kb.get_key()
                        if key == " ":
                            break
                        elif key == "\x03":
                            print("\r\n  ⚠ Ctrl+C 检测到，正在退出...\r\n")
                            # Use a flag to break outer loop
                            raise KeyboardInterrupt
                        time.sleep(0.05)
                    continue

                # Reset for next episode (only if there are more to collect)
                if ep_idx < self.num_episodes - 1:
                    try:
                        logger.info("Resetting for next episode...")
                        self._reset_for_next_episode()
                    except Exception:
                        err_msg = traceback.format_exc()
                        print(
                            f"\r\n  ✗ 复位出错:\r\n  {err_msg}\r\n"
                            f"  按 Space 继续 | Ctrl+C 退出\r\n"
                        )
                        while True:
                            key = kb.get_key()
                            if key == " ":
                                break
                            elif key == "\x03":
                                print("\r\n  ⚠ Ctrl+C 检测到，正在退出...\r\n")
                                raise KeyboardInterrupt
                            time.sleep(0.05)

        except KeyboardInterrupt:
            # Already handled above, just fall through to finally
            pass

        finally:
            kb.stop()
            self._print_final_summary(episodes_saved, data, start_ep)

    # ------------------------------------------------------------------

    def _run_episode(
        self,
        current_ep: int,
        ep_idx: int,
        data: zarr.Group,
        kb: _KeyboardListener,
    ) -> EpisodeStats | None:
        """Record one episode.

        Returns EpisodeStats on success, None if the episode is not saved.
        Raises KeyboardInterrupt on Ctrl+C, other exceptions on hw error.
        """
        print(
            f"\r\n=== Episode {current_ep} ({ep_idx + 1}/{self.num_episodes}) ===\r\n"
            f"  Space: start recording | Enter: end recording | Ctrl+C: abort\r\n"
        )

        # --- Wait for start (space key) ---
        # During wait: arm stays still, agent reads SpaceMouse to update gripper
        while True:
            _action, goal_gripper = self.agent.act()

            obs = self.env.step([0, 0, 0, 0, 0, 0], gripper_action=goal_gripper, speed=100)
            self.cam_arm.step()
            self.cam_fix.step()

            key = kb.get_key()
            if key == " ":
                break
            elif key == "\x03":
                raise KeyboardInterrupt
            time.sleep(0.1)

        # --- Zero force sensor ---
        if self.env.use_force:
            logger.info("Zeroing force sensor...")
            self.env.reset_force_sensor_zero()
            time.sleep(0.2)

        print(f"\r\n  ● 录制 episode {current_ep}... (Enter 结束)\r\n")

        # --- Video writers (optional) ---
        vw_arm = None
        vw_fix = None
        if self.save_video:
            # Save videos to a subfolder: datasets/demo_videos/ep0_arm.mp4
            base = os.path.splitext(self.dataset_path)[0]
            video_dir = f"{base}_videos"
            os.makedirs(video_dir, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            size = (self.image_w, self.image_h)
            vw_arm = cv2.VideoWriter(f"{video_dir}/ep{current_ep}_arm.mp4", fourcc, self.video_fps, size)
            vw_fix = cv2.VideoWriter(f"{video_dir}/ep{current_ep}_fix.mp4", fourcc, self.video_fps, size)
            logger.info("Saving video to: %s/ep%d_{arm,fix}.mp4", video_dir, current_ep)

        # --- Record ---
        buffer: dict[str, list] = {
            "rgb_arm": [], "rgb_fix": [],
            "pos": [], "force": [], "action": [],
            "gripper_state": [], "gripper_action": [],
            "episode": [],
        }
        if self._save_depth:
            buffer["depth_arm"] = []
            buffer["depth_fix"] = []

        goal_gripper = 0  # will be overwritten by agent.act() each step
        steps = 0
        t_start = time.time()

        try:
            while True:
                key = kb.get_key()
                if key == "\r":
                    print(f"\r\n  Episode {current_ep} ended.\r\n")
                    break
                elif key == "\x03":
                    raise KeyboardInterrupt

                # SpaceMouse action
                action, goal_gripper = self.agent.act()

                # Camera observations (before step, matches reference)
                obs_arm = self.cam_arm.step()
                obs_fix = self.cam_fix.step()

                # Extract RGB
                if "rgbd" in obs_arm:
                    rgb_arm = np.asarray(obs_arm["rgbd"].color)
                else:
                    rgb_arm = np.asarray(obs_arm["rgb"])
                if "rgbd" in obs_fix:
                    rgb_fix = np.asarray(obs_fix["rgbd"].color)
                else:
                    rgb_fix = np.asarray(obs_fix["rgb"])

                rgb_arm = cv2.resize(rgb_arm, (self.image_w, self.image_h))
                rgb_fix = cv2.resize(rgb_fix, (self.image_w, self.image_h))

                # Write video frames (RGB → BGR for OpenCV)
                if vw_arm is not None:
                    vw_arm.write(cv2.cvtColor(rgb_arm, cv2.COLOR_RGB2BGR))
                if vw_fix is not None:
                    vw_fix.write(cv2.cvtColor(rgb_fix, cv2.COLOR_RGB2BGR))

                # Extract depth (if saving)
                if self._save_depth:
                    if "rgbd" in obs_arm:
                        depth_arm = np.asarray(obs_arm["rgbd"].depth).squeeze()  # (H, W)
                    else:
                        depth_arm = np.zeros((rgb_arm.shape[0], rgb_arm.shape[1]), dtype=np.uint16)
                    if "rgbd" in obs_fix:
                        depth_fix = np.asarray(obs_fix["rgbd"].depth).squeeze()
                    else:
                        depth_fix = np.zeros((rgb_fix.shape[0], rgb_fix.shape[1]), dtype=np.uint16)

                    depth_arm = cv2.resize(depth_arm, (self.image_w, self.image_h),
                                           interpolation=cv2.INTER_NEAREST)
                    depth_fix = cv2.resize(depth_fix, (self.image_w, self.image_h),
                                           interpolation=cv2.INTER_NEAREST)

                # Robot step
                obs = self.env.step(action, gripper_action=goal_gripper, speed=100)

                # Gripper state/action
                if self._gripper_always_closed:
                    gs = 0.0
                    ga = 0.0
                else:
                    gs = 0.0 if obs["gripper_position"] <= 420 else 1.0
                    ga = 1.0 if goal_gripper == 840 else 0.0

                # Force (fallback to zeros if no force sensor)
                force = obs["ext_force"] if obs["ext_force"] is not None else np.zeros(6)

                # Append to buffer
                buffer["rgb_arm"].append(rgb_arm.transpose(2, 0, 1)[None])  # (1, 3, H, W)
                buffer["rgb_fix"].append(rgb_fix.transpose(2, 0, 1)[None])
                if self._save_depth:
                    buffer["depth_arm"].append(depth_arm[None, None])  # (1, 1, H, W)
                    buffer["depth_fix"].append(depth_fix[None, None])
                buffer["pos"].append(obs["goal_pos"].astype(np.float32)[None])
                buffer["force"].append(np.asarray(force, dtype=np.float32)[None])
                buffer["action"].append(np.asarray(action, dtype=np.float32)[None])
                buffer["gripper_state"].append(np.array([[gs]], dtype=np.float32))
                buffer["gripper_action"].append(np.array([[ga]], dtype=np.float32))
                buffer["episode"].append(np.array([current_ep], dtype=np.uint16))

                steps += 1

                # Print stats every 50 steps
                if steps % 50 == 0:
                    elapsed = time.time() - t_start
                    fps = steps / elapsed if elapsed > 0 else 0
                    fz = force[2] if obs["ext_force"] is not None else 0
                    print(
                        f"\r\n  Step {steps:5d} | FPS {fps:.1f} | "
                        f"force_z {fz:.1f} | gripper {obs['gripper_position']:.0f}\r\n"
                    )

        finally:
            # Release video writers no matter what
            if vw_arm is not None:
                vw_arm.release()
            if vw_fix is not None:
                vw_fix.release()

        # --- Confirm and save buffer ---
        if steps == 0:
            logger.warning("Episode %d: 0 steps, skipping save.", current_ep)
            return None

        duration = time.time() - t_start
        fps = steps / duration if duration > 0 else 0

        saved = self._confirm_episode_and_save(
            kb=kb,
            current_ep=current_ep,
            steps=steps,
            buffer=buffer,
            data=data,
        )

        if not saved:
            return None

        return EpisodeStats(steps=steps, duration=duration, fps=fps)
