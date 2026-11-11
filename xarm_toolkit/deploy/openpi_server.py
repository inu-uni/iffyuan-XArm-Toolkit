"""OpenPI Pi0.5 policy wrapper for XArm6.

Wraps an openpi trained policy to implement :class:`PolicyProtocol` so it can
be served by :class:`VLAServer`.

Config 加载方式：从外部 Python 文件动态导入 ``build_config()`` 函数构建配置，
而非使用 openpi 内置的 config name，以支持自定义训练配置。

Observation key mapping:
    - ``observation/image``       — 固定相机 (base) RGB
    - ``observation/wrist_image`` — 臂上相机 (wrist) RGB
    - ``observation/state``       — goal_pos(6) + gripper(1) = 7-dim float32
    - ``prompt``                  — 语言指令

Usage::

    from xarm_toolkit.deploy.openpi_server import Pi05XArmPolicy
    from xarm_toolkit.deploy.server import VLAServer

    policy = Pi05XArmPolicy(
        config_file="/path/to/pi05_xarm_v2_stage1.py",
        ckpt_dir="/path/to/checkpoint",
    )
    server = VLAServer(policy=policy, port=10093)
    server.run()
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

from xarm_toolkit.utils.logger import get_logger

logger = get_logger("xarm_toolkit.deploy.openpi_server")


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config_from_file(config_file: str | pathlib.Path):
    """从外部 Python 文件动态加载 openpi config.

    该文件必须定义一个 ``build_config()`` 函数，返回 openpi TrainConfig。

    Parameters
    ----------
    config_file : str | Path
        Config Python 文件路径。
    """
    config_file = pathlib.Path(config_file).expanduser().resolve()

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    module_name = f"user_config_{config_file.stem}"
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load config module from: {config_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "build_config"):
        raise AttributeError(
            f"{config_file} must define a function named build_config()"
        )

    config = module.build_config()
    logger.info("Loaded config from: %s", config_file)
    return config


# ---------------------------------------------------------------------------
# Pi0.5 XArm Policy
# ---------------------------------------------------------------------------

class Pi05XArmPolicy:
    """Pi0.5 policy wrapper for XArm6, using openpi.

    Parameters
    ----------
    config_file : str
        外部 config Python 文件路径（必须包含 ``build_config()``）。
    ckpt_dir : str
        本地 checkpoint 目录。
    """

    def __init__(self, config_file: str, ckpt_dir: str):
        self.config_file = config_file
        self.ckpt_dir = ckpt_dir
        self._load_model()

    def _load_model(self):
        from openpi.policies import policy_config

        config = load_config_from_file(self.config_file)

        ckpt_path = pathlib.Path(self.ckpt_dir).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")
        logger.info("Using local checkpoint: %s", ckpt_path)

        logger.info("Creating trained policy ...")
        self.policy = policy_config.create_trained_policy(config, ckpt_path)
        logger.info("Pi0.5 XArm policy ready.")

    def predict_action(self, **kwargs) -> dict:
        """Run inference and return predicted actions.

        VLAClient 发送的 kwargs:
            images : list[np.ndarray]  — [arm_rgb(wrist), fix_rgb(base)], HWC uint8
            instruction : str          — 语言指令
            state : np.ndarray         — goal_pos(6) + gripper(1) = 7-dim

        Returns
        -------
        dict
            ``{"actions": np.ndarray}`` — shape ``(chunk_size, action_dim)``
        """
        images = kwargs.get("images")
        instruction = kwargs.get("instruction")
        state = kwargs.get("state")

        logger.info(
            "Inference: %d images, instruction=%r, state_shape=%s",
            len(images),
            instruction,
            state.shape if hasattr(state, "shape") else "N/A",
        )

        # -----------------------------------------------------------------
        # 组装 openpi observation dict
        #
        #   observation/image       — 固定相机 (base) RGB, HWC uint8
        #   observation/wrist_image — 臂上相机 (wrist) RGB, HWC uint8
        #   observation/state       — [x, y, z, roll, pitch, yaw, gripper] float32
        #   prompt                  — 语言指令 str
        #
        # Client 发送: images[0]=arm(wrist), images[1]=fix(base)
        # -----------------------------------------------------------------
        example = {"prompt": instruction}

        # 图像映射: images[0] = arm → wrist, images[1] = fix → base
        if len(images) >= 1:
            example["observation/wrist_image"] = _ensure_hwc(images[0])
        if len(images) >= 2:
            example["observation/image"] = _ensure_hwc(images[1])

        # 机器人状态: goal_pos(6) + gripper(1)
        if state is not None:
            example["observation/state"] = np.asarray(state, dtype=np.float32)

        # -----------------------------------------------------------------
        # 推理
        # -----------------------------------------------------------------
        result = self.policy.infer(example)
        actions = result["actions"]  # (chunk_size, action_dim)

        logger.info("Inference done, actions shape: %s", actions.shape)
        return {"actions": np.asarray(actions, dtype=np.float32)}

    def reset(self):
        """Clear the OpenPI policy's inference state for a new run."""
        self.policy.reset()
        logger.info("Policy state reset.")


def _ensure_hwc(img: np.ndarray) -> np.ndarray:
    """Ensure image is HWC uint8 (openpi 内部自行 resize)."""
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return img.astype(np.uint8)
