"""xarm_toolkit.env — environment wrappers for robot and camera hardware."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xarm_toolkit.env.coinft_env import CoinFTEnv
    from xarm_toolkit.env.realsense_env import RealsenseEnv
    from xarm_toolkit.env.xarm_env import XArmEnv

__all__ = ["CoinFTEnv", "RealsenseEnv", "XArmEnv"]

# Lazy imports — avoid pulling in heavy / hardware-only dependencies at
# package-level so that e.g. ``from xarm_toolkit.env import RealsenseEnv``
# does not require pytransform3d / xarm SDK to be installed.

_LAZY_IMPORTS: dict[str, str] = {
    "CoinFTEnv": "xarm_toolkit.env.coinft_env",
    "RealsenseEnv": "xarm_toolkit.env.realsense_env",
    "XArmEnv": "xarm_toolkit.env.xarm_env",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
