"""Gym/Gymnasium space conversion helpers for the SB3 BDP runner."""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces


def as_gymnasium_box(box_space: Any, dtype: np.dtype = np.float32) -> spaces.Box:
    low = np.asarray(box_space.low, dtype=dtype)
    high = np.asarray(box_space.high, dtype=dtype)
    return spaces.Box(low=low, high=high, shape=box_space.shape, dtype=dtype)


def convert_to_gymnasium_space(space: Any) -> spaces.Space:
    """
    Convert the small subset of legacy Gym spaces we need into Gymnasium spaces.

    This keeps ``--env_source=gym`` usable even when the PyTorch environment does
    not have ``shimmy`` installed.  The generic BDP wrapper currently supports
    Box observations and Discrete actions/observations, so only those spaces are
    converted here.
    """

    if isinstance(space, spaces.Box):
        return spaces.Box(
            low=np.asarray(space.low, dtype=np.float32),
            high=np.asarray(space.high, dtype=np.float32),
            shape=space.shape,
            dtype=np.float32,
        )
    if isinstance(space, spaces.Discrete):
        return spaces.Discrete(int(space.n), start=int(getattr(space, "start", 0)))
    if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape"):
        return spaces.Box(
            low=np.asarray(space.low, dtype=np.float32),
            high=np.asarray(space.high, dtype=np.float32),
            shape=space.shape,
            dtype=np.float32,
        )
    if hasattr(space, "n"):
        return spaces.Discrete(int(space.n), start=int(getattr(space, "start", 0)))
    raise TypeError(f"Unsupported space for Gymnasium conversion: {space}")
