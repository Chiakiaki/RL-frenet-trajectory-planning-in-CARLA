"""Training callbacks for the SB3 BDP runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class BDPCheckpointCallback(BaseCallback):
    def __init__(self, model_dir: Path, save_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.model_dir = Path(model_dir)
        self.save_freq = int(save_freq)
        self.best_mean_reward = -np.inf
        self._last_step_save = 0

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.num_timesteps - self._last_step_save >= self.save_freq:
            self.model.save(str(self.model_dir / f"step_{self.num_timesteps}"))
            self._last_step_save = self.num_timesteps

        if self.model.ep_info_buffer:
            mean_reward = float(np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]))
            if mean_reward >= self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(str(self.model_dir / f"best_{self.num_timesteps}"))
        return True
