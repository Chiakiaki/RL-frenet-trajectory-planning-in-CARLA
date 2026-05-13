"""External candidate samplers used by the SB3 BDP wrappers."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from gymnasium import spaces


class ExternalCandidateSampler:
    """
    Minimal external-sampler interface used by the generic candidate wrapper.

    The CARLA environment already owns its sampler because candidate trajectories
    depend on the Frenet planner and the current traffic scene.  For ordinary
    Gym tasks, it is cleaner to keep the sampler outside the environment, just
    like the original BDPL idea: the sampler proposes candidate action features,
    the policy scores them, and the wrapper sends only the selected action label
    to the underlying env.
    """

    def sample(self, raw_obs: np.ndarray) -> Tuple[np.ndarray, int, List[Any]]:
        """
        Return candidate action features for one environment state.

        Input:
            raw_obs: (obs_shape) or (Ds,)

        Returns:
            candidates:   (N, Da), float32 action feature matrix
            n_candidates: scalar N
            labels:       length-N list; labels[i] is the raw env action for
                          candidate row i
        """
        raise NotImplementedError


class DiscreteOneHotExternalSampler(ExternalCandidateSampler):
    """
    External sampler for normal Discrete(n) environments.

    It enumerates every possible action and represents each action label as a
    one-hot vector.  For CartPole-v1, for example:

        labels     = [0, 1]
        candidates = [[1, 0],
                      [0, 1]]

    For a 3-action environment the vectors are [[1,0,0], [0,1,0], [0,0,1]].
    This gives the BDP policy the same kind of "candidate action feature" input
    it receives from CARLA trajectories, but with a tiny deterministic sampler.
    """

    def __init__(self, action_space: spaces.Discrete):
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("DiscreteOneHotExternalSampler requires a Discrete action space")
        self.n = int(action_space.n)
        self.start = int(getattr(action_space, "start", 0))
        self.labels = [self.start + action_idx for action_idx in range(self.n)]
        self.candidates = np.eye(self.n, dtype=np.float32)

    def sample(self, raw_obs: np.ndarray) -> Tuple[np.ndarray, int, List[int]]:
        del raw_obs
        # candidates: (N, Da) = (n_actions, n_actions), one row per action.
        # labels:     length N, maps candidate index -> raw Discrete label.
        return self.candidates.copy(), self.n, self.labels
