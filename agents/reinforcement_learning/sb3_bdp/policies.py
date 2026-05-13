"""PyTorch candidate-action policy for SB3 BDP training."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn


def make_mlp(input_dim: int, layer_sizes: Iterable[int], activation_fn: Type[nn.Module], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    last_dim = int(input_dim)
    for layer_size in layer_sizes:
        layers.append(nn.Linear(last_dim, int(layer_size)))
        layers.append(activation_fn())
        last_dim = int(layer_size)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


def init_mlp(module: nn.Module, output_gain: float = 1.0) -> None:
    linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    for idx, layer in enumerate(linear_layers):
        gain = output_gain if idx == len(linear_layers) - 1 else np.sqrt(2)
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0.0)


class BDPBoltzmannPolicy(ActorCriticPolicy):
    """
    Actor-critic policy for candidate-action BDP observations.

    For each observation, the actor scores every candidate trajectory with
    ``goodness_net([state, candidate])``.  The resulting score vector is used as
    categorical logits, which is the PyTorch/SB3 analogue of the TensorFlow
    Boltzmann distribution in the original code.

    No ``grouping_mn`` is needed here.  In the old code, grouping was necessary
    because all pairs from multiple states were concatenated into one long
    tensor and the algorithm had to reconstruct the per-state candidate sets.
    In this policy the per-state candidate set remains a separate row in the
    batch:

        logits.shape == (batch_size, max_candidates)

    PyTorch's Categorical distribution softmaxes along the candidate dimension
    for each row independently.  The candidate mask handles padded rows.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        candidate_net_arch: Optional[List[int]] = None,
        value_net_arch: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        self.candidate_net_arch = candidate_net_arch
        self.value_net_arch = value_net_arch
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _resolve_architectures(self) -> Tuple[List[int], List[int]]:
        if self.candidate_net_arch is not None and self.value_net_arch is not None:
            return list(self.candidate_net_arch), list(self.value_net_arch)

        net_arch = self.net_arch
        if isinstance(net_arch, dict):
            pi_layers = net_arch.get("pi", [64, 64])
            vf_layers = net_arch.get("vf", [64, 64])
        elif isinstance(net_arch, list):
            pi_layers = net_arch
            vf_layers = net_arch
        else:
            pi_layers = [64, 64]
            vf_layers = [64, 64]

        if self.candidate_net_arch is not None:
            pi_layers = self.candidate_net_arch
        if self.value_net_arch is not None:
            vf_layers = self.value_net_arch
        return list(pi_layers), list(vf_layers)

    def _build(self, lr_schedule: Schedule) -> None:
        if not isinstance(self.observation_space, spaces.Dict):
            raise TypeError("BDPBoltzmannPolicy requires a Dict observation space")
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("BDPBoltzmannPolicy requires a Discrete action space")

        raw_obs_space = self.observation_space.spaces["obs"]
        candidate_space = self.observation_space.spaces["candidates"]
        # state_dim Ds: product of the raw observation shape.
        #   CartPole: obs shape (4,)      -> Ds=4
        #   CARLA lidar: obs shape (362,) -> Ds=362
        #   CARLA seq: obs shape (5, 9)   -> Ds=45
        self.state_dim = int(np.prod(raw_obs_space.shape))
        # max_candidates K: fixed number of rows stored in obs["candidates"].
        # candidate_dim Da: flattened feature length of one candidate row.
        #   Discrete one-hot env with n actions: Da=n_actions
        #   CARLA BDP trajectory feature: Da=T_ac_candidates+1
        self.max_candidates = int(candidate_space.shape[0])
        self.candidate_dim = int(np.prod(candidate_space.shape[1:]))

        pi_layers, vf_layers = self._resolve_architectures()
        # goodness_net input:  (B*K, Ds + Da)
        # goodness_net output: (B*K, 1), then reshaped to (B, K) logits.
        self.goodness_net = make_mlp(
            self.state_dim + self.candidate_dim,
            pi_layers,
            self.activation_fn,
            output_dim=1,
        )
        # value_net input:  (B, Ds)
        # value_net output: (B, 1)
        self.value_net = make_mlp(self.state_dim, vf_layers, self.activation_fn, output_dim=1)

        if self.ortho_init:
            init_mlp(self.goodness_net, output_gain=0.01)
            init_mlp(self.value_net, output_gain=1.0)

        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _state_and_candidates(self, obs: PyTorchObs) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        assert isinstance(obs, dict), "BDP policy expects dict observations"
        # SB3 passes a batched dict:
        #   raw_obs:    (B, *obs_shape)
        #   candidates: (B, K, *candidate_shape)
        #   mask:       (B, K)
        raw_obs = obs["obs"].float()
        candidates = obs["candidates"].float()
        mask = obs["candidate_mask"].float()

        # Flatten only the feature dimensions, not the batch:
        #   state:      (B, Ds)
        #   candidates: (B, K, Da)
        state = raw_obs.reshape(raw_obs.shape[0], -1)
        candidates = candidates.reshape(candidates.shape[0], candidates.shape[1], -1)
        return state, candidates, mask

    def _candidate_logits(self, obs: PyTorchObs) -> th.Tensor:
        state, candidates, mask = self._state_and_candidates(obs)
        batch_size, n_candidates, _ = candidates.shape

        # Build all (state, candidate) pairs for the current mini-batch.
        # Example:
        #   state:      (B, Ds)
        #   candidates: (B, K, Da)
        #   input:      (B*K, Ds + Da)
        #
        # This is the same goodness function as the old BDP policy conceptually,
        # but the batch dimension keeps candidate groups separated, so no
        # grouping matrix is required.
        repeated_state = state[:, None, :].expand(-1, n_candidates, -1)
        # repeated_state: (B, K, Ds), candidates: (B, K, Da)
        # scorer_input:   (B, K, Ds + Da), one row per (state, candidate)
        scorer_input = th.cat([repeated_state, candidates], dim=-1)
        # Flatten the first two axes for a normal MLP:
        #   (B, K, Ds + Da) -> (B*K, Ds + Da)
        # Then reshape:
        #   (B*K, 1) -> (B, K)
        logits = self.goodness_net(scorer_input.reshape(batch_size * n_candidates, -1))
        logits = logits.reshape(batch_size, n_candidates)

        # Padding slots get zero probability after the categorical softmax.
        # We use a large finite negative value instead of -inf because it is
        # friendlier to PPO/TRPO math, saving/loading, and debug prints.
        return logits.masked_fill(mask < 0.5, -1e9)

    def _values(self, obs: PyTorchObs) -> th.Tensor:
        state, _, _ = self._state_and_candidates(obs)
        # value_net(state): (B, 1), same convention expected by SB3 on-policy
        # algorithms for critic values.
        return self.value_net(state)

    def _distribution_from_obs(self, obs: PyTorchObs) -> Distribution:
        # logits: (B, K).  Each row is a Boltzmann/Categorical distribution
        # over that observation's candidate set.
        logits = self._candidate_logits(obs)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        distribution = self._distribution_from_obs(obs)
        # actions:  (B,), candidate row index sampled from logits
        # values:   (B, 1), critic estimate V(s)
        # log_prob: (B,), log pi(candidate_idx | state, candidate_set)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._values(obs)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def _predict(self, observation: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        return self._distribution_from_obs(observation).get_actions(deterministic=deterministic)

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        return self._distribution_from_obs(obs)

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        distribution = self._distribution_from_obs(obs)
        # rollout_buffer stores Discrete actions as shape (B, 1) float tensors.
        # Categorical.log_prob expects int64 shape (B,), so flatten/cast here.
        action_idx = actions.long().flatten()
        # log_prob: (B,), entropy: (B,), values: (B, 1)
        log_prob = distribution.log_prob(action_idx)
        entropy = distribution.entropy()
        return self._values(obs), log_prob, entropy

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        return self._values(obs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            {
                "candidate_net_arch": self.candidate_net_arch,
                "value_net_arch": self.value_net_arch,
            }
        )
        return data
