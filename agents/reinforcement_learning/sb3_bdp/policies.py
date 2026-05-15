"""PyTorch candidate-action policy for SB3 BDP training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor, NatureCNN
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule
from torch import nn


# Architecture guide for maintainers, if with config:
#   policy_layers: [64, 64]
#   value_layers:  [64, 64]
# Then, architectures are as follows
#
# Built-in SB3 MLP policy:
#   Actor:
#     raw state -> FlattenExtractor -> state_features
#     state_features -> MlpExtractor actor trunk:
#       Linear(state_dim -> 64) -> activation
#       Linear(64 -> 64)        -> activation
#     latent_pi -> action_net -> raw-action logits
#   Critic:
#     raw state -> FlattenExtractor -> state_features
#     state_features -> MlpExtractor critic trunk:
#       Linear(state_dim -> 64) -> activation
#       Linear(64 -> 64)        -> activation
#     latent_vf -> value_net -> V(s)
#
# BDP MLP policy:
#   Actor:
#     raw state -> FlattenExtractor -> state_features
#     [state_features, candidate] -> MlpExtractor actor trunk:
#       Linear(state_dim + candidate_dim -> 64) -> activation
#       Linear(64 -> 64)                        -> activation
#     latent_pi -> goodness_net -> candidate logit
#   Critic:
#     raw state -> FlattenExtractor -> state_features
#     state_features -> value_mlp_extractor critic trunk:
#       Linear(state_dim -> 64) -> activation
#       Linear(64 -> 64)        -> activation
#     latent_vf -> value_net -> V(s)
#
# Built-in SB3 CNN policy:
#   Actor:
#     image state -> NatureCNN -> state_features, usually 512-dim
#     state_features -> MlpExtractor actor trunk:
#       Linear(512 -> 64) -> activation
#       Linear(64 -> 64)  -> activation
#     latent_pi -> action_net -> raw-action logits
#   Critic:
#     image state -> NatureCNN -> state_features, usually 512-dim
#     state_features -> MlpExtractor critic trunk:
#       Linear(512 -> 64) -> activation
#       Linear(64 -> 64)  -> activation
#     latent_vf -> value_net -> V(s)
#
# BDP CNN policy:
#   Actor:
#     image state -> NatureCNN -> state_features, usually 512-dim
#     [state_features, candidate] -> MlpExtractor actor trunk:
#       Linear(512 + candidate_dim -> 64) -> activation
#       Linear(64 -> 64)                  -> activation
#     latent_pi -> goodness_net -> candidate logit
#   Critic:
#     image state -> NatureCNN -> state_features, usually 512-dim
#     state_features -> value_mlp_extractor critic trunk:
#       Linear(512 -> 64) -> activation
#       Linear(64 -> 64)  -> activation
#     latent_vf -> value_net -> V(s)
#
# The critic mid-layer structure matches SB3 for the same feature extractor and
# value_layers.  The actor hidden layer sizes match too, but BDP concatenates a
# candidate vector to the state features, so the first actor input dimension is
# different by design.


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
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError("BDPBoltzmannPolicy requires a Dict observation space")
        self.raw_observation_space = observation_space.spaces["obs"]
        self.candidate_net_arch = candidate_net_arch
        self.value_net_arch = value_net_arch
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _resolve_architectures(self) -> Tuple[List[int], List[int]]:
        if self.candidate_net_arch is not None and self.value_net_arch is not None:
            return list(self.candidate_net_arch), list(self.value_net_arch)

        net_arch = self.net_arch
        if isinstance(net_arch, dict):
            pi_layers = net_arch.get("pi", [])
            vf_layers = net_arch.get("vf", [])
        elif isinstance(net_arch, list):
            pi_layers = net_arch
            vf_layers = net_arch
        else:
            pi_layers = []
            vf_layers = []

        if self.candidate_net_arch is not None:
            pi_layers = self.candidate_net_arch
        if self.value_net_arch is not None:
            vf_layers = self.value_net_arch
        return list(pi_layers), list(vf_layers)

    def _build_mlp_extractor(self) -> None:
        """
        Build the hidden actor/critic trunks with SB3's ``MlpExtractor``.

        The actor trunk receives a flattened ``[state, candidate]`` pair and
        produces one latent vector per pair.  The critic trunk receives only the
        flattened state.  This mirrors ``ActorCriticPolicy`` more closely than a
        monolithic ``goodness_net`` while keeping the BDP action semantics:

            actor:  (B*K, Ds + Da) -> latent_pi -> scalar candidate logit
            critic: (B,   Ds)      -> latent_vf -> scalar V(s)
        """

        pi_layers, vf_layers = self._resolve_architectures()
        self.mlp_extractor = MlpExtractor(
            feature_dim=self.state_features_dim + self.candidate_dim,
            net_arch=dict(pi=pi_layers, vf=[]),
            activation_fn=self.activation_fn,
            device=self.device,
        )
        self.value_mlp_extractor = MlpExtractor(
            feature_dim=self.state_features_dim,
            net_arch=dict(pi=[], vf=vf_layers),
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _setup_state_features(self) -> None:
        """Use SB3's feature extractor on obs["obs"] as the BDP state feature path."""

        self.state_dim = int(np.prod(self.raw_observation_space.shape))
        self.state_features_dim = self.features_extractor.features_dim

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        return self.features_extractor_class(self.raw_observation_space, **self.features_extractor_kwargs)

    def _extract_state_features(self, raw_obs: th.Tensor) -> th.Tensor:
        preprocessed_obs = preprocess_obs(
            raw_obs,
            self.raw_observation_space,
            normalize_images=self.normalize_images,
        )
        return self.features_extractor(preprocessed_obs)

    def _build(self, lr_schedule: Schedule) -> None:
        if not isinstance(self.observation_space, spaces.Dict):
            raise TypeError("BDPBoltzmannPolicy requires a Dict observation space")
        if not isinstance(self.action_space, spaces.Discrete):
            raise TypeError("BDPBoltzmannPolicy requires a Discrete action space")

        self.raw_observation_space = self.observation_space.spaces["obs"]
        candidate_space = self.observation_space.spaces["candidates"]
        # state_dim Ds: product of the raw observation shape.
        #   CartPole: obs shape (4,)      -> Ds=4
        #   CARLA lidar: obs shape (362,) -> Ds=362
        #   CARLA seq: obs shape (5, 9)   -> Ds=45
        #   CNN BDP: image obs -> CNN features, so Ds is the CNN feature dim.
        self._setup_state_features()
        # max_candidates K: fixed number of rows stored in obs["candidates"].
        # candidate_dim Da: flattened feature length of one candidate row.
        #   Discrete one-hot env with n actions: Da=n_actions
        #   CARLA BDP trajectory feature: Da=T_ac_candidates+1
        self.max_candidates = int(candidate_space.shape[0])
        self.candidate_dim = int(np.prod(candidate_space.shape[1:]))

        self._build_mlp_extractor()
        self.goodness_net = nn.Linear(self.mlp_extractor.latent_dim_pi, 1)
        self.value_net = nn.Linear(self.value_mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            # Same gains as SB3 ActorCriticPolicy: hidden extractor layers use
            # sqrt(2), action logits start small, and critic output uses 1.0.
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.value_mlp_extractor: np.sqrt(2),
                self.goodness_net: 0.01,
                self.value_net: 1.0,
            }
            for module, gain in module_gains.items():
                module.apply(lambda submodule, gain=gain: self.init_weights(submodule, gain=gain))

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
        raw_obs = obs["obs"]
        candidates = obs["candidates"].float()
        mask = obs["candidate_mask"].float()

        # Flatten only the feature dimensions, not the batch:
        #   state_features: (B, Ds)
        #   candidates:     (B, K, Da)
        state_features = self._extract_state_features(raw_obs)
        candidates = candidates.reshape(candidates.shape[0], candidates.shape[1], -1)
        return state_features, candidates, mask

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
        # Flatten the first two axes for a normal MLP extractor:
        #   (B, K, Ds + Da) -> (B*K, Ds + Da)
        # Then reshape:
        #   (B*K, 1) -> (B, K)
        latent_pi = self.mlp_extractor.forward_actor(scorer_input.reshape(batch_size * n_candidates, -1))
        logits = self.goodness_net(latent_pi)
        logits = logits.reshape(batch_size, n_candidates)

        # Padding slots get zero probability after the categorical softmax.
        # We use a large finite negative value instead of -inf because it is
        # friendlier to PPO/TRPO math, saving/loading, and debug prints.
        return logits.masked_fill(mask < 0.5, -1e9)

    def _values(self, obs: PyTorchObs) -> th.Tensor:
        state, _, _ = self._state_and_candidates(obs)
        # value_net(value_mlp_extractor(state)): (B, 1), same convention
        # expected by SB3 on-policy algorithms for critic values.
        latent_vf = self.value_mlp_extractor.forward_critic(state)
        return self.value_net(latent_vf)

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


class BDPCnnBoltzmannPolicy(BDPBoltzmannPolicy):
    """
    Candidate-action BDP policy with a CNN state feature extractor.

    This is the BDP analogue of SB3's ``ActorCriticCnnPolicy``.  The image
    state is first processed by ``NatureCNN`` (by default), then each candidate
    vector is concatenated with the CNN state feature before the BDP goodness
    head scores it.

        image obs -> NatureCNN -> state_features
        [state_features, candidate] -> actor MLP -> goodness_net -> logit
        state_features -> critic MLP -> value_net -> V(s)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Schedule,
        candidate_net_arch: Optional[List[int]] = None,
        value_net_arch: Optional[List[int]] = None,
        features_extractor_class: Type[BaseFeaturesExtractor] = NatureCNN,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            candidate_net_arch=candidate_net_arch,
            value_net_arch=value_net_arch,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            **kwargs,
        )
