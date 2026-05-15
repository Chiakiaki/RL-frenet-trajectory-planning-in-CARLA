"""Custom SB3 feature extractors shared by built-in and BDP CNN policies."""

from __future__ import annotations

import gymnasium as gym
import torch as th
from gymnasium import spaces
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class EfficientCNN(BaseFeaturesExtractor):
    """
    Lightweight CNN feature extractor for image observations.

    SB3's default ``NatureCNN`` is a strong Atari-style baseline.  This extractor
    uses fewer early channels plus adaptive pooling, which is useful as a
    cheaper provisional CNN for CarRacing-style 96x96 inputs while keeping the
    same SB3 ``BaseFeaturesExtractor`` interface.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(f"EfficientCNN requires a Box observation space, got {observation_space}")
        if not is_image_space(observation_space, check_channels=False, normalized_image=normalized_image):
            raise ValueError(
                "EfficientCNN expects image observations. For normalized image tensors, "
                "pass normalize_images=False so SB3 sets normalized_image=True."
            )

        super().__init__(observation_space, features_dim)
        n_input_channels = int(observation_space.shape[0])
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
