import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(BaseFeaturesExtractor):
    """Two-layer CNN for small observations (e.g. 7×7 MiniGrid images).

    SB3's VecTransposeImage converts HxWxC → CxHxW before the extractor sees it,
    so observation_space.shape is (C, H, W) here.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
