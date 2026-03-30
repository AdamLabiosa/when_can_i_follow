from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

_DEFAULT_ACTION_LABELS: dict[int, str] = {
    0: "Turn Left",
    1: "Turn Right",
    2: "Move Forward",
    3: "Request Plan",
}


class ActionTextOverlayWrapper(gym.Wrapper):
    """Renders the last action taken as small text in the top-left corner of each frame.

    Requires PIL (Pillow). Falls back silently if PIL is not installed.
    """

    def __init__(
        self,
        env: gym.Env,
        action_labels: dict[int, str] | None = None,
        font_size: int = 14,
    ) -> None:
        super().__init__(env)
        self._action_labels = action_labels or _DEFAULT_ACTION_LABELS
        self._last_action: int | None = None
        self._font_size = font_size
        self._font = self._load_font(font_size)

    @staticmethod
    def _load_font(size: int):
        if not _PIL_AVAILABLE:
            return None
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except (IOError, OSError):
            return ImageFont.load_default()

    def step(self, action: Any):
        self._last_action = int(action)
        return self.env.step(action)

    def reset(self, **kwargs):
        self._last_action = None
        return self.env.reset(**kwargs)

    def render(self):
        frame = self.env.render()
        if not _PIL_AVAILABLE or frame is None or self._last_action is None:
            return frame

        label = self._action_labels.get(self._last_action, str(self._last_action))
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        x, y = 4, 4
        # Dark shadow for legibility
        draw.text((x + 1, y + 1), label, font=self._font, fill=(0, 0, 0, 200))
        draw.text((x, y), label, font=self._font, fill=(255, 255, 255, 255))
        return np.array(img)


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


class SmallCNNCombinedExtractor(BaseFeaturesExtractor):
    """Drop-in replacement for SB3's CombinedExtractor that uses SmallCNN
    instead of NatureCNN for image subspaces in a Dict observation space."""

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 64):
        super().__init__(observation_space, features_dim=1)

        extractors: dict[str, nn.Module] = {}
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace):
                extractors[key] = SmallCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        encoded = [self.extractors[key](obs) for key, obs in observations.items()]
        return th.cat(encoded, dim=1)
