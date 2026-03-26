"""Hydra entrypoint for PPO training.

Run from the repo root:
    python -m when_can_i_follow.train
    python -m when_can_i_follow.train env.name=lava train.total_timesteps=200000
"""

import hydra
from omegaconf import DictConfig

from when_can_i_follow.train_loop import train


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
