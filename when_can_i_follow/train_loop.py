"""Training loop: instantiates env from config, selects PPO policy, runs training."""

import logging

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO

from envs.gridworld.env import gridworldEnv
from utils.training_utils import SmallCNN

log = logging.getLogger(__name__)

# NatureCNN's first conv has an 8×8 kernel — images smaller than this need a custom CNN
_NATURE_CNN_MIN_SIZE = 36


def record_videos(model: PPO, cfg: DictConfig) -> None:
    """Roll out the trained model and save N episodes as mp4 files."""
    r = cfg.record
    n = r.n_videos
    video_dir: str = r.video_dir
    env_kwargs: dict = OmegaConf.to_container(cfg.env.get("kwargs", {}), resolve=True)

    env = gridworldEnv(render_mode="rgb_array", **env_kwargs)
    env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda ep: ep < n,
        disable_logger=True,
    )

    log.info("Recording %d episode(s) to %s", n, video_dir)
    for _ in range(n):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()
    log.info("Videos saved to %s", video_dir)


def make_env(cfg: DictConfig) -> gym.Env:
    """Instantiate the env named in cfg."""
    env_name: str = cfg.env.name
    env_kwargs: dict = OmegaConf.to_container(cfg.env.get("kwargs", {}), resolve=True)
    if env_name == "gridworld":
        return gridworldEnv(**env_kwargs)
    else:
        raise KeyError(f"Unknown env '{env_name}'. Available: {list(ENV_REGISTRY.keys())}")


def select_policy(obs_space: gym.Space) -> tuple[str, dict]:
    """Return (policy_str, policy_kwargs) for the given observation space.

    - Dict obs                        → MultiInputPolicy
    - 3-D Box, spatial dims ≥ 36     → CnnPolicy with default NatureCNN
    - 3-D Box, spatial dims < 36     → CnnPolicy with SmallCNN extractor
    - Everything else                 → MlpPolicy
    """
    if isinstance(obs_space, gym.spaces.Dict):
        return "MultiInputPolicy", {}
    if isinstance(obs_space, gym.spaces.Box) and len(obs_space.shape) >= 3:
        h, w = obs_space.shape[:2]
        if h < _NATURE_CNN_MIN_SIZE or w < _NATURE_CNN_MIN_SIZE:
            return "CnnPolicy", {"features_extractor_class": SmallCNN}
        return "CnnPolicy", {}
    return "MlpPolicy", {}


def train(cfg: DictConfig) -> PPO:
    env = make_env(cfg)
    policy, policy_kwargs = select_policy(env.observation_space)

    log.info("Env: %s  |  obs_space: %s", cfg.env.name, env.observation_space)
    log.info("Selected policy: %s", policy)

    t = cfg.train
    model = PPO(
        policy=policy,
        env=env,
        policy_kwargs=policy_kwargs,
        n_steps=t.n_steps,
        batch_size=t.batch_size,
        n_epochs=t.n_epochs,
        learning_rate=t.learning_rate,
        gamma=t.gamma,
        gae_lambda=t.gae_lambda,
        clip_range=t.clip_range,
        ent_coef=t.ent_coef,
        verbose=1,
    )

    model.learn(
        total_timesteps=t.total_timesteps,
        log_interval=t.log_interval,
    )

    save_path = t.get("save_path")
    if save_path:
        model.save(save_path)
        log.info("Model saved to %s.zip", save_path)

    env.close()

    if cfg.record.n_videos > 0:
        record_videos(model, cfg)

    return model
