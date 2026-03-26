"""Training loop: instantiates env from config, selects PPO policy, runs training."""

import logging

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
# from sbx import PPO

from envs.lava.env import lavaEnv
from envs.follower.env import followerEnv
from envs.standard.env import standardEnv
from utils.training_utils import SmallCNN, SmallCNNCombinedExtractor

log = logging.getLogger(__name__)

# NatureCNN's first conv has an 8×8 kernel — images smaller than this need a custom CNN
_NATURE_CNN_MIN_SIZE = 36


def record_videos(model: PPO, cfg: DictConfig) -> None:
    """Roll out the trained model and save N episodes as mp4 files."""
    r = cfg.record
    n = r.n_videos
    env_name: str = cfg.env.name
    video_dir: str = f"{r.video_dir}/{env_name}"

    env = make_env(cfg, render_mode="rgb_array")
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


def make_env(cfg: DictConfig, render_mode: str | None = None) -> gym.Env:
    """Instantiate the env named in cfg."""
    env_name: str = cfg.env.name
    env_kwargs: dict = OmegaConf.to_container(cfg.env.get("kwargs", {}), resolve=True)
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode
    if env_name == "lava":
        return lavaEnv(**env_kwargs)
    elif env_name == "follower":
        return followerEnv(**env_kwargs)
    elif env_name == "standard":
        return standardEnv(**env_kwargs)
    else:
        raise KeyError(f"Unknown env '{env_name}'.")


def select_policy(obs_space: gym.Space) -> tuple[str, dict]:
    """Return (policy_str, policy_kwargs) for the given observation space.

    - Dict obs with small image       → MultiInputPolicy + SmallCNNCombinedExtractor
    - Dict obs with large image       → MultiInputPolicy (default NatureCNN)
    - 3-D Box, spatial dims ≥ 36     → CnnPolicy with default NatureCNN
    - 3-D Box, spatial dims < 36     → CnnPolicy with SmallCNN extractor
    - Everything else                 → MlpPolicy
    """
    if isinstance(obs_space, gym.spaces.Dict):
        # Check if any image subspace is too small for NatureCNN
        needs_small_cnn = any(
            isinstance(subspace, gym.spaces.Box)
            and len(subspace.shape) == 3
            and (subspace.shape[0] < _NATURE_CNN_MIN_SIZE or subspace.shape[1] < _NATURE_CNN_MIN_SIZE)
            for subspace in obs_space.spaces.values()
        )
        if needs_small_cnn:
            return "MultiInputPolicy", {"features_extractor_class": SmallCNNCombinedExtractor}
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
        device="cpu",
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
