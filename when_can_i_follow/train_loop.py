"""Training loop: instantiates env from config and trains JAX PPO."""

import logging
import os
from typing import TYPE_CHECKING

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from omegaconf import DictConfig, OmegaConf

from when_can_i_follow.envs.basic.env import basicEnv
from when_can_i_follow.envs.city_block.env import cityBlockEnv
from when_can_i_follow.envs.follower.env import followerEnv
from when_can_i_follow.envs.lava.env import lavaEnv
from when_can_i_follow.envs.moving_lava.env import movingLavaEnv
from when_can_i_follow.envs.moving_openings.env import movingOpeningEnv
from when_can_i_follow.utils.training_utils import ActionTextOverlayWrapper

if TYPE_CHECKING:
    from when_can_i_follow.cleanrl_base import JaxPPOModel

log = logging.getLogger(__name__)


def record_videos(model: "JaxPPOModel", cfg: DictConfig) -> None:
    """Roll out the trained model and save N episodes as mp4 files."""
    r = cfg.record
    n = r.n_videos
    env_name: str = cfg.env.name
    video_dir: str = f"{r.video_dir}/{env_name}"

    env = make_env(cfg, render_mode="rgb_array")
    env = ActionTextOverlayWrapper(env)
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
    if env_name == "basic":
        return basicEnv(**env_kwargs)
    elif env_name == "city_block":
        return cityBlockEnv(**env_kwargs)
    elif env_name == "lava":
        return lavaEnv(**env_kwargs)
    elif env_name == "follower":
        return followerEnv(**env_kwargs)
    elif env_name == "moving_lava":
        return movingLavaEnv(**env_kwargs)
    elif env_name == "moving_opening":
        return movingOpeningEnv(**env_kwargs)
    else:
        raise KeyError(f"Unknown env '{env_name}'.")


def train(cfg: DictConfig) -> "JaxPPOModel":
    t = cfg.train
    # Platform selection happens before importing JAX modules.
    # auto -> prefer CUDA, then fall back to CPU.
    jax_platform = str(t.get("jax_platform", "auto")).strip().lower()
    if jax_platform == "auto":
        if "JAX_PLATFORMS" not in os.environ:
            os.environ["JAX_PLATFORMS"] = "cuda,cpu"
        log.info("Using JAX platform preference: %s", os.environ["JAX_PLATFORMS"])
    elif jax_platform:
        os.environ["JAX_PLATFORMS"] = jax_platform
        log.info("Using JAX platform: %s", jax_platform)

    from when_can_i_follow.cleanrl_base import JaxPPOConfig, JaxPPOModel

    num_envs = int(t.get("num_envs", 1))
    seed = int(t.get("seed", 1))
    envs = gym.vector.SyncVectorEnv([lambda: make_env(cfg) for _ in range(num_envs)])
    log.info("Env: %s  |  obs_space: %s", cfg.env.name, envs.single_observation_space)

    ppo_cfg = JaxPPOConfig(
        n_steps=int(t.n_steps),
        batch_size=int(t.batch_size),
        n_epochs=int(t.n_epochs),
        learning_rate=float(t.learning_rate),
        gamma=float(t.gamma),
        gae_lambda=float(t.gae_lambda),
        clip_range=float(t.clip_range),
        ent_coef=float(t.ent_coef),
        vf_coef=float(t.get("vf_coef", 0.5)),
        max_grad_norm=float(t.get("max_grad_norm", 0.5)),
        num_envs=num_envs,
    )
    model = JaxPPOModel(envs=envs, cfg=ppo_cfg, seed=seed)

    load_path = t.get("load_path")
    if load_path:
        finetune_lr = t.get("finetune_lr")
        lr = float(finetune_lr) if finetune_lr is not None else float(t.learning_rate)
        if lr != model.learning_rate:
            ppo_cfg = JaxPPOConfig(
                n_steps=ppo_cfg.n_steps,
                batch_size=ppo_cfg.batch_size,
                n_epochs=ppo_cfg.n_epochs,
                learning_rate=lr,
                gamma=ppo_cfg.gamma,
                gae_lambda=ppo_cfg.gae_lambda,
                clip_range=ppo_cfg.clip_range,
                ent_coef=ppo_cfg.ent_coef,
                vf_coef=ppo_cfg.vf_coef,
                max_grad_norm=ppo_cfg.max_grad_norm,
                num_envs=ppo_cfg.num_envs,
            )
            model = JaxPPOModel(envs=envs, cfg=ppo_cfg, seed=seed)
        log.info("Loading model from %s  |  learning_rate: %s", load_path, lr)
        model.load(load_path, reset_optimizer=finetune_lr is not None)

    model.learn(
        total_timesteps=int(t.total_timesteps),
        log_interval=int(t.log_interval),
    )

    save_path = t.get("save_path")
    if save_path:
        model.save(save_path)
        log.info("Model saved to %s", save_path)

    envs.close()

    if cfg.record.n_videos > 0:
        record_videos(model, cfg)

    return model
