"""Minimal CleanRL-style PPO implementation backed by JAX/Flax.

This module exposes a small model class with an SB3-like surface:
`learn()`, `predict()`, `save()`, and `load()`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JaxPPOConfig:
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_envs: int = 1


class Encoder(nn.Module):
    has_image: bool
    image_key: str | None
    vector_keys: tuple[str, ...]

    @nn.compact
    def __call__(self, obs: dict[str, jnp.ndarray]) -> jnp.ndarray:
        chunks: list[jnp.ndarray] = []

        if self.has_image and self.image_key is not None:
            x = obs[self.image_key].astype(jnp.float32) / 255.0
            x = nn.Conv(
                32,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
            x = nn.Conv(
                64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="VALID",
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
            x = x.reshape((x.shape[0], -1))
            chunks.append(x)

        for key in self.vector_keys:
            x = obs[key].astype(jnp.float32)
            x = x.reshape((x.shape[0], -1))
            chunks.append(x)

        if not chunks:
            raise ValueError("No observation features found to encode.")

        if len(chunks) == 1:
            x = chunks[0]
        else:
            x = jnp.concatenate(chunks, axis=-1)

        x = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2.0)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        return x


class ActorCritic(nn.Module):
    action_dim: int
    has_image: bool
    image_key: str | None
    vector_keys: tuple[str, ...]

    @nn.compact
    def __call__(self, obs: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        h = Encoder(
            has_image=self.has_image,
            image_key=self.image_key,
            vector_keys=self.vector_keys,
        )(obs)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(h)
        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(h).squeeze(-1)
        return logits, value


class JaxPPOModel:
    """PPO model with a small SB3-like API."""

    def __init__(self, envs: gym.vector.VectorEnv, cfg: JaxPPOConfig, seed: int = 1):
        if not isinstance(envs.single_action_space, gym.spaces.Discrete):
            raise TypeError("Only discrete action spaces are supported by this JAX PPO path.")

        obs_space = envs.single_observation_space
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError("Expected Dict observation space for current environments.")

        self.envs = envs
        self.cfg = cfg
        self._seed = seed
        self._action_dim = int(envs.single_action_space.n)

        image_key: str | None = None
        vector_keys: list[str] = []
        for key, subspace in obs_space.spaces.items():
            if (
                image_key is None
                and isinstance(subspace, gym.spaces.Box)
                and len(subspace.shape) == 3
                and np.issubdtype(subspace.dtype, np.integer)
            ):
                image_key = key
            else:
                vector_keys.append(key)

        self._has_image = image_key is not None
        self._image_key = image_key
        self._vector_keys = tuple(sorted(vector_keys))
        self._model = ActorCritic(
            action_dim=self._action_dim,
            has_image=self._has_image,
            image_key=self._image_key,
            vector_keys=self._vector_keys,
        )

        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        obs_sample = self._obs_to_jnp(self._sample_batched_obs(obs_space))
        params = self._model.init(init_key, obs_sample)
        tx = optax.chain(
            optax.clip_by_global_norm(cfg.max_grad_norm),
            optax.adam(cfg.learning_rate, eps=1e-5),
        )
        self._state = TrainState.create(apply_fn=self._model.apply, params=params, tx=tx)
        self._rng = key

        self._apply = jax.jit(self._model.apply)
        self._sample_action = jax.jit(self._sample_action_impl, static_argnums=(3,))
        self._logprob_value = jax.jit(self._logprob_value_impl)
        self._update_step = jax.jit(self._update_step_impl)

    @property
    def learning_rate(self) -> float:
        return float(self.cfg.learning_rate)

    def _sample_batched_obs(self, obs_space: gym.spaces.Dict) -> dict[str, np.ndarray]:
        sample = obs_space.sample()
        return {k: np.repeat(sample[k][None, ...], self.cfg.num_envs, axis=0) for k in sample}

    @staticmethod
    def _obs_to_jnp(obs: dict[str, np.ndarray]) -> dict[str, jnp.ndarray]:
        return {k: jnp.asarray(v) for k, v in obs.items()}

    @staticmethod
    def _obs_to_numpy(obs: dict[str, Any]) -> dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in obs.items()}

    def _sample_action_impl(
        self, params: flax.core.FrozenDict, obs: dict[str, jnp.ndarray], key: jax.Array, deterministic: bool
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        logits, values = self._model.apply(params, obs)
        log_probs = jax.nn.log_softmax(logits)
        if deterministic:
            actions = jnp.argmax(logits, axis=-1)
        else:
            key, subkey = jax.random.split(key)
            actions = jax.random.categorical(subkey, logits, axis=-1)
        chosen_logprobs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(-1)
        return actions, chosen_logprobs, values, key

    def _logprob_value_impl(
        self,
        params: flax.core.FrozenDict,
        obs: dict[str, jnp.ndarray],
        actions: jnp.ndarray,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        logits, values = self._model.apply(params, obs)
        log_probs = jax.nn.log_softmax(logits)
        probs = jax.nn.softmax(logits)
        chosen_logprobs = jnp.take_along_axis(log_probs, actions[:, None], axis=1).squeeze(-1)
        entropy = -(probs * log_probs).sum(axis=-1)
        return chosen_logprobs, entropy, values

    def _update_step_impl(
        self,
        state: TrainState,
        obs_mb: dict[str, jnp.ndarray],
        actions_mb: jnp.ndarray,
        old_logprob_mb: jnp.ndarray,
        adv_mb: jnp.ndarray,
        returns_mb: jnp.ndarray,
    ) -> tuple[TrainState, jax.Array, jax.Array, jax.Array, jax.Array]:
        def loss_fn(params: flax.core.FrozenDict) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
            new_logprob, entropy, new_values = self._logprob_value(params, obs_mb, actions_mb)
            adv_norm = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
            logratio = new_logprob - old_logprob_mb
            ratio = jnp.exp(logratio)
            pg1 = -adv_norm * ratio
            pg2 = -adv_norm * jnp.clip(ratio, 1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range)
            pg_loss = jnp.maximum(pg1, pg2).mean()
            value_loss = 0.5 * jnp.mean((new_values - returns_mb) ** 2)
            entropy_loss = entropy.mean()
            total_loss = pg_loss + self.cfg.vf_coef * value_loss - self.cfg.ent_coef * entropy_loss
            approx_kl = jnp.mean((ratio - 1.0) - logratio)
            return total_loss, (pg_loss, value_loss, entropy_loss, approx_kl)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        pg_loss, value_loss, entropy_loss, approx_kl = aux
        return state, loss, pg_loss, value_loss, entropy_loss, approx_kl

    def predict(self, obs: dict[str, np.ndarray], deterministic: bool = True) -> tuple[np.ndarray, None]:
        obs_np = self._obs_to_numpy(obs)
        batched = {
            k: v[None, ...] if v.ndim == len(self.envs.single_observation_space.spaces[k].shape) else v
            for k, v in obs_np.items()
        }
        obs_jnp = self._obs_to_jnp(batched)
        actions, _, _, self._rng = self._sample_action(self._state.params, obs_jnp, self._rng, deterministic)
        action = np.asarray(actions)[0]
        return action, None

    def _flatten_obs(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: obs[k].reshape((-1,) + obs[k].shape[2:]) for k in obs}

    def learn(self, total_timesteps: int, log_interval: int = 10) -> "JaxPPOModel":
        rollout_size = self.cfg.n_steps * self.cfg.num_envs
        if rollout_size % self.cfg.batch_size != 0:
            raise ValueError(
                f"rollout_size ({rollout_size}) must be divisible by batch_size ({self.cfg.batch_size})."
            )

        updates = total_timesteps // rollout_size
        if updates <= 0:
            log.warning("Requested total_timesteps=%d yields zero PPO updates; skipping learn.", total_timesteps)
            return self

        obs, _ = self.envs.reset(seed=self._seed)
        obs = self._obs_to_numpy(obs)
        dones = np.zeros(self.cfg.num_envs, dtype=np.float32)
        episodic_returns = np.zeros(self.cfg.num_envs, dtype=np.float32)
        recent_returns: list[float] = []
        global_step = 0
        start_time = time.time()

        for update in range(1, updates + 1):
            obs_buf = {k: np.zeros((self.cfg.n_steps, self.cfg.num_envs) + v.shape[1:], dtype=v.dtype) for k, v in obs.items()}
            actions_buf = np.zeros((self.cfg.n_steps, self.cfg.num_envs), dtype=np.int32)
            logprob_buf = np.zeros((self.cfg.n_steps, self.cfg.num_envs), dtype=np.float32)
            rewards_buf = np.zeros((self.cfg.n_steps, self.cfg.num_envs), dtype=np.float32)
            dones_buf = np.zeros((self.cfg.n_steps, self.cfg.num_envs), dtype=np.float32)
            values_buf = np.zeros((self.cfg.n_steps, self.cfg.num_envs), dtype=np.float32)

            for step in range(self.cfg.n_steps):
                for key in obs:
                    obs_buf[key][step] = obs[key]
                dones_buf[step] = dones

                action, logprob, value, self._rng = self._sample_action(
                    self._state.params,
                    self._obs_to_jnp(obs),
                    self._rng,
                    False,
                )
                action_np = np.asarray(action, dtype=np.int32)
                next_obs, rewards, terminations, truncations, _ = self.envs.step(action_np)
                done_mask = np.logical_or(terminations, truncations)

                actions_buf[step] = action_np
                logprob_buf[step] = np.asarray(logprob, dtype=np.float32)
                values_buf[step] = np.asarray(value, dtype=np.float32)
                rewards_buf[step] = rewards.astype(np.float32)

                episodic_returns += rewards.astype(np.float32)
                finished = np.where(done_mask)[0]
                if finished.size > 0:
                    recent_returns.extend(episodic_returns[finished].tolist())
                    episodic_returns[finished] = 0.0

                obs = self._obs_to_numpy(next_obs)
                dones = done_mask.astype(np.float32)
                global_step += self.cfg.num_envs

            _, _, last_values = self._logprob_value(
                self._state.params,
                self._obs_to_jnp(obs),
                jnp.zeros((self.cfg.num_envs,), dtype=jnp.int32),
            )
            last_values_np = np.asarray(last_values, dtype=np.float32)

            advantages = np.zeros_like(rewards_buf)
            lastgaelam = np.zeros(self.cfg.num_envs, dtype=np.float32)
            for t in reversed(range(self.cfg.n_steps)):
                if t == self.cfg.n_steps - 1:
                    nextnonterminal = 1.0 - dones
                    nextvalues = last_values_np
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + self.cfg.gamma * nextvalues * nextnonterminal - values_buf[t]
                lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + values_buf

            b_obs = self._flatten_obs(obs_buf)
            b_actions = actions_buf.reshape(-1)
            b_logprob = logprob_buf.reshape(-1)
            b_adv = advantages.reshape(-1)
            b_returns = returns.reshape(-1)

            last_loss = 0.0
            last_pg_loss = 0.0
            last_v_loss = 0.0
            last_entropy = 0.0
            last_approx_kl = 0.0

            b_inds = np.arange(rollout_size)
            for _ in range(self.cfg.n_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, rollout_size, self.cfg.batch_size):
                    mb_inds = b_inds[start : start + self.cfg.batch_size]
                    obs_mb = {k: jnp.asarray(v[mb_inds]) for k, v in b_obs.items()}
                    self._state, loss, pg_loss, value_loss, entropy_loss, approx_kl = self._update_step(
                        self._state,
                        obs_mb,
                        jnp.asarray(b_actions[mb_inds]),
                        jnp.asarray(b_logprob[mb_inds]),
                        jnp.asarray(b_adv[mb_inds]),
                        jnp.asarray(b_returns[mb_inds]),
                    )
                    last_loss = float(loss)
                    last_pg_loss = float(pg_loss)
                    last_v_loss = float(value_loss)
                    last_entropy = float(entropy_loss)
                    last_approx_kl = float(approx_kl)

            if update % log_interval == 0 or update == updates:
                avg_return = float(np.mean(recent_returns[-50:])) if recent_returns else float("nan")
                elapsed = max(time.time() - start_time, 1e-9)
                sps = int(global_step / elapsed)
                log.info(
                    "update=%d/%d global_step=%d sps=%d avg_return=%.3f loss=%.4f pg=%.4f v=%.4f ent=%.4f kl=%.5f",
                    update,
                    updates,
                    global_step,
                    sps,
                    avg_return,
                    last_loss,
                    last_pg_loss,
                    last_v_loss,
                    last_entropy,
                    last_approx_kl,
                )
        return self

    def save(self, path: str) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"state": self._state, "rng": self._rng}
        out_path.write_bytes(flax.serialization.to_bytes(payload))

    def load(self, path: str, reset_optimizer: bool = False) -> "JaxPPOModel":
        in_path = Path(path)
        if not in_path.exists():
            zip_path = Path(f"{path}.zip")
            if zip_path.exists():
                in_path = zip_path
            else:
                raise FileNotFoundError(f"Model checkpoint not found: {path}")

        try:
            payload = flax.serialization.from_bytes(
                {"state": self._state, "rng": self._rng},
                in_path.read_bytes(),
            )
        except Exception as exc:
            raise ValueError(
                f"Incompatible checkpoint format at '{in_path}'. "
                "JAX PPO checkpoints are not compatible with SB3 .zip files."
            ) from exc

        loaded_state = payload["state"]
        if reset_optimizer:
            loaded_state = TrainState.create(
                apply_fn=self._model.apply,
                params=loaded_state.params,
                tx=self._state.tx,
            )
        self._state = loaded_state
        self._rng = payload["rng"]
        return self