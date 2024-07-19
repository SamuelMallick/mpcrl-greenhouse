from functools import partial
from itertools import chain
from operator import neg

import numpy as np
import torch as th
from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers import TimeLimit, TransformReward
from mpcrl.wrappers.envs import MonitorEpisodes
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model


class AugmentedObservationWrapper(ObservationWrapper):
    """Wrapper that augments the observation with the one-step yield improvement,
    weather disturbances and previous input."""

    def __init__(self, env: LettuceGreenHouse) -> None:
        super().__init__(env)
        nx, nd = env.get_wrapper_attr("nx"), env.get_wrapper_attr("nd")
        act = env.action_space
        lby, uby = [-np.inf, 0.0, -273.15, 0.0], np.full(nx, np.inf)
        lbd, ubd = [0.0, 0.0, -273.15, 0.0], np.full(nd, np.inf)
        self.observation_space = spaces.Box(
            np.concatenate((lby, lbd, act.low)),
            np.concatenate((uby, ubd, act.high)),
            dtype=act.dtype,
            seed=act.np_random,
        )

    def observation(self, state: np.ndarray) -> np.ndarray:
        env: LettuceGreenHouse = self.env.unwrapped
        output = Model.output(state, env.p)
        output[0] -= env.previous_lettuce_yield
        new_state = np.concatenate(
            (output, env.current_disturbance, env.previous_action), axis=None
        )
        assert self.observation_space.contains(new_state), "Invalid observation."
        return new_state


def clip_grad(grad: th.Tensor, max_norm: float) -> th.Tensor:
    """Clips the gradient's L2 norm to a maximum value."""
    norm = th.linalg.vector_norm(grad)
    clip_coef = th.clamp_max(max_norm / (norm + 1e-6), 1.0)
    return grad.mul(clip_coef)


def make_env(
    gamma: float, days: int, evaluation: bool = False, seed: int | None = None
) -> tuple[VecNormalize, int]:
    """Creates and appropriately wraps the greenhouse env for training or evaluation,
    and returns also the number of steps per episode."""
    greenhouse = LettuceGreenHouse(
        growing_days=days,
        model_type="continuous",
        cost_parameters_dict={
            "c_u": [10, 1, 1],
            "c_y": 0.0,
            "c_dy": 100,
            "w_y": np.full((1, 4), 1e5),
        },
        disturbance_profiles_type="single",
        noisy_disturbance=True,
        testing="none",
        clip_action_variation=True,
    )
    max_episode_steps = days * LettuceGreenHouse.steps_per_day
    env = MonitorEpisodes(TimeLimit(greenhouse, max_episode_steps=max_episode_steps))

    # make the env compatible with RL - augment state, and flip cost into reward
    env = TransformReward(AugmentedObservationWrapper(env), neg)

    # add wrappers for SB3
    env = Monitor(env)
    venv = DummyVecEnv([lambda: env])
    venv.set_options({"initial_day": 0, "noise_coeff": 1.0})
    venv = VecNormalize(
        venv, not evaluation, clip_obs=np.inf, clip_reward=np.inf, gamma=gamma
    )
    venv.seed(seed)
    return venv, min(max_episode_steps, greenhouse.yield_step)


def train_ddpg(
    agent_num: int,
    episodes: int,
    days_per_episode: int,
    learning_rate: float,
    gradient_threshold: float,
    l2_regularization: float,
    batch_size: int,
    buffer_size: int,
    gamma: float,
    seed: int,
    device: str,
    verbose: int,
) -> tuple[MonitorEpisodes, MonitorEpisodes]:
    """Trains a DDPG agent on the greenhouse environment.

    Parameters
    ----------
    agent_num : int
        Number of this current agent, used for saving.
    episodes : int
        Number of episodes to train the agent for.
    days_per_episode : int
        Number of days per episode to simualte.
    learning_rate : float
        The learning rate of the RL algorithm.
    gradient_threshold : float
        Threshold for the gradient norm clipping.
    l2_regularization : float
        L2 regularization strength, i.e., rate of weight decay.
    batch_size : int
        Size of mini-batches sampled from the replay buffer when updating.
    buffer_size : int
        Size of the whole replay buffer.
    gamma : float
        Discount factor.
    seed : int
        RNG seed for the simulation.
    device : "auto", or "cpu", or "cuda:0", etc.
        Device to use for training.
    verbose : {0, 1, 2, 3}
        Level of verbosity of the agent's logger.

    Returns
    -------
    tuple of two envs wrapped in `MonitorEpisodes`
        Returns the training and evaluation environments wrapped in `MonitorEpisodes`.
    """
    set_random_seed(seed, using_cuda=device.startswith("cuda"))

    # create the model
    env, steps_per_episode = make_env(gamma, days_per_episode, seed=seed)
    na = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(np.zeros(na), np.full(na, 0.3), dt=1.0)
    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=batch_size,
        tau=1e-3,
        gamma=gamma,
        train_freq=(96, "step"),
        gradient_steps=-1,
        action_noise=action_noise,
        policy_kwargs={
            "net_arch": [256, 256],
            "optimizer_kwargs": {"weight_decay": l2_regularization},
        },
        verbose=verbose,
        seed=seed,
        device=device,
    )
    model.set_logger(configure(".", ["log"]))

    # add hooks to perform L2 gradient norm clipping
    for p in chain(model.actor.parameters(), model.critic.parameters()):
        p.register_hook(partial(clip_grad, max_norm=gradient_threshold))

    # create the eval callback
    eval_env, _ = make_env(gamma, days_per_episode, evaluation=True, seed=seed)
    cb = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=20,
        eval_freq=int(steps_per_episode * episodes / 50),
        verbose=verbose,
    )

    # launch the training
    total_timesteps = steps_per_episode * episodes
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=cb)

    # save to disk the trained agent and the training env (it has the normalizations)
    env = model.get_env()
    env.save(f"ddpg_env_{agent_num}.pkl")
    model.save(f"ddpg_agent_{agent_num}")

    # return as data the `MonitorEpisodes` from the training and evaluation envs - ugly,
    # but they must be digged out from the `VecNormalize` wrapper
    return env.envs[0].env.env.env, eval_env.envs[0].env.env.env
