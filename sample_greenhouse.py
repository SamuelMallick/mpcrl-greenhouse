import logging
import pickle
from typing import Literal

# import networkx as netx
import numpy as np
from gymnasium.wrappers import TimeLimit
from mpcrl import WarmStartStrategy
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes

from agents.greenhouse_agent import GreenhouseSampleAgent
from greenhouse.env import LettuceGreenHouse
from mpcs.sample_based import SampleBasedMpc
from utils.plot import plot_greenhouse

np_random = np.random.default_rng(1)

STORE_DATA = True
PLOT = False

days = 40
episode_len = days * 24 * 4  # x days of 15 minute timesteps
env = MonitorEpisodes(
    TimeLimit(
        LettuceGreenHouse(
            growing_days=days,
            model_type="continuous",
            disturbance_profiles_type="single",
            noisy_disturbance=True,
            testing="none",
        ),
        max_episode_steps=int(episode_len),
    )
)
num_episodes = 100
initial_days = [0]

multistarts = 1
num_samples = 2
prediction_model: Literal["euler", "rk4"] = "rk4"
sample_mpc = SampleBasedMpc(
    n_samples=num_samples,
    greenhouse_env=env,
    cost_parameters_dict={
        "c_u": np.array([10, 1, 1]),
        "c_y": 1e3,
        "w_y": 1e5 * np.ones(4),
    },  # MPC cost tuned from 2022 paper
    prediction_model=prediction_model,
    multistarts=multistarts,
    np_random=np_random,
)
agent = Log(
    GreenhouseSampleAgent(
        mpc=sample_mpc,
        fixed_parameters={},
        warmstart=WarmStartStrategy(
            random_points=sample_mpc.random_start_points,
            update_biases_for_random_points=False,
            seed=np_random,
        ),
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
    to_file=True,
    log_name=f"log_sample_{num_samples}_{multistarts}_{prediction_model}",
)

for initial_day in initial_days:
    agent.evaluate(
        env=env,
        episodes=num_episodes,
        seed=1,
        raises=False,
        env_reset_options={"initial_day": initial_day},
    )

    # extract data
    X = np.asarray(env.observations)
    U = np.asarray(env.actions).squeeze(-1)
    R = np.asarray(env.rewards)
    d = np.asarray(env.disturbance_profiles_all_episodes).transpose(0, 2, 1)

    print(f"Return = {R.sum(axis=1)}")

    if PLOT:
        plot_greenhouse(X, U, d, R, None)

    identifier = f"sample_greenhouse_{prediction_model}_{num_samples}_{multistarts}_{initial_day}"
    if STORE_DATA:
        with open(
            identifier + ".pkl",
            "wb",
        ) as file:
            pickle.dump({"name": identifier, "X": X, "U": U, "R": R, "d": d}, file)
