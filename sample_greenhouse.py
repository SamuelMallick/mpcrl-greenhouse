import logging
import pickle
from typing import Literal

# import networkx as netx
import numpy as np
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes

from greenhouse.env import LettuceGreenHouse
from mpcs.sample_based import SampleBasedMpc
from agents.greenhouse_agent import GreenhouseSampleAgent
from utils.plot import plot_greenhouse

np_random = np.random.default_rng(1)

# TODO compare performance with previous implementation

STORE_DATA = False
PLOT = True

days = 1
episode_len = days * 24 * 4  # x days of 15 minute timesteps
env = MonitorEpisodes(
    TimeLimit(
        LettuceGreenHouse(
            growing_days=days,
            model_type="continuous",
            disturbance_type="multiple",
            testing=True,
        ),
        max_episode_steps=int(episode_len),
    )
)
num_episodes = 1

multistarts = 1
num_samples = 2
prediction_model: Literal["euler", "rk4"] = "rk4"
sample_mpc = SampleBasedMpc(n_samples=num_samples, greenhouse_env=env, prediction_model=prediction_model, multistarts=multistarts, np_random=np_random)
agent = Log(
    GreenhouseSampleAgent(
        mpc=sample_mpc,
        fixed_parameters={},
        # TODO: fix warm starting
        # warmstart=WarmStartStrategy(  
        #     random_points=sample_mpc.random_points,
        #     update_biases_for_random_points=False,
        #     seed=np_random,
        # ),
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
    to_file=True,
    log_name=f"log_sample_{num_samples}_{multistarts}_{prediction_model}",
)
agent.evaluate(env=env, episodes=num_episodes, seed=1, raises=False)

# extract data
X = np.asarray(env.observations)
U = np.asarray(env.actions).squeeze(-1)
R = np.asarray(env.rewards)
d = np.asarray(env.disturbance_profiles_all_episodes).transpose(0, 2, 1)

print(f"Return = {R.sum(axis=1)}")

if PLOT:
    plot_greenhouse(X, U, d, R, None)

param_dict: dict = {}
identifier = f"sample_greenhouse_{prediction_model}_{num_samples}_{multistarts}"
if STORE_DATA:
    with open(
        identifier + ".pkl",
        "wb",
    ) as file:
        pickle.dump({"X": X, "U": U, "R": R, "d": d, "param_dict": param_dict}, file)