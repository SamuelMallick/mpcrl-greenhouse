import logging
import pickle
from typing import Literal

# import networkx as netx
import numpy as np
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes

from agents.greenhouse_agent import GreenhouseAgent
from greenhouse.env import LettuceGreenHouse
from mpcs.nominal import NominalMpc
from utils.plot import plot_greenhouse

np_random = np.random.default_rng(1)

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

prediction_model: Literal["euler", "rk4"] = "rk4"
correct_model = True
mpc = NominalMpc(
    greenhouse_env=env,
    prediction_model=prediction_model,
    correct_model=correct_model,
    np_random=np_random,
)

agent = Log(
    GreenhouseAgent(
        mpc, {}
    ),  # TODO Pass the fixed parameter names in, rather than letting them just be set by the agent in the first time step
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
    to_file=True,
    log_name=f"nominal_greenhouse_{prediction_model}_{correct_model}",
)
agent.evaluate(
    env=env,
    episodes=num_episodes,
    seed=1,
    raises=False,
)

# extract data
X = np.asarray(env.observations)
U = np.asarray(env.actions).squeeze(-1)
R = np.asarray(env.rewards)
d = np.asarray(env.disturbance_profiles_all_episodes).transpose(0, 2, 1)
# generate outputs


print(f"Return = {R.sum(axis=1)}")

if PLOT:
    plot_greenhouse(X, U, d, R, None)

param_dict: dict = {}
identifier = f"nominal_greenhouse_{prediction_model}_{correct_model}"
if STORE_DATA:
    with open(
        identifier + ".pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(y, file)
        pickle.dump(d, file)
        pickle.dump(R, file)
        pickle.dump(TD, file)
        pickle.dump(param_dict, file)
