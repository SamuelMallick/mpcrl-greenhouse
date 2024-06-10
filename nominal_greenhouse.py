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

STORE_DATA = True
PLOT = False

days = 40
episode_len = days * LettuceGreenHouse.steps_per_day  # x days of 15 minute timesteps
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
initial_days = list(range(20))

prediction_model: Literal["euler", "rk4"] = "rk4"
correct_model = True
mpc = NominalMpc(
    greenhouse_env=env,
    cost_parameters_dict={
        "c_u": np.array([10, 1, 1]),
        "c_y": 1e3,
        "w_y": 1e5 * np.ones(4),
    },  # MPC cost tuned from 2022 paper
    prediction_model=prediction_model,
    correct_model=correct_model,
    np_random=np_random,
)

agent = Log(
    GreenhouseAgent(mpc, fixed_parameters={}),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
    to_file=True,
    log_name=f"nominal_greenhouse_{prediction_model}_{correct_model}",
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

    identifier = f"nominal_greenhouse_{prediction_model}_{correct_model}_{initial_day}"
    if STORE_DATA:
        with open(
            identifier + ".pkl",
            "wb",
        ) as file:
            pickle.dump({"name": identifier, "X": X, "U": U, "R": R, "d": d}, file)
