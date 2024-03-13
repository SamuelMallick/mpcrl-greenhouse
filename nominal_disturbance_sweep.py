import datetime
import logging
import pickle
from typing import Literal

import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes

from nominal_MPC import NominalMpc
from envs.env import GreenhouseAgent, LettuceGreenHouse
from envs.model import (
    euler_perturbed,
    euler_true,
    get_control_bounds,
    get_initial_perturbed_p,
    get_model_details,
    output_perturbed,
    output_true,
    rk4_perturbed,
    rk4_true,
)
from plot_green import plot_greenhouse

np.random.seed(1)

STORE_DATA = True

nx, nu, nd, ts, steps_per_day = get_model_details()
u_min, u_max, du_lim = get_control_bounds()

c_u = np.array([10, 1, 1])  # penalty on each control signal
c_y = np.array([1e3])  # reward on yield
w = 1e3 * np.ones((1, nx))  # penalty on constraint violations


days = 40
ep_len = days * 24 * 4  # x days of 15 minute timesteps
env = MonitorEpisodes(
    TimeLimit(
        LettuceGreenHouse(days_to_grow=days, model_type="nonlinear"),
        max_episode_steps=int(ep_len),
    )
)
num_episodes = 1
TD = []

mpc = NominalMpc(prediction_model="rk4", correct_model=True, perturb_list=[])
agent = Log(
    GreenhouseAgent(mpc, {}),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)

for first_day_index in range(324):
    agent.evaluate(env=env, episodes=num_episodes, seed=1, raises=False, env_reset_options={"first_day_index": first_day_index})

    # extract data
    if len(env.observations) > 0:
        X = np.hstack([env.observations[i].squeeze().T for i in range(num_episodes)]).T
        U = np.hstack([env.actions[i].squeeze().T for i in range(num_episodes)]).T
        R = np.hstack([env.rewards[i].squeeze().T for i in range(num_episodes)]).T
    else:
        X = np.squeeze(env.ep_observations)
        U = np.squeeze(env.ep_actions)
        R = np.squeeze(env.ep_rewards)

    print(f"Return = {sum(R.squeeze())}")

    R_eps = [sum(R[ep_len * i : ep_len * (i + 1)]) for i in range(num_episodes)]
    TD_eps = [sum(TD[ep_len * i : ep_len * (i + 1)]) / ep_len for i in range(num_episodes)]
    # generate output
    y = np.asarray([output_true(X[k, :]) for k in range(X.shape[0])]).squeeze()
    d = env.disturbance_profile_data

    param_dict = {}
    if STORE_DATA:
        with open(
            f"nom_first_day_{first_day_index}" + ".pkl",
            "wb",
        ) as file:
            pickle.dump(X, file)
            pickle.dump(U, file)
            pickle.dump(y, file)
            pickle.dump(d, file)
            pickle.dump(R, file)
            pickle.dump(TD, file)
            pickle.dump(param_dict, file)
