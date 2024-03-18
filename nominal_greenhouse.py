import logging
import pickle

# import networkx as netx
import numpy as np
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes

from envs.env import GreenhouseAgent, LettuceGreenHouse
from envs.model import output_true
from nominal_MPC import NominalMpc
from plot_green import plot_greenhouse

np.random.seed(1)

STORE_DATA = True
PLOT = True

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
# mpc = NominalMpc(prediction_model="rk4")
agent = Log(
    GreenhouseAgent(mpc, {}),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)
agent.evaluate(
    env=env,
    episodes=num_episodes,
    seed=1,
    raises=False,
    env_reset_options={"first_day_index": 0},
)

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
d = env.disturbance_profile

if PLOT:
    plot_greenhouse(X, U, y, d, TD, R, num_episodes, ep_len)

param_dict = {}
identifier = "day_0"
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
