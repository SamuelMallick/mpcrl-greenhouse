import logging
import pickle

# import networkx as netx
import numpy as np
from gymnasium.wrappers import TimeLimit
from mpcrl import WarmStartStrategy
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes

from greenhouse.env import GreenhouseSampleAgent, LettuceGreenHouse
from greenhouse.model import (
    generate_parameters,
    get_control_bounds,
    get_model_details,
    output_true,
)
from mpcs import SampleBasedMpc
from utils.plot import plot_greenhouse

np_random = np.random.default_rng(1)

STORE_DATA = True
PLOT = False

nx, nu, nd, ts, _ = get_model_details()
u_min, u_max, du_lim = get_control_bounds()

c_u = np.array([10, 1, 1])  # penalty on each control signal
c_y = np.array([1e3])  # reward on yield

# generate the perturbed parameters
generate_parameters(0.2)

days = 40
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps
env = MonitorEpisodes(
    TimeLimit(
        LettuceGreenHouse(days_to_grow=days, model_type="nonlinear"),
        max_episode_steps=int(ep_len),
    )
)
num_episodes = 1

TD = []

multistarts = 10
Ns = 20
sample_mpc = SampleBasedMpc(
    Ns=Ns, prediction_model="rk4", multistarts=multistarts, np_random=np_random
)
agent = Log(
    GreenhouseSampleAgent(
        mpc=sample_mpc,
        fixed_parameters={},
        warmstart=WarmStartStrategy(
            random_points=sample_mpc.random_points,
            update_biases_for_random_points=False,
            seed=np_random,
        ),
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
    to_file=True,
    log_name=f"log_sample_{Ns}",
)
agent.evaluate(env=env, episodes=num_episodes, seed=1, raises=False)

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

if PLOT:
    plot_greenhouse(X, U, y, d, TD, R, num_episodes, ep_len)

param_dict = {}
if STORE_DATA:
    with open(
        f"sample_{Ns}.pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(y, file)
        pickle.dump(d, file)
        pickle.dump(R, file)
        pickle.dump(TD, file)
        pickle.dump(param_dict, file)
