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
from greenhouse.model import Model
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
if len(env.observations) > 0:
    X = np.hstack([env.observations[i].squeeze().T for i in range(num_episodes)]).T
    U = np.hstack([env.actions[i].squeeze().T for i in range(num_episodes)]).T
    R = np.hstack([env.rewards[i].squeeze().T for i in range(num_episodes)]).T
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

print(f"Return = {sum(R.squeeze())}")

TD: list[float] = []  # TD error not generated in this script
R_eps = [sum(R[episode_len * i : episode_len * (i + 1)]) for i in range(num_episodes)]
TD_eps = [
    sum(TD[episode_len * i : episode_len * (i + 1)]) / episode_len
    for i in range(num_episodes)
]
# generate outputs
y = np.asarray(
    [Model.output(X[k, :], Model.get_true_parameters()) for k in range(X.shape[0])]
).squeeze()
d = env.disturbance_profiles_all_episodes

if PLOT:
    plot_greenhouse(X, U, y, d, TD, R, num_episodes, episode_len)

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
