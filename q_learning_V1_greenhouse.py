import importlib
import logging
import pickle
import sys

import casadi as cs

# import networkx as netx
import numpy as np
from envs.env import GreenhouseLearningAgent, LettuceGreenHouse
from envs.model import (
    generate_parameters,
    get_control_bounds,
    get_model_details,
    get_p_learn_bounds,
    output_true,
)
from gymnasium.wrappers import TimeLimit
from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.wrappers.agents import Evaluate, Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from utils.plot import plot_greenhouse

np.random.seed(1)

# if a config file passed on command line, otherwise use dedault config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"test_configs.{config_file}")
    test = mod.Test()

    STORE_DATA = True
    PLOT = False
else:
    from test_configs.default import Test

    test = Test()

    STORE_DATA = True
    PLOT = True

nx, nu, nd, ts, _ = get_model_details()
u_min, u_max, du_lim = get_control_bounds()

if hasattr(test, "perturb"):
    generate_parameters(test.perturb)
else:
    generate_parameters()


ep_len = test.ep_len
env = MonitorEpisodes(
    TimeLimit(
        LettuceGreenHouse(
            days_to_grow=test.num_days,
            model_type=test.base_model,
            rl_cost=test.rl_cost,
            testing=False,
        ),
        max_episode_steps=int(ep_len),
    )
)
eval_env = MonitorEpisodes(
    TimeLimit(
        LettuceGreenHouse(
            days_to_grow=test.num_days,
            model_type=test.base_model,
            rl_cost=test.rl_cost,
            testing=True,
        ),
        max_episode_steps=int(ep_len),
    )
)


mpc = LearningMpc()
param_bounds = get_p_learn_bounds()
param_bounds.update(test.learn_bounds)
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(
            name,
            val.shape,
            val,
            sym=mpc.parameters[name],
            lb=param_bounds[name][0] if name in param_bounds.keys() else -np.inf,
            ub=param_bounds[name][1] if name in param_bounds.keys() else np.inf,
        )
        for name, val in mpc.learnable_pars_init.items()
    )
)

agent = Evaluate(
    Log(  # type: ignore[var-annotated]
        RecordUpdates(
            GreenhouseLearningAgent(
                mpc=mpc,
                update_strategy=test.update_strategy,
                discount_factor=mpc.discount_factor,
                optimizer=test.optimizer,
                learnable_parameters=learnable_pars,
                fixed_parameters=mpc.fixed_pars,
                exploration=test.exploration,
                experience=test.experience,
                hessian_type=test.hessian_type,
                record_td_errors=True,
            )
        ),
        level=logging.DEBUG,
        log_frequencies={"on_timestep_end": 1},
        to_file=True,
        log_name=f"log_{test.test_ID}",
    ),
    eval_env,
    hook="on_episode_end",
    frequency=10,  # eval once every 10 episodes
    eval_immediately=True,
    deterministic=True,
    raises=False,
)
# evaluate train
agent.train(env=env, episodes=test.num_episodes, seed=1, raises=False)

# extract data
TD = np.squeeze(agent.td_errors)
if len(env.observations) > 0:
    X = np.hstack([env.observations[i].squeeze().T for i in range(test.num_episodes)]).T
    U = np.hstack([env.actions[i].squeeze().T for i in range(test.num_episodes)]).T
    R = np.hstack([env.rewards[i].squeeze().T for i in range(test.num_episodes)]).T
else:
    X = np.squeeze(env.ep_observations)
    U = np.squeeze(env.ep_actions)
    R = np.squeeze(env.ep_rewards)

print(f"Return = {sum(R.squeeze())}")

R_eps = [sum(R[ep_len * i : ep_len * (i + 1)]) for i in range(test.num_episodes)]
TD_eps = [
    sum(TD[ep_len * i : ep_len * (i + 1)]) / ep_len for i in range(test.num_episodes)
]
# generate output
y = np.asarray([output_true(X[k, :]) for k in range(X.shape[0])]).squeeze()
d = env.disturbance_profile_data

if PLOT:
    plot_greenhouse(X, U, y, d, TD, R, test.num_episodes, ep_len)

param_dict = {}
for key, val in agent.updates_history.items():
    param_dict[key] = val

identifier = test.test_ID
if STORE_DATA:
    with open(
        f"{identifier}.pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(y, file)
        pickle.dump(d, file)
        pickle.dump(R, file)
        pickle.dump(TD, file)
        pickle.dump(param_dict, file)
