import importlib
import logging
import os
import pickle
import sys
from typing import Literal

sys.path.append(os.getcwd())

import casadi as cs
import numpy as np
from gymnasium.wrappers import TimeLimit

# import networkx as netx
from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from agents.greenhouse_agent import GreenhouseLearningAgent
from greenhouse.env import LettuceGreenHouse
from mpcs.learning import LearningMpc
from utils.plot import plot_greenhouse

np_random = np.random.default_rng(1)

STORE_DATA = True
PLOT = False

# get the env and the final learned parameters from a specific test
test_num = 80
mod = importlib.import_module(f"sims.configs.test_{test_num}")
test = mod.Test()
file_name = f"results/test_{test_num}_train.pkl"
with open(
    file_name,
    "rb",
) as file:
    data = pickle.load(file)
params = {
    key: val[-1] for key, val in data["param_dict"].items()
}  # take final value for learned parameters

episode_len = test.ep_len
eval_env = MonitorEpisodes(
    TimeLimit(
        LettuceGreenHouse(
            growing_days=test.num_days,
            model_type=test.base_model,
            cost_parameters_dict=test.rl_cost,
            disturbance_profiles_type=test.disturbance_type,
            noisy_disturbance=test.noisy_disturbance,
            clip_action_variation=test.clip_action_variation,
        ),
        max_episode_steps=int(episode_len),
    )
)

prediction_model: Literal["euler", "rk4"] = "rk4"
mpc = LearningMpc(
    greenhouse_env=eval_env,
    test=test,
    prediction_model=prediction_model,
    np_random=np_random,
    constrain_control_rate=True,
)
# assert that the parameters loaded are the same as the ones in the MPC class
if set(params.keys()) != set(mpc.learnable_pars_init.keys()):
    raise ValueError("Learned parameters do not match the MPC class parameters")

learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
        for name, val in params.items()
    )
)

agent = Log(  # type: ignore[var-annotated]
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
    log_frequencies={"on_timestep_end": 100},
    to_file=True,
    log_name=f"log_eval_{test.test_ID}",
)
# evaluate train
agent.evaluate(
    env=eval_env,
    episodes=100,
    seed=1,
    raises=False,
    env_reset_options={
        "initial_day": test.initial_day,
        "noise_coeff": test.noise_coeff if test.noisy_disturbance else 1.0,
    }
    if test.disturbance_type == "single"
    else {},
)

# extract data
TD = agent.td_errors
TD = np.asarray(TD).reshape(test.num_episodes, -1)
param_dict = {}
for key, val in agent.updates_history.items():
    temp = [
        val[0]
    ] * test.skip_first  # repeat the first value as first skip_first updates are not performed
    val = [*temp, *val[1:]]  # take index from 1 as first valeu is prior to any updates
    param_dict[key] = np.asarray(val).reshape(test.num_episodes, -1)


X_tr = np.asarray(eval_env.observations)
U_tr = np.asarray(eval_env.actions).squeeze(-1)
R_tr = np.asarray(eval_env.rewards)
d_tr = np.asarray(eval_env.disturbance_profiles_all_episodes).transpose(0, 2, 1)

print(f"Average solve time = {np.mean(agent.solve_times)}")

if PLOT:  # plot training data
    plot_greenhouse(X_tr, U_tr, d_tr, R_tr, TD)

identifier_ev = test.test_ID + "_eval_final"
if STORE_DATA:
    with open(
        f"{identifier_ev}.pkl",
        "wb",
    ) as file:
        pickle.dump(
            {
                "name": identifier_ev,
                "X": X_tr,
                "U": U_tr,
                "R": R_tr,
                "d": d_tr,
                "TD": TD,
                "param_dict": param_dict,
            },
            file,
        )
