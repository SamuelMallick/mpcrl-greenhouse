import importlib
import logging
import pickle
import sys
from typing import Literal

import casadi as cs
import numpy as np
from gymnasium.wrappers import TimeLimit

# import networkx as netx
from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.wrappers.agents import Evaluate, Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from agents.greenhouse_agent import GreenhouseLearningAgent
from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model
from mpcs.learning import LearningMpc
from sims.configs.default import DefaultTest
from utils.plot import plot_greenhouse


def load_test(config_file: str = None) -> DefaultTest:
    """Utility to load the test configuration from command line, or a default one."""
    if config_file is None:
        if len(sys.argv) > 1:
            config_file = sys.argv[1]
    if config_file is None:
        return DefaultTest()
    mod = importlib.import_module(f"sims.configs.{config_file}")
    return mod.Test()


def run_q_learning(
    test: DefaultTest, plot: bool = False, store: bool = False, seed: int = 1
) -> dict[str, str | np.ndarray]:
    """Runs the Q-learning algorithm on the greenhouse environment."""
    np_random = np.random.default_rng(seed)

    episode_len = test.ep_len
    train_env = MonitorEpisodes(
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
        greenhouse_env=train_env,
        test=test,
        prediction_model=prediction_model,
        np_random=np_random,
        constrain_control_rate=True,
    )
    param_bounds = (
        Model.get_learnable_parameter_bounds()
    )  # includes bounds just on model parameters
    param_bounds.update(
        test.learn_bounds
    )  # ad dalso the bounds on other learnable parameters
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
        env_reset_options={"initial_day": test.initial_day}
        if test.disturbance_type == "single"
        else {},
        seed=1,
    )
    # evaluate train
    agent.train(
        env=train_env,
        episodes=test.num_episodes,
        seed=1,
        raises=False,
        env_reset_options={"initial_day": test.initial_day}
        if test.disturbance_type == "single"
        else {},
    )

    # extract data
    TD = agent.td_errors
    TD = np.asarray(TD).reshape(test.num_episodes, -1)
    param_dict = {}
    for key, val in agent.updates_history.items():
        # repeat the first value as first skip_first updates are not performed
        temp = [val[0]] * test.skip_first
        # take index from 1 as first valeu is prior to any updates
        val = [*temp, *val[1:]]
        param_dict[key] = np.asarray(val).reshape(test.num_episodes, -1)

    # from train env
    X_tr = np.asarray(train_env.observations)
    U_tr = np.asarray(train_env.actions).squeeze(-1)
    R_tr = np.asarray(train_env.rewards)
    d_tr = np.asarray(train_env.disturbance_profiles_all_episodes).transpose(0, 2, 1)

    X_ev = np.asarray(eval_env.observations)
    U_ev = np.asarray(eval_env.actions).squeeze(-1)
    R_ev = np.asarray(eval_env.rewards)
    d_ev = np.asarray(eval_env.disturbance_profiles_all_episodes).transpose(0, 2, 1)
    if plot:  # plot training data
        plot_greenhouse(X_tr, U_tr, d_tr, R_tr, TD)

    identifier_train = test.test_ID + "_train"
    identifier_eval = test.test_ID + "_eval"
    data_train = {
        "name": identifier_train,
        "X": X_tr,
        "U": U_tr,
        "R": R_tr,
        "d": d_tr,
        "TD": TD,
        "param_dict": param_dict,
    }
    data_eval = {"name": identifier_eval, "X": X_ev, "U": U_ev, "R": R_ev, "d": d_ev}
    if store:
        with open(f"{identifier_train}.pkl", "wb") as file:
            pickle.dump(data_train, file)
        with open(f"{identifier_eval}.pkl", "wb") as file:
            pickle.dump(data_eval, file)
    return data_train, data_eval


if __name__ == "__main__":
    STORE_DATA = True
    PLOT = True
    run_q_learning(load_test(), store=STORE_DATA, plot=PLOT)
