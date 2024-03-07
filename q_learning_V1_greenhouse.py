import importlib
import logging
import pickle
import sys

import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit
from mpcrl import LearnableParameter, LearnableParametersDict
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from envs.env import GreenhouseLearningAgent, LettuceGreenHouse
from envs.model import (
    euler_learnable,
    generate_parameters,
    get_control_bounds,
    get_initial_perturbed_p,
    get_model_details,
    get_p_learn_bounds,
    output_learnable,
    output_true,
    rk4_learnable,
)
from plot_green import plot_greenhouse

np.random.seed(1)

# if a config file passed on command line, otherwise use dedault config file
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    mod = importlib.import_module(f"test_configs.{config_file}")
    test = mod.Test()

    STORE_DATA = True
    PLOT = False
else:
    from test_configs.test_19 import Test

    test = Test()

    STORE_DATA = True
    PLOT = True

nx, nu, nd, ts, _ = get_model_details()
u_min, u_max, du_lim = get_control_bounds()

if hasattr(test, "perturb"):
    generate_parameters(test.perturb)
else:
    generate_parameters()


class LearningMpc(Mpc[cs.SX]):
    """Non-linear MPC for greenhouse control."""

    horizon = test.horizon
    discount_factor = test.discount_factor

    # add dynamics params to learnable pars init
    learnable_pars_init = test.learnable_pars_init
    p_perturb_full = get_initial_perturbed_p()
    for idx in test.p_learn:
        # use perturbed value as initial guess or true value
        learnable_pars_init[f"p_{idx}"] = (
            np.asarray(p_perturb_full[idx]) if idx in test.p_perturb else np.asarray(1)
        )

    # add disturbance prediction and output constraints to fixed pars
    fixed_pars = test.fixed_pars
    fixed_pars["d"] = np.zeros((nx, horizon))
    for k in range(horizon + 1):
        fixed_pars[f"y_min_{k}"] = np.zeros((nx, 1))
        fixed_pars[f"y_max_{k}"] = np.zeros((nx, 1))

    def __init__(self) -> None:
        N = self.horizon
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, N)

        # variables (state, action, dist, slack)
        x, _ = self.state("x", nx, lb=0, ub=1e3)
        u, _ = self.action("u", nu, lb=u_min, ub=u_max)
        self.disturbance("d", nd)
        s, _, _ = self.variable("s", (nx, N + 1), lb=0)  # slack vars

        # init parameters
        V0 = self.parameter("V0", (1,))
        c_u = self.parameter("c_u", (nu,))
        c_dy = self.parameter("c_dy", (1,))
        c_y = self.parameter("c_y", (1,))
        w = self.parameter("w", (1, 4))
        olb = self.parameter("olb", (4, 1))
        oub = self.parameter("oub", (4, 1))
        y_fin = self.parameter("y_fin", (1,))
        # build tuple of learnable params and their indexes
        p_learn_tuples = [
            (idx, self.parameter(f"p_{idx}", (1,))) for idx in test.p_learn
        ]

        # dynamics
        if test.prediction_model == "rk4":
            dynam = rk4_learnable
        elif test.prediction_model == "euler":
            dynam = euler_learnable
        else:
            raise ValueError(
                f"{test.prediction_model} is not a valid prediction model."
            )
        self.set_dynamics(
            lambda x, u, d: dynam(x, u, d, test.p_perturb, p_learn_tuples),
            n_in=3,
            n_out=1,
        )

        output = lambda x: output_learnable(x, test.p_perturb, p_learn_tuples)

        # other constraints
        y_min_list = [self.parameter(f"y_min_{k}", (nx, 1)) for k in range(N + 1)]
        y_max_list = [self.parameter(f"y_max_{k}", (nx, 1)) for k in range(N + 1)]
        y_k = [output(x[:, [0]])]

        self.constraint(f"y_min_0", y_k[0], ">=", (1 + olb) * y_min_list[0] - s[:, [0]])
        self.constraint(f"y_max_0", y_k[0], "<=", (1 + oub) * y_max_list[0] + s[:, [0]])
        for k in range(1, N):
            # control change constraints
            self.constraint(f"du_geq_{k}", u[:, [k]] - u[:, [k - 1]], "<=", du_lim)
            self.constraint(f"du_leq_{k}", u[:, [k]] - u[:, [k - 1]], ">=", -du_lim)

        for k in range(1, N + 1):
            y_k.append(output(x[:, [k]]))
            # output constraints
            self.constraint(
                f"y_min_{k}", y_k[k], ">=", (1 + olb) * y_min_list[k] - s[:, [k]]
            )
            self.constraint(
                f"y_max_{k}", y_k[k], "<=", (1 + oub) * y_max_list[k] + s[:, [k]]
            )

        obj = V0
        # penalize control effort
        for k in range(N):
            for j in range(nu):
                obj += (self.discount_factor**k) * c_u[j] * u[j, k]

        # penalize constraint violations
        for k in range(N + 1):
            obj += (self.discount_factor**k) * w @ s[:, [k]]

        # reward step wise weight increase
        for k in range(1, N + 1):
            obj += -(self.discount_factor**k) * c_dy * (y_k[k][0] - y_k[k - 1][0])

        # reward final weight a.k.a terminal cost
        obj += (self.discount_factor ** (N + 1)) * c_dy * c_y * (y_fin - y_k[N][0])

        self.minimize(obj)

        # solver
        opts = {
            "expand": True,
            "show_eval_warnings": True,
            "warn_initial_bounds": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            # "jit": True,
            # "jit_cleanup": True,
            "ipopt": {
                # "linear_solver": "ma97",
                # "linear_system_scaling": "mc19",
                # "nlp_scaling_method": "equilibration-based",
                "max_iter": 500,
                "sb": "yes",
                "print_level": 0,
            },
        }
        self.init_solver(opts, solver="ipopt")


ep_len = test.ep_len
env = MonitorEpisodes(
    TimeLimit(
        LettuceGreenHouse(
            days_to_grow=test.num_days, model_type=test.base_model, rl_cost=test.rl_cost
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
    log_frequencies={"on_timestep_end": 1000},
    to_file=True,
    log_name=f"log_{test.test_ID}",
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
