import datetime
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
from mpcrl.core.experience import ExperienceReplay
from mpcrl.core.schedulers import ExponentialScheduler
from mpcrl.wrappers.agents import Log, RecordUpdates
from mpcrl.wrappers.envs import MonitorEpisodes

from envs.env import (
    GreenhouseLearningAgent,
    LettuceGreenHouse,
)
from envs.model import (
    get_control_bounds,
    get_initial_perturbed_p,
    get_model_details,
    output_learnable,
    output_real,
    rk4_learnable,
)
from plot_green import plot_greenhouse

np.random.seed(1)

STORE_DATA = True
PLOT = False

nx, nu, nd, ts = get_model_details()
u_min, u_max, du_lim = get_control_bounds()

c_u = np.array([10, 1, 1])  # penalty on each control signal
c_y = np.array([10e3])  # reward on yield

class LearningMpc(Mpc[cs.SX]):
    """Non-linear MPC for greenhouse control."""

    horizon = 6 * 4  # prediction horizon
    discount_factor = 1  # TODO add the gamma scaling into the local cost

    p_indexes = [
        0,
        2,
        3,
        5,
    ]  # list of indexes in p to which learnable parameters correspond
    w = 1e3 * np.ones((1, nx))  # penalty on constraint violations

    # par inits
    learnable_pars_init = {
        "V0": np.zeros((1,)),
        "c_u": c_u,
        "c_y": c_y,
    }
    p_init = get_initial_perturbed_p()
    for i in range(4):
        learnable_pars_init[f"p_{i}"] = np.array([p_init[p_indexes[i]]])

    fixed_pars_init = {"d": np.zeros((nx, horizon))}
    for k in range(horizon + 1):
        fixed_pars_init[f"y_min_{k}"] = np.zeros((nx, 1))
        fixed_pars_init[f"y_max_{k}"] = np.zeros((nx, 1))

    def __init__(self) -> None:
        N = self.horizon
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, N)

        # variables (state, action, slack)
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu, lb=u_min, ub=u_max)
        self.disturbance("d", nd)
        s, _, _ = self.variable("s", (nx, N), lb=0)  # slack vars

        # init parameters
        V0_learn = self.parameter("V0", (1,))
        c_u_learn = self.parameter("c_u", (nu,))
        c_y_learn = self.parameter("c_y", (1,))

        p_learnable = [self.parameter(f"p_{i}", (1,)) for i in range(4)]

        # dynamics
        self.set_dynamics(
            lambda x, u, d: rk4_learnable(x, u, d, p_learnable, self.p_indexes),
            n_in=3,
            n_out=1,
        )

        # other constraints
        y_min_list = [self.parameter(f"y_min_{k}", (nx, 1)) for k in range(N + 1)]
        y_max_list = [self.parameter(f"y_max_{k}", (nx, 1)) for k in range(N + 1)]
        for k in range(1, N):
            # control change constraints
            self.constraint(f"du_geq_{k}", u[:, [k]] - u[:, [k - 1]], "<=", du_lim)
            self.constraint(f"du_leq_{k}", u[:, [k]] - u[:, [k - 1]], ">=", -du_lim)

            # output constraints
            y_k = output_learnable(x[:, [k]], p_learnable, self.p_indexes)
            self.constraint(f"y_min_{k}", y_k, ">=", y_min_list[k] - s[:, [k]])
            self.constraint(f"y_max_{k}", y_k, "<=", y_max_list[k] + s[:, [k]])

        y_N = output_learnable(x[:, [N]], p_learnable, self.p_indexes)
        self.constraint(f"y_min_{N}", y_N, ">=", y_min_list[N] - s[:, [k]])
        self.constraint(f"y_max_{N}", y_N, "<=", y_max_list[N] + s[:, [k]])

        obj = V0_learn
        for k in range(N):
            for j in range(nu):
                obj += c_u_learn[j] * u[j, k]
            obj += self.w @ s[:, [k]]
        obj += -c_y_learn * y_N[0]
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


days = 40
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps
env = MonitorEpisodes(
    TimeLimit(LettuceGreenHouse(days_to_grow=days), max_episode_steps=int(ep_len))
)
num_episodes = 50
if len(sys.argv) > 1:
    num_episodes = int(sys.argv[1])

mpc = LearningMpc()
learnable_pars = LearnableParametersDict[cs.SX](
    (
        LearnableParameter(name, val.shape, val, sym=mpc.parameters[name])
        for name, val in mpc.learnable_pars_init.items()
    )
)

learning_rate = 1e-1
agent = Log(  # type: ignore[var-annotated]
    RecordUpdates(
        GreenhouseLearningAgent(
            mpc=mpc,
            learnable_parameters=learnable_pars,
            fixed_parameters=mpc.fixed_pars_init,
            discount_factor=mpc.discount_factor,
            update_strategy=ep_len,
            learning_rate=ExponentialScheduler(learning_rate, factor=1),
            hessian_type="approx",
            record_td_errors=True,
            exploration=None,
            experience=ExperienceReplay(
                maxlen=3 * ep_len,
                sample_size=int(1.5 * ep_len),
                include_latest=ep_len,
                seed=0,
            ),
        )
    ),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
)
# evaluate train
agent.train(env=env, episodes=num_episodes, seed=1, raises=False)

# extract data
TD = np.squeeze(agent.td_errors)
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
y = np.asarray([output_real(X[k, :]) for k in range(X.shape[0])]).squeeze()
d = env.disturbance_profile_data

if PLOT:
    plot_greenhouse(X, U, y, d, TD, R, num_episodes, ep_len)

param_dict = {}
for key, val in agent.updates_history.items():
    param_dict[key] = val

identifier = f"_V1_lr_{learning_rate}_ne_{num_episodes}_"
if STORE_DATA:
    with open(
        "green" + identifier + datetime.datetime.now().strftime("%d%H%M%S%f") + ".pkl",
        "wb",
    ) as file:
        pickle.dump(X, file)
        pickle.dump(U, file)
        pickle.dump(y, file)
        pickle.dump(d, file)
        pickle.dump(R, file)
        pickle.dump(TD, file)
        pickle.dump(param_dict, file)
