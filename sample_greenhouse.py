import datetime
import logging
import pickle

import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.agents import Log
from mpcrl.wrappers.envs import MonitorEpisodes

from envs.env import GreenhouseSampleAgent, LettuceGreenHouse
from envs.model import (
    get_control_bounds,
    get_model_details,
    multi_sample_output,
    multi_sample_rk4_step,
    output_real,
)
from plot_green import plot_greenhouse

np.random.seed(1)

STORE_DATA = False
PLOT = True

nx, nu, nd, ts, _ = get_model_details()
u_min, u_max, du_lim = get_control_bounds()

c_u = np.array([10, 1, 1])  # penalty on each control signal
c_y = np.array([10e3])  # reward on yield


class SampleBasedMpc(Mpc[cs.SX]):
    """Non-linear Sample Based Robust MPC for greenhouse control."""

    horizon = 6 * 4  # prediction horizon
    discount_factor = 1

    def __init__(self, Ns) -> None:
        N = self.horizon
        self.Ns = Ns
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, N)

        # variables (state, action, slack)
        # state needs to be done manually as we have one state per scenario
        x = self.nlp.variable(
            "x",
            (nx * Ns, self._prediction_horizon + 1),
            lb=cs.vertcat(*[[0], [0], [-float("inf")], [0]] * Ns),
        )[0]
        x0 = self.nlp.parameter("x_0", (nx, 1))
        self.nlp.constraint("x_0", x[:, 0], "==", cs.repmat(x0, Ns, 1))
        self._states["x"] = x
        self._initial_states["x_0"] = x0
        u, _ = self.action("u", nu, lb=u_min, ub=u_max)
        self.disturbance("d", nd)

        # dynamics
        self.set_dynamics(
            lambda x, u, d: multi_sample_rk4_step(x, u, d, Ns), n_in=3, n_out=1
        )

        # other constraints
        y_min_list = [self.parameter(f"y_min_{k}", (nx * Ns, 1)) for k in range(N + 1)]
        y_max_list = [self.parameter(f"y_max_{k}", (nx * Ns, 1)) for k in range(N + 1)]
        for k in range(1, N):
            # control change constraints
            self.constraint(f"du_geq_{k}", u[:, [k]] - u[:, [k - 1]], "<=", du_lim)
            self.constraint(f"du_leq_{k}", u[:, [k]] - u[:, [k - 1]], ">=", -du_lim)

            # output constraints
            y_k = multi_sample_output(x[:, [k]], Ns)
            self.constraint(f"y_min_{k}", y_k, ">=", y_min_list[k])
            self.constraint(f"y_max_{k}", y_k, "<=", y_max_list[k])

        y_N = multi_sample_output(x[:, [N]], Ns)
        self.constraint(f"y_min_{N}", y_N, ">=", y_min_list[N])
        self.constraint(f"y_max_{N}", y_N, "<=", y_max_list[N])

        obj = 0
        for k in range(N):
            for j in range(nu):
                obj += Ns * c_u[j] * u[j, k]
        for i in range(Ns):
            y_N_i = y_N[nx * i : nx * (i + 1), :]
            obj += -c_y * y_N_i[0]
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


days = 2
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps
env = MonitorEpisodes(
    TimeLimit(LettuceGreenHouse(days_to_grow=days), max_episode_steps=int(ep_len))
)
num_episodes = 1

TD = []

sample_mpc = SampleBasedMpc(Ns=2)
agent = Log(
    GreenhouseSampleAgent(sample_mpc, {}),
    level=logging.DEBUG,
    log_frequencies={"on_timestep_end": 1},
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
y = np.asarray([output_real(X[k, :]) for k in range(X.shape[0])]).squeeze()
d = env.disturbance_profile_data

if PLOT:
    plot_greenhouse(X, U, y, d, TD, R, num_episodes, ep_len)

param_dict = {}
identifier = "e_3_50"
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
