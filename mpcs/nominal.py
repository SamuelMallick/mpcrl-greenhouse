from typing import Literal

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from mpcrl.util.seeding import RngType

from greenhouse.model import (
    Model,
    euler_step,
    get_control_bounds,
    get_model_details,
    multi_sample_output,
    multi_sample_step,
    output,
    rk4_step,
)


class NominalMpc(Mpc[cs.SX]):
    """Non-linear MPC for greenhouse control."""

    def __init__(
        self,
        greenhouse_model: Model,
        prediction_horizon: int = 6 * 4,
        prediction_model: Literal["euler", "rk4"] = "rk4",
        correct_model: bool = True,
        perturb_list: list[int] | None = None,
        np_random: RngType = None,
    ) -> None:
        # define some constants
        nx, nu, nd, _, _ = get_model_details()
        u_min, u_max, du_lim = get_control_bounds()
        w = np.full((1, nx), 1e3)  # penalty on constraint violations
        c_u = np.array([10, 1, 1])  # penalty on each control signal
        c_y = np.array([1e3])  # reward on yield

        # initialize base mpc
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, prediction_horizon=prediction_horizon)
        N = self.prediction_horizon

        # variables (state, action, dist, slack)
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu, lb=u_min, ub=u_max)
        self.disturbance("d", nd)
        s, _, _ = self.variable("s", (nx, N + 1), lb=0)  # slack vars

        # get either true or perturbed parameters
        if correct_model:
            mdl_params = greenhouse_model.get_true_parameters()
        else:
            if perturb_list is None:
                perturb_list = list(range(greenhouse_model.n_params))
            mdl_params = greenhouse_model.get_perturbed_parameters(
                perturb_list, np_random=np_random
            )

        # dynamics
        if prediction_model == "euler":
            model = lambda x, u, d: euler_step(x, u, d, mdl_params)
        else:  # if prediction_model == "rk4"
            model = lambda x, u, d: rk4_step(x, u, d, mdl_params)
        self.set_dynamics(lambda x, u, d: model(x, u, d), n_in=3, n_out=1)

        # other constraints
        for k in range(N + 1):
            # output constraints
            y_min_k = self.parameter(f"y_min_{k}", (nx, 1))
            y_max_k = self.parameter(f"y_max_{k}", (nx, 1))
            y_k = output(x[:, k], mdl_params)
            self.constraint(f"y_min_{k}", y_k, ">=", y_min_k - s[:, k])
            self.constraint(f"y_max_{k}", y_k, "<=", y_max_k + s[:, k])

            if 1 < k < N:
                # control change constraints
                self.constraint(f"du_min_{k}", u[:, k] - u[:, k - 1], "<=", du_lim)
                self.constraint(f"du_max_{k}", u[:, k] - u[:, k - 1], ">=", -du_lim)

        # objective
        obj = 0
        for k in range(N):
            # control action cost
            for j in range(nu):
                obj += c_u[j] * u[j, k]
            # constraint violation cost
            obj += w @ s[:, [k]]
        obj += w @ s[:, [N]]
        # yield terminal reward
        y_N = output(x[:, N], mdl_params)
        obj += -c_y * y_N[0]
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
