from typing import Literal

import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc

from envs.model import (
    euler_perturbed,
    euler_true,
    get_control_bounds,
    get_initial_perturbed_p,
    get_model_details,
    output_perturbed,
    output_true,
    rk4_perturbed,
    rk4_true,
)

np.random.seed(1)

nx, nu, nd, ts, steps_per_day = get_model_details()
u_min, u_max, du_lim = get_control_bounds()

c_u = np.array([10, 1, 1])  # penalty on each control signal
c_y = np.array([1e3])  # reward on yield
w = 1e3 * np.ones((1, nx))  # penalty on constraint violations


class NominalMpc(Mpc[cs.SX]):
    """Non-linear MPC for greenhouse control."""

    horizon = 6 * 4  # prediction horizon
    discount_factor = 1

    def __init__(
        self,
        prediction_model: Literal["euler", "rk4"] = "rk4",
        correct_model: bool = True,
        perturb_list: list[int] = [],
    ) -> None:
        N = self.horizon
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, N)

        # variables (state, action, dist, slack)
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu, lb=u_min, ub=u_max)
        self.disturbance("d", nd)
        s, _, _ = self.variable("s", (nx, N + 1), lb=0)  # slack vars

        # dynamics
        if prediction_model == "euler":
            if correct_model:
                model = lambda x, u, d: euler_true(x, u, d)
            else:
                if len(perturb_list) == 0:
                    perturb_list = [i for i in range(len(get_initial_perturbed_p()))]
                model = lambda x, u, d: euler_perturbed(x, u, d, perturb_list)
        elif prediction_model == "rk4":
            if correct_model:
                model = lambda x, u, d: rk4_true(x, u, d)
            else:
                if len(perturb_list) == 0:
                    perturb_list = [i for i in range(len(get_initial_perturbed_p()))]
                model = lambda x, u, d: rk4_perturbed(x, u, d, perturb_list)
        else:
            raise RuntimeError(
                f"{prediction_model} is not a valid prediction model option."
            )
        self.set_dynamics(lambda x, u, d: model(x, u, d), n_in=3, n_out=1)

        # output function
        if correct_model:
            output = lambda x: output_true(x)
        else:
            if len(perturb_list) == 0:
                perturb_list = [i for i in range(len(get_initial_perturbed_p()))]
            output = lambda x: output_perturbed(x, perturb_list)

        # other constraints
        y_min_list = [self.parameter(f"y_min_{k}", (nx, 1)) for k in range(N + 1)]
        y_max_list = [self.parameter(f"y_max_{k}", (nx, 1)) for k in range(N + 1)]

        y_0 = output(x[:, [0]])
        self.constraint(f"y_min_0", y_0, ">=", y_min_list[0] - s[:, [0]])
        self.constraint(f"y_max_0", y_0, "<=", y_max_list[0] + s[:, [0]])
        for k in range(1, N):
            # control change constraints
            self.constraint(f"du_geq_{k}", u[:, [k]] - u[:, [k - 1]], "<=", du_lim)
            self.constraint(f"du_leq_{k}", u[:, [k]] - u[:, [k - 1]], ">=", -du_lim)

            # output constraints
            y_k = output(x[:, [k]])
            self.constraint(f"y_min_{k}", y_k, ">=", y_min_list[k] - s[:, [k]])
            self.constraint(f"y_max_{k}", y_k, "<=", y_max_list[k] + s[:, [k]])

        y_N = output(x[:, [N]])
        self.constraint(f"y_min_{N}", y_N, ">=", y_min_list[N] - s[:, [N]])
        self.constraint(f"y_max_{N}", y_N, "<=", y_max_list[N] + s[:, [N]])

        obj = 0
        for k in range(N):
            obj += (self.discount_factor**k) * w @ s[:, [k]]
            for j in range(nu):
                obj += (self.discount_factor**k) * c_u[j] * u[j, k]

        obj += (self.discount_factor**N) * w @ s[:, [N]]
        obj += -c_y * y_N[0]
        self.minimize(obj)

        # TODO add in stepwise weight increase as optional obj

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
