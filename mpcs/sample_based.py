from typing import Literal

import casadi as cs
import numpy as np
from csnlp import Nlp
from csnlp import multistart as ms
from csnlp.wrappers import Mpc
from mpcrl.util.seeding import RngType

from greenhouse.model import (
    get_control_bounds,
    get_model_details,
    multi_sample_output,
    multi_sample_step,
)


class SampleBasedMpc(Mpc[cs.SX]):
    """Non-linear Sample Based Robust MPC for greenhouse control."""

    def __init__(
        self,
        n_samples: int,
        prediction_horizon: int = 6 * 4,
        prediction_model: Literal["euler", "rk4"] = "rk4",
        multistarts: int = 1,
        np_random: RngType = None,
    ) -> None:
        # define some constants
        nx, nu, nd, _, _ = get_model_details()
        u_min, u_max, du_lim = get_control_bounds()
        w = np.full((1, nx * n_samples), 1e3)  # penalty on constraint violations
        c_u = np.array([10, 1, 1])  # penalty on each control signal
        c_y = np.array([1e3])  # reward on yield

        # initialize base mpc
        nlp: Nlp[cs.SX] = (
            Nlp() if multistarts == 1 else ms.ParallelMultistartNlp(starts=multistarts)
        )
        super().__init__(nlp, prediction_horizon=prediction_horizon)
        self.n_samples = n_samples
        N = self.prediction_horizon

        # state needs to be done manually as we have one state per scenario
        # TODO: remove hacking of stacked state variables
        x = self.nlp.variable(
            "x",
            (nx * n_samples, N + 1),
            lb=cs.vertcat(*[[0], [0], [-float("inf")], [0]] * n_samples),
        )[0]
        x0 = self.nlp.parameter("x_0", (nx, 1))
        self.nlp.constraint("x_0", x[:, 0], "==", cs.repmat(x0, n_samples, 1))
        self._states["x"] = x
        self._initial_states["x_0"] = x0
        u, _ = self.action("u", nu, lb=u_min, ub=u_max)
        self.disturbance("d", nd)
        s, _, _ = self.variable("s", (nx * n_samples, N + 1), lb=0)  # slack vars

        # TODO: genereate parameter samples here and pass them to `multi_sample_step`

        # dynamics
        self.set_dynamics(
            lambda x, u, d: multi_sample_step(x, u, d, n_samples, prediction_model),
            n_in=3,
            n_out=1,
        )

        # other constraints
        for k in range(N + 1):
            # output constraints
            y_min_k = self.parameter(f"y_min_{k}", (nx * n_samples, 1))
            y_max_k = self.parameter(f"y_max_{k}", (nx * n_samples, 1))
            y_k = multi_sample_output(x[:, k], n_samples)
            self.constraint(f"y_min_{k}", y_k, ">=", y_min_k - s[:, k])
            self.constraint(f"y_max_{k}", y_k, "<=", y_max_k + s[:, k])

            if 1 < k < N:
                # control change constraints
                self.constraint(f"du_min_{k}", u[:, k] - u[:, k - 1], "<=", du_lim)
                self.constraint(f"du_max_{k}", u[:, k] - u[:, k - 1], ">=", -du_lim)

        # objective
        # TODO: see if possible to remove loops
        obj = 0
        for k in range(N):
            # control action cost
            for j in range(nu):
                obj += n_samples * c_u[j] * u[j, k]
            # constraint violation cost
            obj += w @ s[:, k]
        obj += w @ s[:, N]
        # yield terminal reward
        y_N = multi_sample_output(x[:, N], n_samples)
        for i in range(n_samples):
            y_N_i = y_N[nx * i : nx * (i + 1), :]
            obj += -c_y * y_N_i[0]
        self.minimize(obj)

        # solver
        opts = {
            "expand": True,
            "show_eval_warnings": False,
            "warn_initial_bounds": True,
            "print_time": False,
            "bound_consistency": True,
            "calc_lam_x": True,
            "calc_lam_p": False,
            "ipopt": {
                "sb": "yes",
                "print_level": 0,
                "max_iter": 2000,
                "print_user_options": "yes",
                "print_options_documentation": "no",
                "linear_solver": "ma57",  # spral
                "nlp_scaling_method": "gradient-based",
                "nlp_scaling_max_gradient": 10,
            },
        }
        self.init_solver(opts, solver="ipopt")

        # initialize multistart point generators (only if multistart is on)
        if multistarts <= 1:
            self.random_start_points = None
        else:
            # TODO: put the same bounds as in the NLP for x and u, and set
            # WarmStartStrategy(update_biases_for_random_points=False)
            bounds_and_size = {
                "x": (..., ..., x.shape),
                "u": (..., ..., u.shape),
            }
            self.random_start_points = ms.RandomStartPoints(
                {
                    n: ms.RandomStartPoint("uniform", *args)
                    for n, args in bounds_and_size.items()
                },
                multistarts - 1,
                np_random,
            )
