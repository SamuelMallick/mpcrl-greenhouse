from typing import Literal

import casadi as cs

# import networkx as netx
import numpy as np
from csnlp import Nlp
from csnlp.wrappers import Mpc
from mpcrl.util.seeding import RngType

from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model
from sims.configs.default import Test


class LearningMpc(Mpc[cs.SX]):
    """Non-linear MPC for greenhouse control."""

    def __init__(
        self,
        greenhouse_env: LettuceGreenHouse,
        test: Test,
        prediction_horizon: int = 6 * 4,
        prediction_model: Literal["euler", "rk4"] = "rk4",
        np_random: RngType = None,
    ) -> None:
        """Initialize the learning-based MPC for greenhouse control.

        Parameters
        ----------
        greenhouse_env : LettuceGreenHouse
            The greenhouse environment.
        test : Test
            The test configuration. Contains all learning hyper-parameters and MPC parameters.
        prediction_horizon : int, optional
            The prediction horizon, by default 6 * 4.
        prediction_model : Literal["euler", "rk4"], optional
            The prediction model to use, by default "rk4".
        np_random : RngType, optional
            The random number generator, by default None.
        """
        nx, nu, nd, ts = (
            greenhouse_env.nx,
            greenhouse_env.nu,
            greenhouse_env.nd,
            greenhouse_env.ts,
        )
        u_min, u_max, du_lim = Model.get_u_min(), Model.get_u_max(), Model.get_du_lim()
        # initialize base mpc
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, prediction_horizon=prediction_horizon)
        N = self.prediction_horizon
        self.discount_factor = test.discount_factor

        # learnable parameters
        learnable_pars_init = test.learnable_pars_init
        p = Model.get_perturbed_parameters(
            test.p_perturb
        )  # test.p_perturb contains the indexes of parameters to perturb
        for idx in test.p_learn:  # p_learn contains the indexes of parameters to learn
            learnable_pars_init[f"p_{idx}"] = np.asarray(p[idx])

        # fixed parameters
        fixed_pars = test.fixed_pars
        fixed_pars["d"] = np.zeros((nd, N))
        for k in range(N + 1):
            fixed_pars[f"y_min_{k}"] = np.zeros((nx,))
            fixed_pars[f"y_max_{k}"] = np.zeros((nx,))

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
