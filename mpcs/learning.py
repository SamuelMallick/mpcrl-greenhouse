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

        # create parameters
        # cost parameters
        V0 = self.parameter("V0", (1,))
        c_u = self.parameter("c_u", (nu,))
        c_dy = self.parameter("c_dy", (1,))
        c_y = self.parameter("c_y", (1,))
        y_fin = self.parameter("y_fin", (1,))
        # constraint violation parameters
        w = self.parameter("w", (4,))
        olb = self.parameter("olb", (4,))
        oub = self.parameter("oub", (4,))
        # dynamics parameters
        p = [self.parameter(f"p_{i}", (1,)) for i in range(Model.n_params)]

        p_values = Model.get_perturbed_parameters(test.p_perturb)

        # parameters initial values dictionaries
        learnable_pars_init = test.learnable_pars_init
        fixed_pars = test.fixed_pars
        # test.p_perturb contains the indexes of parameters to perturb
        for i in range(
            Model.n_params
        ):  # p_learn contains the indexes of parameters to learn
            if i in test.p_learn:
                learnable_pars_init[f"p_{i}"] = np.asarray(p_values[i])
            else:
                fixed_pars[f"p_{i}"] = np.asarray(p_values[i])
        fixed_pars["d"] = np.zeros((nd, N))
        for k in range(N + 1):
            fixed_pars[f"y_min_{k}"] = np.zeros((nx,))
            fixed_pars[f"y_max_{k}"] = np.zeros((nx,))
        self.learnable_pars_init = learnable_pars_init
        self.fixed_pars = fixed_pars

        # variables (state, action, dist, slack)
        x, _ = self.state("x", nx, lb=0, ub=1e3)
        u, _ = self.action("u", nu, lb=u_min.reshape(-1, 1), ub=u_max.reshape(-1, 1))
        self.disturbance("d", nd)
        s, _, _ = self.variable("s", (nx, N + 1), lb=0)  # slack vars
        p = cs.vertcat(*p)  # stack the parameters for the dynamics

        # dynamics
        if prediction_model == "euler":
            model = lambda x, u, d: Model.euler_step(x, u, d, p, ts)
        else:
            model = lambda x, u, d: Model.rk4_step(x, u, d, p, ts)
        self.set_dynamics(lambda x, u, d: model(x, u, d), n_in=3, n_out=1)

        # other constraints
        y = [Model.output(x[:, k], p) for k in range(N + 1)]
        for k in range(N + 1):
            # output constraints
            y_min_k = self.parameter(f"y_min_{k}", (nx, 1))
            y_max_k = self.parameter(f"y_max_{k}", (nx, 1))
            self.constraint(f"y_min_{k}", y[k], ">=", (1 + olb) * y_min_k - cs.dot((1 + oub) * y_max_k - (1 + olb) * y_min_k, s[:, k]))
            self.constraint(f"y_max_{k}", y[k], "<=", (1 + oub) * y_max_k + cs.dot((1 + oub) * y_max_k - (1 + olb) * y_min_k, s[:, k]))

        for k in range(1, N):
            # control variation constraints
            self.constraint(f"du_min_{k}", u[:, k] - u[:, k - 1], "<=", du_lim)
            self.constraint(f"du_max_{k}", u[:, k] - u[:, k - 1], ">=", -du_lim)

        # objective
        obj = V0
        # penalize control effort
        for k in range(N):
            for j in range(nu):
                obj += (self.discount_factor**k) * c_u[j] * u[j, k]

        # penalize constraint violations
        for k in range(N + 1):
            obj += (self.discount_factor**k) * cs.dot(w, s[:, k])

        # reward step wise weight increase
        for k in range(1, N + 1):
            obj += -(self.discount_factor**k) * c_dy * (y[k][0] - y[k - 1][0])

        # reward final weight a.k.a terminal cost
        obj += (self.discount_factor ** (N + 1)) * c_dy * c_y * (y_fin - y[N][0])
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
