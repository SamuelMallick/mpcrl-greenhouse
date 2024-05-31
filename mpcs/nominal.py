from typing import Literal

import casadi as cs
from csnlp import Nlp
from csnlp.wrappers import Mpc
from mpcrl.util.seeding import RngType

from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model


class NominalMpc(Mpc[cs.SX]):
    """Non-linear nominal MPC for greenhouse control."""

    def __init__(
        self,
        greenhouse_env: LettuceGreenHouse,
        prediction_horizon: int = 6 * 4,
        cost_parameters_dict: dict = {},
        prediction_model: Literal["euler", "rk4"] = "rk4",
        correct_model: bool = True,
        perturb_list: list[int] | None = None,
        np_random: RngType = None,
    ) -> None:
        """Initialize the nominal MPC for greenhouse control.

        Parameters
        ----------
        greenhouse_env : LettuceGreenHouse
            The greenhouse environment.
        prediction_horizon : int, optional
            The prediction horizon, by default 6 * 4.
        cost_parameters_dict : dict, optional
            The cost parameters dictionary, by default {} and
            the cost parameters of the environment are used.
        prediction_model : Literal["euler", "rk4"], optional
            The prediction model to use, by default "rk4".
        correct_model : bool, optional
            Whether to use the correct model, by default True.
        perturb_list : list[int] | None, optional
            The list of parameters to perturb, by default None.
            If correct_model is False and this is None, all parameters are perturbed.
        """
        nx, nu, nd, ts = (
            greenhouse_env.nx,
            greenhouse_env.nu,
            greenhouse_env.nd,
            greenhouse_env.ts,
        )
        u_min, u_max, du_lim = Model.get_u_min(), Model.get_u_max(), Model.get_du_lim()
        if not cost_parameters_dict:
            cost_parameters_dict = greenhouse_env.get_cost_parameters()
        c_u = cost_parameters_dict["c_u"]  # penalty on each control signal
        c_y = cost_parameters_dict["c_y"]  # reward on yield
        w = cost_parameters_dict["w"]  # penalty on constraint violations

        # initialize base mpc
        nlp = Nlp[cs.SX](debug=False)
        super().__init__(nlp, prediction_horizon=prediction_horizon)
        N = self.prediction_horizon

        # variables (state, action, dist, slack)
        x, _ = self.state("x", nx)
        u, _ = self.action("u", nu, lb=u_min.reshape(-1, 1), ub=u_max.reshape(-1, 1))
        self.disturbance("d", nd)
        s, _, _ = self.variable("s", (nx, N + 1), lb=0)  # slack vars

        # get either true or perturbed parameters
        if correct_model:
            p = Model.get_true_parameters()
        else:
            if perturb_list is None:
                perturb_list = list(range(Model.n_params))
            p = Model.get_perturbed_parameters(perturb_list, np_random=np_random)

        # dynamics
        if prediction_model == "euler":
            model = lambda x, u, d: Model.euler_step(x, u, d, p, ts)
        else:  # if prediction_model == "rk4"
            model = lambda x, u, d: Model.rk4_step(x, u, d, p, ts)
        self.set_dynamics(lambda x, u, d: model(x, u, d), n_in=3, n_out=1)

        # other constraints
        for k in range(N + 1):
            # output constraints
            y_min_k = self.parameter(f"y_min_{k}", (nx, 1))
            y_max_k = self.parameter(f"y_max_{k}", (nx, 1))
            y_k = Model.output(x[:, k], p)
            self.constraint(f"y_min_{k}", y_k, ">=", y_min_k - s[:, k])
            self.constraint(f"y_max_{k}", y_k, "<=", y_max_k + s[:, k])

        for k in range(1, N):
            # control variation constraints
            self.constraint(f"du_min_{k}", u[:, k] - u[:, k - 1], "<=", du_lim)
            self.constraint(f"du_max_{k}", u[:, k] - u[:, k - 1], ">=", -du_lim)

        # objective
        obj = 0
        for k in range(N):
            # control action cost
            for j in range(nu):
                obj += c_u[j] * u[j, k]
            # constraint violation cost
            obj += cs.dot(w, s[:, k])
        obj += cs.dot(w, s[:, N])
        # yield terminal reward
        y_N = Model.output(x[:, N], p)
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
