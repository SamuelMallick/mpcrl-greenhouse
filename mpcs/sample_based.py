from collections.abc import Callable, Sequence
from typing import Any, Literal

import casadi as cs
import numpy as np
import numpy.typing as npt
from csnlp import Nlp
from csnlp import multistart as ms
from csnlp.wrappers import Mpc
from csnlp.wrappers.mpc.scenario_based_mpc import ScenarioBasedMpc, _n
from mpcrl.util.seeding import RngType

from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model


class SampleBasedMpc(ScenarioBasedMpc[cs.SX]):
    """Non-linear Sample Based Robust MPC for greenhouse control. Uses the scenario
    approach with samples of unknown model parameters."""

    def __init__(
        self,
        n_samples: int,
        greenhouse_env: LettuceGreenHouse,
        prediction_horizon: int = 6 * 4,
        cost_parameters_dict: dict = {},
        prediction_model: Literal["euler", "rk4"] = "rk4",
        constrain_control_rate: bool = True,
        multistarts: int = 1,
        np_random: RngType = None,
    ) -> None:
        """Initialize the sample based robust MPC for greenhouse control.

        Parameters
        ----------
        n_samples : int
            The number of samples to use.
        greenhouse_env : LettuceGreenHouse
            The greenhouse environment.
        prediction_horizon : int, optional
            The prediction horizon, by default 6 * 4.
        cost_parameters_dict : dict, optional
            The cost parameters dictionary, by default {} and
            the cost parameters of the environment are used.
        prediction_model : Literal["euler", "rk4"], optional
            The prediction model to use, by default "rk4".
        costrain_control_rate : bool, optional
            Whether to constrain the control rate, by default False.
        multistarts : int, optional
            The number of multistarts to use for solving the NLPs, by default 1.
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
        if not cost_parameters_dict:
            cost_parameters_dict = greenhouse_env.get_cost_parameters()
        c_u = cost_parameters_dict["c_u"]  # penalty on each control signal
        c_y = cost_parameters_dict["c_y"]  # reward on yield
        w_y = cost_parameters_dict["w_y"]  # penalty on constraint violations

        # initialize base mpc
        nlp: Nlp[cs.SX] = (
            Nlp()
            if multistarts == 1
            else ms.ParallelMultistartNlp(
                starts=multistarts, parallel_kwargs={"n_jobs": multistarts}
            )
        )
        super().__init__(
            nlp, n_scenarios=n_samples, prediction_horizon=prediction_horizon
        )
        self._dynamics: Sequence[cs.Function] | None = None
        self.n_samples = n_samples
        N = self.prediction_horizon

        u_min = u_min.reshape(-1, 1)
        u_max = u_max.reshape(-1, 1)
        x, xs, _ = self.state("x", nx)  # NOTE: if infeasible, try to set bounds
        u, _ = self.action("u", nu, lb=u_min, ub=u_max)
        s = [
            self.variable(_n("s", i), (nx, N + 1), lb=0)[0] for i in range(n_samples)
        ]  # slack vars for each scenario
        self.disturbance("d", nd)

        p = [
            Model.get_perturbed_parameters(
                list(range(Model.n_params)), np_random=np_random
            )
            for _ in range(n_samples)
        ]
        if prediction_model == "euler":
            dynamics = [
                lambda x, u, d: Model.euler_step(x, u, d, p[i], ts)
                for i in range(n_samples)
            ]
        else:
            dynamics = [
                lambda x, u, d: Model.rk4_step(x, u, d, p[i], ts)
                for i in range(n_samples)
            ]
        self.set_dynamics(dynamics, n_in=3, n_out=1)

        # other constraints
        for k in range(N + 1):
            # same bounds on output for all scenarios, as bounds are determined by disturbances
            y_min_k = self.parameter(f"y_min_{k}", (nx, 1))
            y_max_k = self.parameter(f"y_max_{k}", (nx, 1))
            for i in range(n_samples):
                y_k = Model.output(xs[i][:, k], p[i])
                # dividing by the range here instead of in the objective as the y_min_k and y_max_k are calcualted here
                self.constraint(
                    _n(f"y_min_{k}", i),
                    y_k,
                    ">=",
                    y_min_k - s[i][:, k] / (y_max_k - y_min_k),
                )
                self.constraint(
                    _n(f"y_max_{k}", i),
                    y_k,
                    "<=",
                    y_max_k + s[i][:, k] / (y_max_k - y_min_k),
                )

        if constrain_control_rate:
            for k in range(1, N):
                # control variation constraints
                self.constraint(f"du_min_{k}", u[:, k] - u[:, k - 1], "<=", du_lim)
                self.constraint(f"du_max_{k}", u[:, k] - u[:, k - 1], ">=", -du_lim)

        # objective
        obj = 0
        for k in range(N):
            # control action cost
            for j in range(nu):
                obj += n_samples * c_u[j] * u[j, k]
            # constraint violation cost
            obj += sum(cs.dot(w_y, s[i][:, k]) for i in range(n_samples))
        obj += sum(cs.dot(w_y, s[i][:, N]) for i in range(n_samples))
        # yield terminal reward
        y_N = [Model.output(xs[i][:, N], p[i]) for i in range(n_samples)]
        obj += sum(-c_y * y_N[i][0] for i in range(n_samples))
        self.minimize(obj)

        # solver
        opts = {
            "expand": True,
            "show_eval_warnings": False,
            "warn_initial_bounds": True,
            "print_time": False,
            "record_time": True,
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
            points = {"u": ms.RandomStartPoint("uniform", u_min, u_max, u.shape)}
            x_min = np.zeros((nx, 1))
            x_max = np.asarray([[0.5], [0.01], [30], [0.02]])
            for state in self.states:
                points[state] = ms.RandomStartPoint("uniform", x_min, x_max, x.shape)
            self.random_start_points = ms.RandomStartPoints(
                points, multistarts - 1, seed=np_random
            )

    def disturbance(self, name: str, size: int = 1) -> tuple[cs.SX, list[cs.SX]]:
        """Adds a disturbance parameter to the stochastic MPC controller along the whole prediction
        horizon. Only one disturbance is used for all samples.

        Parameters
        ----------
        name : str
            Name of the disturbance.
        size : int, optional
            Size of the disturbance (assumed to be a vector). Defaults to 1.

        Returns
        -------
        casadi.SX or MX
            The symbol for the new disturbance in the MPC controller.
        """
        return Mpc.disturbance(self, name, size)

    def set_dynamics(
        self,
        F: Sequence[
            (
                cs.Function
                | Callable[[tuple[npt.ArrayLike, ...]], tuple[npt.ArrayLike, ...]]
            )
        ],
        n_in: int | None = None,
        n_out: int | None = None,
    ) -> None:
        """Sets the dynamics of all samples in the scenario based MPC.
        Each element of F corresponds to the dynamics of a sample.

        Parameters
        ----------
        F : list[Union[casadi.Function, Callable[[tuple[npt.ArrayLike, ...]], tuple[npt.ArrayLike, ...]]]
            The dynamics of each sample. F[i] is a CasADi function of the form `x+ = F(x,u)` or `x+ = F(x,u,d)`, where
            `x, u, d` are the state, action, disturbances respectively, and `x+` is the
            next state. The function can have multiple outputs, in which case `x+` is
            assumed to be the first one.
        n_in : int, optional
            In case a callable is passed instead of a casadi.Function, then the number
            of inputs must be manually specified via this argument.
        n_out : int, optional
            Same as above, for outputs."""
        if len(F) != self.n_samples:
            raise ValueError(
                "The number of dynamics must be equal to the number of samples."
            )
        if self._dynamics is not None:
            raise RuntimeError("Dynamics were already set.")
        if isinstance(F, cs.Function):
            n_in = F.n_in()
            n_out = F.n_out()
        elif n_in is None or n_out is None:
            raise ValueError(
                "Args `n_in` and `n_out` must be manually specified when F is not a "
                "casadi function."
            )
        if n_in is None or n_in < 2 or n_in > 3 or n_out is None or n_out < 1:
            raise ValueError(
                "The dynamics function must accepted 2 or 3 arguments and return at "
                f"at least 1 output; got {n_in} inputs and {n_out} outputs instead."
            )
        if self._is_multishooting:
            self._multishooting_dynamics(F, n_in, n_out)
        else:
            self._singleshooting_dynamics(F, n_in, n_out)
        self._dynamics = F

    def _singleshooting_dynamics(
        self, F: Sequence[cs.Function], _: int, n_out: int
    ) -> None:
        raise NotImplementedError("This method is not implemented for SampleBasedMpc")

    def _multishooting_dynamics(
        self, F: Sequence[cs.Function], n_in: int, n_out: int
    ) -> None:
        state_names = self.single_states.keys()
        U = cs.vcat(self._actions_exp.values())
        args_at: Callable[[int], tuple[Any, ...]]
        for i in range(self._n_scenarios):
            X_i = cs.vcat([self._states[_n("x", i)] for n in state_names])
            if n_in < 3:
                args_at = lambda k: (X_i[:, k], U[:, k])
            else:
                D = cs.vcat(self._disturbances.values())
                args_at = lambda k: (X_i[:, k], U[:, k], D[:, k])
            xs_i_next = []
            for k in range(self._prediction_horizon):
                x_i_next = F[i](*args_at(k))
                if n_out != 1:
                    x_i_next = x_i_next[0]
                xs_i_next.append(x_i_next)
            self.constraint(_n("dyn", i), cs.hcat(xs_i_next), "==", X_i[:, 1:])
