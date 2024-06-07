from typing import Any, Literal

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from greenhouse.model import Model


class LettuceGreenHouse(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A lettuce greenhouse environment"""

    nx = 4  # number of states
    nu = 3  # number of control inputs
    nd = 4  # number of disturbances
    ts = 60.0 * 15.0  # time step (15 minutes) in seconds
    steps_per_day = 24 * 4  # number of time steps per day
    du_lim = Model.get_du_lim()  # maximum allowed variation in control inputs

    # disturbance data
    disturbance_data = np.load("data/disturbances.npy")
    VIABLE_STARTING_IDX = np.arange(20)  # valid starting days of distrubance data

    def __init__(
        self,
        growing_days: int,
        model_type: Literal["continuous", "rk4", "euler"],
        cost_parameters_dict: dict = {},
        disturbance_type: Literal["noisy", "multiple"] = "multiple",
        testing: bool = False,
    ) -> None:
        """Initializes the environment.

        Parameters
        ----------
        growing_days : int
            The number of days the lettuce is grown for. Defines an episode length.
        model_type : str
            The type of model used for the environment.
        cost_parameters_dict : dict
            The cost parameters for the environment.
        disturbance_type : str
            The type of disturbances used. "noisy" uses one disturbance profile for every episode with noise added.
            "multiple" uses one of a subset of deterministic disturbance profiles for each episode.
        testing : bool
            Whether the environment is in testing mode. Otherwise training mode.
        """
        super().__init__()

        # get the true parameters of the model, and initialize a storage for disturbance
        # profiles that will be used in the environment
        p = self.p = Model.get_true_parameters()
        self.disturbance_profiles_all_episodes: list[np.ndarray] = []

        # define the observation and action space
        lbx = np.asarray([0.0, 0.0, -273.15, 0.0])
        self.observation_space = Box(lbx, np.inf, (self.nx,), np.float64)
        self.action_space = Box(
            Model.get_u_min(), Model.get_u_max(), (self.nu,), np.float64
        )

        # define the dynamics of the environment - clip states to lower bound as
        # sometimes the dynamics will produce lower values
        ts = self.ts
        x = cs.MX.sym("x", (self.nx, 1))
        u = cs.MX.sym("u", (self.nu, 1))
        d = cs.MX.sym("d", (self.nd, 1))
        if model_type == "continuous":
            o = cs.vertcat(u, d)

            ode = {"x": x, "p": o, "ode": Model.df(x, u, d, p)}
            integrator = cs.integrator(
                "integrator", "cvodes", ode, 0.0, ts, {"abstol": 1e-8, "reltol": 1e-8}
            )
            xf = integrator(x0=x, p=cs.vertcat(u, d))["xf"]
            dynamics_cvodes = cs.Function("dynamics", [x, u, d], [xf])
            dynamics_rk4_fallback = cs.Function(
                "fallback", [x, u, d], [Model.rk4_step(x, u, d, p, ts, steps_per_ts=50)]
            )

            def inner_dynamics(x, u, d):
                try:
                    return dynamics_cvodes(x, u, d)
                except RuntimeError:
                    return dynamics_rk4_fallback(x, u, d)

        elif model_type == "rk4":
            xf = Model.rk4_step(x, u, d, p, self.ts)
            inner_dynamics = cs.Function("dynamics", [x, u, d], [xf])
        elif model_type == "euler":
            xf = Model.euler_step(x, u, d, p, self.ts)
            inner_dynamics = cs.Function("dynamics", [x, u, d], [xf])

        def dynamics(x, u, d):
            x_new = np.asarray(inner_dynamics(x, u, d)).reshape(self.nx)
            return np.maximum(np.asarray(x_new).reshape(self.nx), lbx)

        self.dynamics = dynamics

        self.c_u = cost_parameters_dict.get(
            "c_u", np.array([10, 1, 1])
        )  # penalty on control inputs
        self.c_y = cost_parameters_dict.get("c_y", 0.0)  # reward on final lettuce yield
        self.c_dy = cost_parameters_dict.get(
            "c_dy", 100.0
        )  # reward on step-wise lettuce yield
        self.w_y = cost_parameters_dict.get(
            "w_y", np.full(self.nx, 1e4)
        )  # penatly on constraint violations
        self.w_du = cost_parameters_dict.get(
            "w_du", np.full(self.nu, 1e3)
        )  # penatly on control variation constraint violations

        self.yield_step = (
            self.steps_per_day * growing_days - 1
        )  # the time step at which the yield reward is caclculated
        self.growing_days = growing_days
        self.disturbance_type = disturbance_type
        self.testing = testing

    @property
    def current_disturbance(self) -> npt.NDArray[np.floating]:
        """Gets the disturbance values at the current timestep."""
        return self.disturbance_profile[:, self.step_counter]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the environment to its initial state.

        Parameters
        ----------
        seed : int
            The seed for the random number generator.
        options : dict
            The reset options for the environment.

        Returns
        -------
        tuple
            The initial state of the environment and an empty dictionary."""
        super().reset(seed=seed, options=options)
        self.observation_space.seed(seed)
        self.action_space.seed(seed)

        # set initial condition of the system and initial weight (first output element)
        self.x = np.array([0.0035, 0.001, 15, 0.008])
        self.previous_lettuce_yield = Model.output(self.x, self.p)[0]

        if options is not None and "initial_day" in options:
            self.disturbance_profile = self.generate_disturbance_profile(
                options["initial_day"]
            )
        else:
            # randomly shuffle the disturbance data's starting indeces (but do it only once)
            # and then reset the disturbance profile
            if not hasattr(self, "TRAIN_VIABLE_STARTING_IDX"):
                training_percentage = 0.8  # 80% of the valid data is used for training
                idx = int(np.floor(training_percentage * self.VIABLE_STARTING_IDX.size))
                if len(self.VIABLE_STARTING_IDX) == 1:
                    self.TRAIN_VIABLE_STARTING_IDX = self.VIABLE_STARTING_IDX
                    self.TEST_VIABLE_STARTING_IDX = self.VIABLE_STARTING_IDX
                else:
                    self.np_random.shuffle(self.VIABLE_STARTING_IDX)
                    self.TRAIN_VIABLE_STARTING_IDX = self.VIABLE_STARTING_IDX[:idx]
                    self.TEST_VIABLE_STARTING_IDX = self.VIABLE_STARTING_IDX[idx:]
            self.disturbance_profile = self.generate_disturbance_profile()

        # add in this episodes disturbance to the data, adding only the episode length of data
        self.disturbance_profiles_all_episodes.append(
            self.disturbance_profile[:, : self.growing_days * self.steps_per_day]
        )  # only append growing days worth of data

        self.step_counter = 0
        self.previous_action = np.zeros(self.nu)
        assert self.observation_space.contains(self.x) and self.action_space.contains(
            self.previous_action
        ), f"Invalid state or action in `reset`: {self.x}, {self.previous_action}."
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        """Calculates the stage cost of the environment.

        Parameters
        ----------
        state : np.ndarray
            The current state of the environment.
        action : np.ndarray
            The action taken in the environment.

        Returns
        -------
        float
            The stage cost of the environment."""
        cost = 0.0
        y = Model.output(state, self.p)  # current output
        y_max = Model.get_output_max(self.current_disturbance)
        y_min = Model.get_output_min(self.current_disturbance)

        # penalize control inputs
        cost += np.dot(self.c_u, action).item()

        # cost step change in lettuce yield
        cost -= self.c_dy * (y[0] - self.previous_lettuce_yield)

        # penalize constraint violations
        cost += np.dot(self.w_y, np.maximum(0, (y_min - y)/(y_max - y_min))).item()
        cost += np.dot(self.w_y, np.maximum(0, (y - y_max)/(y_max - y_min))).item()
        if self.step_counter > 0:
            cost += np.dot(
                self.w_du,
                np.maximum(0, (np.abs(action - self.previous_action) - self.du_lim)/(self.du_lim)),
            )

        # reward final yield
        if self.step_counter == self.yield_step:
            cost -= self.c_y * y[0]

        return cost

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the greenhouse environment.

        Parameters
        ----------
        action : cs.DM
            The action taken in the environment.

        Returns
        -------
        tuple
            The new state of the environment, the reward, whether the episode is truncated, whether the episode is terminated, and an empty dictionary.
        """
        u = np.asarray(action).reshape(self.nu)
        assert self.action_space.contains(u), f"Invalid action in `step`: {u}."
        x = self.x
        r = float(self.get_stage_cost(x, u))
        d = self.current_disturbance
        x_new = np.asarray(self.dynamics(x, u, d)).reshape(self.nx)
        assert self.observation_space.contains(
            x_new
        ), f"Invalid next state in `step` {x_new}."

        self.previous_lettuce_yield = Model.output(x, self.p)[0]
        self.previous_action = u
        self.x = x_new.copy()
        truncated = self.step_counter == self.yield_step
        self.step_counter += 1
        return x_new, r, truncated, False, {}

    def get_current_disturbance(self, length: int) -> npt.NDArray[np.floating]:
        """Returns the disturbance profile for a certain length starting from the current time step.

        Parameters
        ----------
        length : int
            The length of the disturbance profile.

        Returns
        -------
        np.ndarray
            The disturbance profile for the given length."""
        return self.disturbance_profile[
            :, self.step_counter : self.step_counter + length
        ]

    def generate_disturbance_profile(
        self, initial_day: int | None = None
    ) -> npt.NDArray[np.floating]:
        """Returns the disturbance profile.

        Parameters
        ----------
        initial_day : int | None
            The day to start the disturbance profile from. If none, a random
            day is chosen from the viable starting days.

        Returns
        -------
        np.ndarray
            The disturbance profile."""
        if initial_day is None:
            if self.disturbance_type == "noisy":
                raise NotImplementedError("This method is not implemented.")
            elif self.disturbance_type == "multiple":
                initial_day = self.np_random.choice(
                    self.TEST_VIABLE_STARTING_IDX
                    if self.testing
                    else self.TRAIN_VIABLE_STARTING_IDX
                )
            else:
                raise ValueError("Invalid disturbance type.")
        return self.pick_disturbance(
            initial_day, self.growing_days + 1
        )  # one extra day in the disturbance profile for the MPC prediction horizon

    def pick_disturbance(
        self, initial_day: int, num_days: int
    ) -> npt.NDArray[np.floating]:
        """Returns the disturbance profile of a certain length, starting from a given day.

        Parameters
        ----------
        initial_day : int
            The day to start the disturbance profile from.
        num_days : int
            The number of days in the disturbance profile.

        Returns
        -------
        np.ndarray
            The disturbance profile."""
        # disturbance data has 324 days
        if initial_day + num_days > 324:
            raise ValueError(
                "The requested initial day and length of the disturbance profile exceeds the data available."
            )
        idx1 = initial_day * self.steps_per_day
        idx2 = (initial_day + num_days) * self.steps_per_day
        return self.disturbance_data[:, idx1:idx2]

    def get_cost_parameters(self) -> dict:
        """Returns the cost parameters of the environment.

        Returns
        -------
        dict
            The cost parameters of the environment."""
        return {
            "c_u": self.c_u,
            "c_y": self.c_y,
            "c_dy": self.c_dy,
            "w_y": self.w_y,
            "w_du": self.w_du,
        }
