from typing import Any, Literal

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt

from greenhouse.model import Model


class LettuceGreenHouse(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """A lettuce greenhouse environment"""

    nx = 4  # number of states
    nu = 3  # number of control inputs
    nd = 4  # number of disturbances
    ts = 60.0 * 15.0  # time step (15 minutes) in seconds
    steps_per_day = 24 * 4  # number of time steps per day

    # disturbance data
    disturbance_data = np.load("data/disturbances.npy")
    VIABLE_STARTING_IDX = np.array(
        [0]
    )  # valid starting days of distrubance data  # TODO make these legit
    training_percentage = 0.8  # 80% of the valid data is used for training
    split_indx = int(np.floor(training_percentage * len(VIABLE_STARTING_IDX)))

    disturbance_profiles_all_episodes: list[
        np.ndarray
    ] = []  # store all disturbances over all episodes
    step_counter = 0

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

        self.p = Model.get_true_parameters()

        # define the dynamics of the environment
        if model_type == "continuous":
            x = cs.MX.sym("x", (self.nx, 1))
            u = cs.MX.sym("u", (self.nu, 1))
            d = cs.MX.sym("d", (self.nd, 1))
            o = cs.vertcat(u, d)

            ode = {"x": x, "p": o, "ode": Model.df(x, u, d, self.p)}
            integrator = cs.integrator(
                "env_integrator",
                "cvodes",
                ode,
                0.0,
                self.ts,
                {"abstol": 1e-8, "reltol": 1e-8},
            )
            xf = integrator(x0=x, p=cs.vertcat(u, d))["xf"]
            self.dynamics = cs.Function("dynamics", [x, u, d], [xf])
        elif model_type == "rk4":
            self.dynamics = lambda x, u, d: Model.rk4_step(x, u, d, self.p, self.ts)
        elif model_type == "euler":
            self.dynamics = lambda x, u, d: Model.euler_step(x, u, d, self.p, self.ts)

        self.c_u = cost_parameters_dict.get(
            "c_u", np.array([10, 1, 1])
        )  # penalty on control inputs
        self.c_y = cost_parameters_dict.get("c_y", 0.0)  # reward on final lettuce yield
        self.c_dy = cost_parameters_dict.get(
            "c_dy", 100.0
        )  # reward on step-wise lettuce yield
        self.w = cost_parameters_dict.get(
            "w", 1e3 * np.ones(4)
        )  # penatly on constraint violations

        if len(self.VIABLE_STARTING_IDX) == 1:
            self.TRAIN_VIABLE_STARTING_IDX = self.VIABLE_STARTING_IDX
            self.TEST_VIABLE_STARTING_IDX = self.VIABLE_STARTING_IDX
        else:
            self.np_random.shuffle(self.VIABLE_STARTING_IDX)
            self.TRAIN_VIABLE_STARTING_IDX = self.VIABLE_STARTING_IDX[: self.split_indx]
            self.TEST_VIABLE_STARTING_IDX = self.VIABLE_STARTING_IDX[self.split_indx :]

        self.yield_step = (
            self.steps_per_day * growing_days - 1
        )  # the time step at which the yield reward is caclculated
        self.growing_days = growing_days
        self.disturbance_type = disturbance_type
        self.testing = testing

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
        self.x = np.array([0.0035, 0.001, 15, 0.008])  # initial condition of the system
        self.previous_lettuce_yield = Model.output(self.x, self.p)[
            0
        ]  # get the initial weight (first element of output)

        # reset the disturbance profile
        self.disturbance_profile = self.generate_disturbance_profile()
        # add in this episodes disturbance to the data, adding only the episode length of data
        self.disturbance_profiles_all_episodes.append(
            self.disturbance_profile[:, : self.growing_days * self.steps_per_day]
        )  # only append growing days worth of data

        self.step_counter = 0
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
        y_max = Model.get_output_max(self.disturbance_profile[:, self.step_counter])
        y_min = Model.get_output_min(self.disturbance_profile[:, self.step_counter])

        # penalize control inputs
        cost += np.dot(self.c_u, action).item()

        # cost step change in lettuce yield
        cost -= self.c_dy * (y[0] - self.previous_lettuce_yield)

        # penalize constraint violations
        cost += np.dot(self.w, np.maximum(0, y_min - y)).item()
        cost += np.dot(self.w, np.maximum(0, y - y_max)).item()

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
        action = np.asarray(action.elements())
        r = float(self.get_stage_cost(self.x, action))
        self.previous_lettuce_yield = Model.output(self.x, self.p)[
            0
        ]  # update the previous lettuce yield
        self.x = np.asarray(
            self.dynamics(
                self.x, action, self.disturbance_profile[:, self.step_counter]
            )
        ).reshape(self.nx)

        truncated = self.step_counter == self.yield_step
        self.step_counter += 1
        return self.x, r, truncated, False, {}

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

    def generate_disturbance_profile(self) -> npt.NDArray[np.floating]:
        """Returns the disturbance profile.

        Returns
        -------
        np.ndarray
            The disturbance profile."""
        if self.disturbance_type == "noisy":
            raise NotImplementedError("This method is not implemented.")
        elif self.disturbance_type == "multiple":
            initial_day = self.np_random.choice(
                self.TEST_VIABLE_STARTING_IDX
                if self.testing
                else self.TRAIN_VIABLE_STARTING_IDX
            )
            return self.pick_disturbance(
                initial_day, self.growing_days + 1
            )  # one extra day in the disturbance profile for the MPC prediction horizon
        else:
            raise ValueError("Invalid disturbance type.")

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
            "w": self.w,
        }
