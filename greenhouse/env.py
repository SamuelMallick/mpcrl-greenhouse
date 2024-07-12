from typing import Any, Literal

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.spaces import Box

from greenhouse.model import Model
from utils.brownian_motion import brownian_excursion


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
        disturbance_profiles_type: Literal["multiple", "single"] = "multiple",
        noisy_disturbance: bool = False,
        testing: Literal["none", "random", "deterministic"] = "none",
        clip_action_variation: bool = False,
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
        disturbance_profiles_type : str
            The type of disturbances used. 'muliple' uses of of a subset of possible disturbances
            each episode. 'single' uses a single disturbance profile for all episodes.
        noisy_disturbance : bool
            Whether disturbance profiles are perturbed with brownian noise.
        testing : "none", "random", "deterministic"
            Applicable if disturbance_type == 'multiple'.
            Whether the disturbances used in the env are drawn from the training or testing set.
            If "none", the training disturbances are used. If "random", testing disturbances are drawn randomly.
            If "deterministic", testing disturbances are drawn deterministically.
        clip_actions : bool
            If True the input action is clipped such that the change in control inputs is within the limits.
            By default, False.
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
            xf = Model.rk4_step(x, u, d, p, ts)
            inner_dynamics = cs.Function("dynamics", [x, u, d], [xf])
        elif model_type == "euler":
            xf = Model.euler_step(x, u, d, p, ts)
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
            "w_y", np.full(self.nx, 1e5)
        )  # penatly on constraint violations
        self.w_du = cost_parameters_dict.get(
            "w_du", np.full(self.nu, 0.0)
        )  # penatly on control variation constraint violations

        self.yield_step = (
            self.steps_per_day * growing_days - 1
        )  # the time step at which the yield reward is caclculated
        self.growing_days = growing_days
        self.disturbance_profiles_type = disturbance_profiles_type
        self.noisy_disturbance = noisy_disturbance
        self.testing = testing
        self._testing_counter = 0  # for deterministic testing
        self.clip_action_variation = clip_action_variation

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
                options["initial_day"], noise_coeff=options.get("noise_coeff", 1.0)
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
        cost += np.dot(self.w_y, np.maximum(0, (y_min - y) / (y_max - y_min))).item()
        cost += np.dot(self.w_y, np.maximum(0, (y - y_max) / (y_max - y_min))).item()
        if self.step_counter > 0:
            cost += np.dot(
                self.w_du,
                np.maximum(
                    0,
                    (np.abs(action - self.previous_action) - self.du_lim)
                    / (self.du_lim),
                ),
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
        if self.clip_action_variation:
            u = np.clip(
                u,
                self.previous_action - self.du_lim,
                self.previous_action + self.du_lim,
            )
        x = self.x
        r = float(self.get_stage_cost(x, u))
        d = self.current_disturbance
        x_new = np.asarray(self.dynamics(x, u, d)).reshape(self.nx)
        assert self.observation_space.contains(
            x_new
        ), f"Invalid next state in `step` {x_new} with (x,u,d)=({x},{u},{d})."

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
        self, initial_day: int | None = None, noise_coeff: float = 1.0
    ) -> npt.NDArray[np.floating]:
        """Returns the disturbance profile. One extra day (growing_days + 1) is added
        to the returned profile to allow for predictions in an MPC horizon.

        Parameters
        ----------
        initial_day : int | None
            The day to start the disturbance profile from. If none, a random
            day is chosen from the viable starting days.
        noise_coeff : float
            The scaling factor for the noise added to the disturbance profile. By default, 1.0.

        Returns
        -------
        np.ndarray
            The disturbance profile."""
        if initial_day is not None:
            if self.disturbance_profiles_type != "single":
                raise ValueError(
                    "The initial day should not be specified when using multiple disturbance profiles."
                )
        else:
            if self.disturbance_profiles_type == "multiple":
                if self.testing == "none":
                    initial_day = self.np_random.choice(self.TRAIN_VIABLE_STARTING_IDX)
                elif self.testing == "random":
                    initial_day = self.np_random.choice(self.TEST_VIABLE_STARTING_IDX)
                else:  # deterministic
                    initial_day = self.TEST_VIABLE_STARTING_IDX[
                        self._testing_counter % len(self.TEST_VIABLE_STARTING_IDX)
                    ]
                    self._testing_counter += 1
            else:  # "single"
                initial_day = self.VIABLE_STARTING_IDX[0]  # use first day in the data

        return self.pick_perturbed_disturbance_profile(
            initial_day,
            self.growing_days + 1,
            (
                noise_coeff * np.array([0.02, 0.01, 0.02, 0.01])
                if self.noisy_disturbance
                else 0.0
            ),
        )

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

    def pick_perturbed_disturbance_profile(
        self, initial_day: int, num_days: int, noise_scaling: float | np.ndarray
    ) -> np.ndarray:
        """Returns the disturbance profile starting from a given day with a brownian
        noise random process added to it. The brownian noise is generated as the cumulative sum
        of a white noise signal. The white noise is drawn from a unifrom distribution of width noise_scaling*range, where
        range is the range of the given element in the disturbance. For the temerature and the radation the noise
        is generated with Brownian excursions, to prevent impossible values, i.e., negative radiation.

        Parameters
        ----------
        initial_day : int
            The day to start the disturbance profile from.
        num_days : int
            The number of days in the disturbance profile.
        noise_scaling : float
            The width of the uniform distribution from which the white noise samples are drawn.

        Returns
        -------
        np.ndarray
            The perturbed disturbance profile."""
        nominal_disturbance = self.pick_disturbance(initial_day, num_days)
        noise_width = noise_scaling * (
            np.max(nominal_disturbance, axis=1) - np.min(nominal_disturbance, axis=1)
        )
        # radiation noise done seperately with Brownian excursion
        non_zero_mask = nominal_disturbance[0] > 0.5
        diff_array = np.diff(
            non_zero_mask.astype(int)
        )  # compute the difference in boolean array to find where changes from false/true or true/false occur
        starts = np.where(diff_array == 1)[0] + 1
        ends = np.where(diff_array == -1)[0] + 1
        radiation_noise = np.zeros((num_days * self.steps_per_day,))
        temperature_noise = np.zeros((num_days * self.steps_per_day,))
        for i in range(starts.size):
            brownian_excursion_rad = brownian_excursion(
                ends[i] - starts[i], noise_width[0], self.np_random
            )
            radiation_noise[starts[i] : ends[i]] = brownian_excursion_rad
            brownian_excursion_temp = brownian_excursion(
                ends[i] - starts[i], noise_width[2], self.np_random
            )
            temperature_noise[starts[i] : ends[i]] = brownian_excursion_temp

        white_noise = self.np_random.uniform(
            -noise_width[[1, 3], np.newaxis] / 2,
            noise_width[[1, 3], np.newaxis] / 2,
            (self.nd - 2, num_days * self.steps_per_day),
        )
        brownian_noise = np.vstack(
            (
                radiation_noise,
                np.cumsum(white_noise[0]),
                temperature_noise,
                np.cumsum(white_noise[1]),
            )
        )
        noisy_disturbance = nominal_disturbance + brownian_noise
        noisy_disturbance[0] = np.maximum(
            0, noisy_disturbance[0]
        )  # radiation cannot be negative
        return noisy_disturbance

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
