from typing import Any, Literal

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import Env
from mpcrl import Agent, LstdQLearningAgent

from envs.model import (
    df_true,
    euler_true,
    get_disturbance_profile,
    get_model_details,
    get_y_max,
    get_y_min,
    output_true,
    rk4_true,
)


class LettuceGreenHouse(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """Continuous time environment for a luttuce greenhouse."""

    nx, nu, nd, ts, steps_per_day = get_model_details()
    disturbance_profile = get_disturbance_profile(
        init_day=0, days_to_grow=40
    )  # gets re-called in reset
    disturbance_profile_data = np.empty(
        (4, 0)
    )  # used for plotting the disturbance over a range of episodes
    step_counter = 0

    # noise terms for dynamics
    mean = np.zeros((nx, 1))
    sd = np.array([[0], [0], [0], [0]])

    # store previous weight for stage cost calculation
    prev_weight = None

    def __init__(
        self,
        days_to_grow: int,
        model_type: Literal["nonlinear", "rk4", "euler"],
        rl_cost: dict = {},
    ) -> None:
        super().__init__()

        self.model_type = model_type
        self.c_u = rl_cost.pop("c_u", [100, 1, 1])
        self.c_y = rl_cost.pop("c_y", 1000)
        self.c_dy = rl_cost.pop("c_dy", 100)
        self.w = rl_cost.pop("w", 1e3 * np.ones((1, 4)))

        self.days_to_grow = days_to_grow
        self.yield_step = (
            self.steps_per_day * days_to_grow - 1
        )  # the time step at which the yield reward is caclculated

        # set-up continuous time integrator for dynamics simulation
        x = cs.SX.sym("x", (self.nx, 1))
        u = cs.SX.sym("u", (self.nu, 1))
        d = cs.SX.sym("d", (self.nd, 1))
        p = cs.vertcat(u, d)
        x_new = df_true(x, u, d)
        ode = {"x": x, "p": p, "ode": x_new}
        self.integrator = cs.integrator(
            "env_integrator",
            "cvodes",
            ode,
            0,
            self.ts,
            {"abstol": 1e-8, "reltol": 1e-8},
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the system."""
        self.x = np.array(
            [[0.0035], [0.001], [15], [0.008]]
        )  # initial condition used in the robust approach paper - 2022

        if options is not None and "first_day_index" in options:
            first_day_index = options["first_day_index"]
        else:
            first_day_index = 0
        self.disturbance_profile = get_disturbance_profile(
            first_day_index, days_to_grow=self.days_to_grow
        )

        # add in this episodes disturbance to the data
        self.disturbance_profile_data = np.hstack(
            (
                self.disturbance_profile_data,
                self.disturbance_profile[:, : self.steps_per_day * self.days_to_grow],
            )
        )

        self.step_counter = 0
        super().reset(seed=seed, options=options)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        reward = 0.0
        y = output_true(state)  # get output from current state
        y_max = get_y_max(self.disturbance_profile[:, [self.step_counter]])
        y_min = get_y_min(self.disturbance_profile[:, [self.step_counter]])

        # penalize control inputs
        for i in range(self.nu):
            reward += self.c_u[i] * action[i]

        # reward step change in weight
        if self.step_counter > 0:
            reward -= self.c_dy * (y[0] - self.prev_weight)
        self.prev_weight = y[0]

        # penalize constraint viols
        reward += self.w @ np.maximum(0, y_min - y)
        reward += self.w @ np.maximum(0, y - y_max)

        # reward final yield
        if self.step_counter == self.yield_step:
            reward -= self.c_y * y[0]

        return reward

    def step(
        self, action: cs.DM
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the system."""
        r = float(self.get_stage_cost(self.x, action))

        if self.model_type == "euler":
            x_new = euler_true(
                self.x, action, self.disturbance_profile[:, [self.step_counter]]
            )
        elif self.model_type == "rk4":
            x_new = rk4_true(
                self.x, action, self.disturbance_profile[:, [self.step_counter]]
            )
        elif self.model_type == "nonlinear":
            x_new = self.integrator(
                x0=self.x,
                p=cs.vertcat(action, self.disturbance_profile[:, [self.step_counter]]),
            )["xf"]
        else:
            raise RuntimeError(f"{self.model_type} is not a valid model option.")

        # to add uncertainty to the dynamics
        # self.np_random.normal(self.mean, self.sd, (self.nx, 1))

        self.x = x_new  # + model_uncertainty
        self.step_counter += 1
        return x_new, r, False, False, {}


class GreenhouseAgent(Agent):
    # set the disturbance at start of episode and each new timestep
    def on_episode_start(self, env: Env, episode: int, state) -> None:
        d_pred = env.disturbance_profile[:, : self.V.prediction_horizon + 1]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        # then we use the first entry of the predicted disturbance to determine y bounds
        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = get_y_min(d_pred[:, [k]])
            self.fixed_parameters[f"y_max_{k}"] = get_y_max(d_pred[:, [k]])
        return super().on_episode_start(env, episode, state)

    def on_env_step(self, env: Env, episode: int, timestep: int) -> None:
        d_pred = env.disturbance_profile[
            :, timestep + 1 : (timestep + 1 + self.V.prediction_horizon + 1)
        ]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = get_y_min(d_pred[:, [k]])
            self.fixed_parameters[f"y_max_{k}"] = get_y_max(d_pred[:, [k]])
        return super().on_env_step(env, episode, timestep)


# TODO request bug fix from Fillipo so that the training and evaluation can use the same indexes - at the moment it is okay because we use step counter instead
class GreenhouseLearningAgent(LstdQLearningAgent):
    # set the disturbance at start of episode and each new timestep
    def on_episode_start(self, env: Env, episode: int, state) -> None:
        d_pred = env.disturbance_profile[:, : self.V.prediction_horizon + 1]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = get_y_min(d_pred[:, [k]])
            self.fixed_parameters[f"y_max_{k}"] = get_y_max(d_pred[:, [k]])
        return super().on_episode_start(env, episode, state)

    def on_env_step(self, env: Env, episode: int, timestep: int) -> None:
        d_pred = env.disturbance_profile[
            :,
            timestep + 1 : (timestep + 1 + self.V.prediction_horizon + 1),
        ]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = get_y_min(d_pred[:, [k]])
            self.fixed_parameters[f"y_max_{k}"] = get_y_max(d_pred[:, [k]])
        return super().on_env_step(env, episode, timestep)


class GreenhouseSampleAgent(Agent):
    # set the disturbance at start of episode and each new timestep
    def on_episode_start(self, env: Env, episode: int, state) -> None:
        d_pred = env.disturbance_profile[:, : self.V.prediction_horizon + 1]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        Ns = self.V.Ns
        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = cs.vertcat(
                *[get_y_min(d_pred[:, [k]])] * Ns
            )
            self.fixed_parameters[f"y_max_{k}"] = cs.vertcat(
                *[get_y_max(d_pred[:, [k]])] * Ns
            )
        return super().on_episode_start(env, episode, state)

    def on_env_step(self, env: Env, episode: int, timestep: int) -> None:
        d_pred = env.disturbance_profile[
            :, env.step_counter : (env.step_counter + self.V.prediction_horizon + 1)
        ]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        Ns = self.V.Ns
        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = cs.vertcat(
                *[get_y_min(d_pred[:, [k]])] * Ns
            )
            self.fixed_parameters[f"y_max_{k}"] = cs.vertcat(
                *[get_y_max(d_pred[:, [k]])] * Ns
            )
        return super().on_env_step(env, episode, timestep)
