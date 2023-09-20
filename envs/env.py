from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import casadi as cs
import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import Env
from envs.model import (
    df_real,
    get_disturbance_profile,
    get_model_details,
    get_y_max,
    get_y_min,
    output_real,
)
from mpcrl import Agent, LstdQLearningAgent


class LettuceGreenHouse(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """Continuous time environment for a luttuce greenhouse."""

    nx, nu, nd, ts = get_model_details()
    disturbance_profile = get_disturbance_profile(init_day=0)
    step_counter = 0

    # cost constants
    c_u = [10, 1, 1]  # penalty on each control signal
    c_y = 10e3  # reward on yield
    yield_step = 191 # 3839 
    w = np.array([1000, 1000, 1000, 1000])  # penalty on constraint violations

    # noise terms
    mean = np.zeros((nx, 1))
    # sd = np.array([[1e-5], [1e-5], [0.1], [1e-5]])
    sd = np.array([[0], [0], [0], [0]])

    def __init__(self) -> None:
        super().__init__()

        # set-up continuous time integrator for dynamics simulation
        x = cs.SX.sym("x", (self.nx, 1))
        u = cs.SX.sym("u", (self.nu, 1))
        d = cs.SX.sym("d", (self.nd, 1))
        p = cs.vertcat(u, d)
        x_new = df_real(x, u, d)
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
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[npt.NDArray[np.floating], Dict[str, Any]]:
        """Resets the state of the system."""
        self.x = np.array([[0.0035], [0.001], [15], [0.008]])
        self.step_counter = 0
        super().reset(seed=seed, options=options)
        return self.x, {}

    def get_stage_cost(
        self, state: npt.NDArray[np.floating], action: npt.NDArray[np.floating]
    ) -> float:
        reward = 0.0
        y = output_real(state)
        y_max = get_y_max(self.disturbance_profile[:, [self.step_counter]])
        y_min = get_y_min(self.disturbance_profile[:, [self.step_counter]])
        for i in range(self.nu):
            reward += self.c_u[i] * action[i]
        reward += self.w @ np.maximum(0, y_min - y)
        reward += self.w @ np.maximum(0, y - y_max)
        if self.step_counter == self.yield_step:
            reward -= self.c_y * y[0]
        return reward

    def step(
        self, action: cs.DM
    ) -> Tuple[npt.NDArray[np.floating], float, bool, bool, Dict[str, Any]]:
        """Steps the system."""
        r = float(self.get_stage_cost(self.x, action))
        x_new = self.integrator(
            x0=self.x,
            p=cs.vertcat(action, self.disturbance_profile[:, [self.step_counter]]),
        )["xf"]
        # x_new = rk4_step(self.x, action, self.disturbance_profile[:, [self.step_counter]])
        model_uncertainty = self.np_random.normal(self.mean, self.sd, (self.nx, 1))
        self.x = x_new + model_uncertainty
        self.step_counter += 1
        return x_new, r, False, False, {}


class GreenhouseAgent(Agent):
    # set the disturbance at start of episode and each new timestep
    def on_episode_start(self, env: Env, episode: int) -> None:
        d_pred = env.disturbance_profile[:, : self.V.prediction_horizon + 1]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        # then we use the first entry of the predicted disturbance to determine y bounds
        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = get_y_min(d_pred[:, [k]])
            self.fixed_parameters[f"y_max_{k}"] = get_y_max(d_pred[:, [k]])
        return super().on_episode_start(env, episode)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        d_pred = env.disturbance_profile[
            :, timestep : (timestep + self.V.prediction_horizon + 1)
        ]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        # then we use the first entry of the predicted disturbance to determine y bounds
        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = get_y_min(d_pred[:, [k]])
            self.fixed_parameters[f"y_max_{k}"] = get_y_max(d_pred[:, [k]])
        return super().on_timestep_end(env, episode, timestep)

# TODO REMOVE USE OF TIMESTEP!
class GreenhouseLearningAgent(LstdQLearningAgent):
    # set the disturbance at start of episode and each new timestep
    def on_episode_start(self, env: Env, episode: int) -> None:
        d_pred = env.disturbance_profile[:, : self.V.prediction_horizon + 1]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        # then we use the first entry of the predicted disturbance to determine y bounds
        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = get_y_min(d_pred[:, [k]])
            self.fixed_parameters[f"y_max_{k}"] = get_y_max(d_pred[:, [k]])
        return super().on_episode_start(env, episode)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        d_pred = env.disturbance_profile[
            :, env.step_counter + 1 : (env.step_counter + 1 + self.V.prediction_horizon + 1)
        ]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        # then we use the first entry of the predicted disturbance to determine y bounds
        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = get_y_min(d_pred[:, [k]])
            self.fixed_parameters[f"y_max_{k}"] = get_y_max(d_pred[:, [k]])
        return super().on_timestep_end(env, episode, timestep)


class GreenhouseSampleAgent(Agent):
    # set the disturbance at start of episode and each new timestep
    def on_episode_start(self, env: Env, episode: int) -> None:
        d_pred = env.disturbance_profile[:, : self.V.prediction_horizon + 1]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        Ns = self.V.Ns
        # then we use the first entry of the predicted disturbance to determine y bounds
        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = cs.vertcat(
                *[get_y_min(d_pred[:, [k]])] * Ns
            )
            self.fixed_parameters[f"y_max_{k}"] = cs.vertcat(
                *[get_y_max(d_pred[:, [k]])] * Ns
            )
        return super().on_episode_start(env, episode)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        d_pred = env.disturbance_profile[
            :, timestep : (timestep + self.V.prediction_horizon + 1)
        ]
        self.fixed_parameters["d"] = d_pred[:, :-1]

        Ns = self.V.Ns
        # then we use the first entry of the predicted disturbance to determine y bounds
        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = cs.vertcat(
                *[get_y_min(d_pred[:, [k]])] * Ns
            )
            self.fixed_parameters[f"y_max_{k}"] = cs.vertcat(
                *[get_y_max(d_pred[:, [k]])] * Ns
            )
        return super().on_timestep_end(env, episode, timestep)
