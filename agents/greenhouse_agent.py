import casadi as cs
import numpy as np
from mpcrl import Agent, LstdQLearningAgent

from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model


class GreenhouseAgent(Agent):
    """An agent controlling the greenhouse."""

    def set_mpc_parameters(self, d: np.ndarray) -> None:
        """Sets the disturbance and constraints parameters of the agent's MPC.

        Parameters
        ----------
        d : np.ndarray
            The disturbance.
        """
        self.fixed_parameters["d"] = d[:, :-1]
        for k in range(self.V.prediction_horizon + 1):
            self.fixed_parameters[f"y_min_{k}"] = Model.get_output_min(d[:, [k]])
            self.fixed_parameters[f"y_max_{k}"] = Model.get_output_max(d[:, [k]])

    def on_episode_start(
        self, env: LettuceGreenHouse, episode: int, state: np.ndarray
    ) -> None:
        """Call back for on episode start. Set the disturbance profile over the MPC prediction horizon and update the constraints.

        Parameters
        ----------
        env : LettuceGreenHouse
            The environment.
        episode : int
            The episode number.
        state : np.ndarray
            The initial state.
        """
        d = env.get_current_disturbance(self.V.prediction_horizon + 1)
        self.set_mpc_parameters(d)
        return super().on_episode_start(env, episode, state)

    def on_env_step(self, env: LettuceGreenHouse, episode: int, timestep: int) -> None:
        """Call back for on environment step. Set the disturbance profile over the MPC prediction horizon and update the constraints.

        Parameters
        ----------
        env : LettuceGreenHouse
            The environment.
        episode : int
            The episode number.
        timestep : int
            The timestep number."""
        d = env.get_current_disturbance(self.V.prediction_horizon + 1)
        self.set_mpc_parameters(d)
        return super().on_env_step(env, episode, timestep)


class GreenhouseSampleAgent(GreenhouseAgent):
    """An agent controlling the greenhouse using a sample based robust MPC."""


class GreenhouseLearningAgent(LstdQLearningAgent, GreenhouseAgent):
    """An agent controlling the greenhouse who can learn the MPC policy using LSTD Q-learning."""
