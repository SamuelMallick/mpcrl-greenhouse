from typing import Literal

import numpy as np
from mpcrl import ExperienceReplay, UpdateStrategy, optim
from mpcrl.core.exploration import EpsilonGreedyExploration
from mpcrl.core.schedulers import ExponentialScheduler
# Second-order


class Test:
    # simulation and training params
    test_ID = "9"
    num_days = 40
    ep_len = num_days * 24 * 4  # 'x' days of 15 minute timesteps
    num_episodes = 100

    # mpc and model params
    base_model: Literal[
        "nonlinear", "rk4", "euler"
    ] = "rk4"  # underlying simulation model
    prediction_model = "rk4"  # mpc prediction model
    horizon = 24
    discount_factor = 0.99
    rl_cost = {"c_u": [0, 0, 0], "c_y": 0, "c_dy": 100, "w": 0 * np.ones((1, 4))}

    # learning params
    learn_all_p = True  # if false we only learn the sensitive subset of p
    # fixed pars and learable pars
    learnable_pars_init = {
        "V0": np.zeros((1,)),
        "c_dy": 100 * np.ones((1,)),
    }
    fixed_pars = {
        "c_u": np.array([0, 0, 0]),
        "c_y": 0 * np.ones((1,)),
        "w": 1e3 * np.ones((1, 4)),
    }

    update_strategy = UpdateStrategy(frequency=1, hook="on_episode_end")
    learning_rate = 1e-2
    optimizer = optim.NetwonMethod(learning_rate=learning_rate)
    exploration = EpsilonGreedyExploration(
                    epsilon=ExponentialScheduler(0.5, factor=0.9),
                    hook="on_episode_end",
                    strength=0.2*np.array([[1.2, 7.5, 150]]).T,
                ),
    experience = ExperienceReplay(
        maxlen=3 * ep_len,
        sample_size=2 * ep_len,
        include_latest=ep_len,
        seed=0,
    )
    hessian_type = "approx"
