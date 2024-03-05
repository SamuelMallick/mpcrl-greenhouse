from typing import Literal

import numpy as np
from mpcrl import ExperienceReplay, UpdateStrategy, optim
from mpcrl.core.schedulers import ExponentialScheduler


class Test:
    # simulation and training params
    test_ID = "test_13"
    num_days = 40
    ep_len = num_days * 24 * 4  # 'x' days of 15 minute timesteps
    num_episodes = 50

    # mpc and model params
    base_model: Literal[
        "nonlinear", "rk4", "euler"
    ] = "rk4"  # underlying simulation model
    prediction_model = "rk4"  # mpc prediction model
    horizon = 24
    discount_factor = 0.99
    rl_cost = {"c_u": [0, 0, 0], "c_y": 0, "c_dy": 100, "w": 1e3 * np.ones((1, 4))}
    p_perturb: list = []  # index of parameters that are perturbed

    # learning params
    p_learn = [i for i in range(28)]  # index of parameters to learn
    # fixed pars and learable pars
    learnable_pars_init = {
        "V0": np.zeros((1,)),
        "c_dy": 100 * np.ones((1,)),
        "c_y": 1e4 * np.ones((1,)),
    }
    # bounds on learnable pars
    learn_bounds = {
        "c_dy": [0, np.inf],
        "c_y": [0, np.inf],
    }
    fixed_pars = {
        "c_u": np.array([0, 0, 0]),
        "w": 1e3 * np.ones((1, 4)),
    }

    update_strategy = UpdateStrategy(1, hook="on_episode_end")
    learning_rate = 1e-2
    optimizer = optim.NetwonMethod(
        learning_rate=ExponentialScheduler(learning_rate, factor=1)
    )
    exploration = None
    experience = ExperienceReplay(
        maxlen=3 * ep_len,
        sample_size=2 * ep_len,
        include_latest=ep_len,
        seed=0,
    )
    hessian_type = "approx"
