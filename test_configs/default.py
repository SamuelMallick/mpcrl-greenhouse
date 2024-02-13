from typing import Literal

import numpy as np
from mpcrl import ExperienceReplay, UpdateStrategy, optim
from mpcrl.core.schedulers import ExponentialScheduler


class Test:
    # simulation and training params
    test_ID = "default"
    num_days = 1
    ep_len = num_days * 24 * 4  # 'x' days of 15 minute timesteps
    num_episodes = 1

    # mpc and model params
    base_model: Literal[
        "nonlinear", "rk4", "euler"
    ] = "rk4"  # underlying simulation model
    prediction_model = "rk4"  # mpc prediction model
    horizon = 24
    discount_factor = 0.99
    rl_cost = {"c_u": [10, 1, 1], "c_y": 1e3, "c_dy": 100, "w": 1e3 * np.ones((1, 4))}

    # learning params
    learn_all_p = True  # if false we only learn the sensitive subset of p
    # fixed pars and learable pars
    learnable_pars_init = {
        "V0": np.zeros((1,)),
        "c_u": np.array([10, 1, 1]),
        "c_y": 1e3 * np.ones((1,)),
        "c_dy": 100 * np.ones((1,)),
        "w": 1e3 * np.ones((1, 4)),
    }
    fixed_pars = {}

    update_strategy = UpdateStrategy(int(ep_len / 2), skip_first=2)
    learning_rate = 1e-3
    optimizer = optim.NetwonMethod(
        learning_rate=ExponentialScheduler(learning_rate, factor=1)
    )
    exploration = None
    experience = ExperienceReplay(
        maxlen=10 * ep_len,
        sample_size=3 * ep_len,
        include_latest=ep_len,
        seed=0,
    )
    hessian_type = "approx"
