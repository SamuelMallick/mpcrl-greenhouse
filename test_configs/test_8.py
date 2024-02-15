from typing import Literal

import numpy as np
from mpcrl import UpdateStrategy, optim

# Recursive first order q-learning


class Test:
    # simulation and training params
    test_ID = "8"
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

    # learning params
    learn_all_p = True  # if false we only learn the sensitive subset of p
    update_strategy = UpdateStrategy(frequency=1, hook="on_timestep_end")
    learning_rate = 1e-20
    optimizer = optim.GradientDescent(learning_rate=learning_rate)
    exploration = None
    experience = None
    hessian_type = "none"
