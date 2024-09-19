from typing import Any, Literal

import numpy as np
from mpcrl import ExperienceReplay, UpdateStrategy, optim
from mpcrl.core.schedulers import ExponentialScheduler

from greenhouse.model import Model
from sims.configs.default import DefaultTest


# Test 80 but different seed, and expl/exp using new seed, model params not from seed
class Test(DefaultTest):
    # simulation and training params
    test_ID = "test_94"
    num_days = 40
    ep_len = num_days * 24 * 4  # 'x' days of 15 minute timesteps
    num_episodes = 100
    disturbance_type: Literal["multiple", "single"] = "single"
    noisy_disturbance = True
    noise_coeff = 1.0  # scales the noise generation
    initial_day: int | None = 0 if disturbance_type == "single" else None
    clip_action_variation = True
    normalize_reward = False
    seed = 3
    model_parameters_from_seed = False

    # mpc and model params
    base_model: Literal[
        "continuous", "rk4", "euler"
    ] = "continuous"  # underlying simulation model
    prediction_model = "rk4"  # mpc prediction model
    horizon = 24
    discount_factor = 0.99
    rl_cost = {"c_u": [10, 1, 1], "c_y": 0.0, "c_dy": 100, "w_y": 1e5 * np.ones((1, 4))}
    p_perturb = list(range(Model.n_params))  # index of parameters that are perturbed

    # learning params
    p_learn = list(range(Model.n_params))  # index of parameters to learn
    # fixed pars and learable pars
    learnable_pars_init = {
        "V0": np.zeros((1,)),
        "c_dy": 100 * np.ones((1,)),
        "w": 1e5 * np.ones((4,)),
        "olb": np.zeros((4,)),
        "oub": np.zeros((4,)),
        "y_fin": 135 * np.ones((1,)),
        "c_y": 1 * np.ones((1,)),
        "c_u": np.array([10, 1, 1]),
    }
    # bounds on learnable pars
    fixed_pars: dict[str, Any] = {}
    learn_bounds = {
        "c_dy": [0, np.inf],
        "w": [0, np.inf],
        "olb": [-0, 0],
        "oub": [-0, 0],
        "y_fin": [0, np.inf],
        "c_y": [0, np.inf],
        "c_u": [0, np.inf],
    }
    skip_first = 0
    update_strategy = UpdateStrategy(1, skip_first=skip_first, hook="on_episode_end")
    learning_rate = 1e-1
    optimizer = optim.NetwonMethod(
        learning_rate=ExponentialScheduler(learning_rate, factor=1),
        max_percentage_update=0.05,
        bound_consistency=True,
    )
    exploration = None
    experience = ExperienceReplay(
        maxlen=3 * ep_len,
        sample_size=2 * ep_len,
        include_latest=1 * ep_len,
        seed=seed,
    )
    hessian_type = "approx"
