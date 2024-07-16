import os
import pickle
import sys

sys.path.append(os.getcwd())

import re
from pathlib import Path

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from agents.ddpg_agent import make_env
from utils.plot import plot_greenhouse

# main options
folder = "results/ddpg_lr1e-5"
n_eval_episodes = 100
days_per_episode = 40
device = "cpu"
set_random_seed(1, using_cuda=device.startswith("cuda"))
STORE_DATA = True
PLOT = False


def evaluate_single(filename: str) -> None:
    """Evaluates a single DDPG model from a given file."""
    # load the DDPG model
    model = DDPG.load(filename, device=device)

    # first, create a fresh environment for evaluation
    eval_env, _ = make_env(float("nan"), days_per_episode, evaluation=True)

    # then, replace the fresh `VecNormalize` wrapper with the saved one
    env_filename = re.sub(r"ddpg_agent_(\d+)\.zip", r"ddpg_env_\1.pkl", str(filename))
    eval_env_loaded = VecNormalize.load(env_filename, eval_env.venv)
    eval_env_loaded.training = False  # set to evaluation mode

    # create an evaluation environment and launch evaluation
    evaluate_policy(model, eval_env, n_eval_episodes=n_eval_episodes)

    # extract our `MonitorEpisodes` wrapper from the SB3 vectorized env
    eval_env = eval_env.envs[0].env.env.env
    return {
        "name": os.path.splitext(filename)[0] + "_eval_final",
        "X": np.asarray(eval_env.observations),
        "U": np.asarray(eval_env.actions),
        "R": np.asarray(eval_env.rewards),
        "d": np.asarray(eval_env.disturbance_profiles_all_episodes).transpose(0, 2, 1),
    }


# find in the given folder all agents' .zip files, and evaluate each of them - each must
# have a corresponding env's .pkl file with the same naming convention
data = [evaluate_single(fn) for fn in Path(folder).glob("ddpg_*.zip")]


# storing and plotting
if STORE_DATA:
    for datum in data:
        with open(datum["name"] + ".pkl", "wb") as file:
            pickle.dump(datum, file)
if PLOT:
    for datum in data:
        plot_greenhouse(datum["X"], datum["U"], datum["d"], datum["R"])
