import os
import pickle
import sys

sys.path.append(os.getcwd())

import re
from pathlib import Path
from time import perf_counter

import numpy as np
from gymnasium.wrappers.normalize import RunningMeanStd
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from agents.ddpg_agent import make_env
from utils.plot import plot_greenhouse

# read which folder to evaluate from the command line
if len(sys.argv) != 2:
    raise ValueError("Please provide a folder to evaluate the agents from.")
folder = Path(sys.argv[1])
assert folder.is_dir(), f"Provided path {folder} is not a directory."

# main options
n_eval_episodes = 100
days_per_episode = 40
device = "cuda:0"
set_random_seed(1, using_cuda=device.startswith("cuda"))
STORE_DATA = True
PLOT = False

# measure how long it takes to make each prediction - no need to use torch.cuda.Event
# because the predictions are always brought back to CPU by SB3
inference_time_rms = RunningMeanStd(epsilon=0)


def timing_wrapper(func):
    def wrapper(*args, **kwargs):
        start = perf_counter()
        out = func(*args, **kwargs)
        inference_time_rms.update(np.asarray([perf_counter() - start]))
        return out

    return wrapper


def evaluate_single(filename: Path) -> None:
    """Evaluates a single DDPG model from a given file."""
    # load the DDPG model and allow for timings measurements
    model = DDPG.load(filename, device=device)
    model.policy._predict = timing_wrapper(model.policy._predict)

    # first, create a fresh environment for evaluation
    eval_env, _ = make_env(float("nan"), days_per_episode, evaluation=True)

    # then, replace the fresh `VecNormalize` wrapper with the saved one
    env_filename = re.sub(r"ddpg_agent_(\d+)\.zip", r"ddpg_env_\1.pkl", str(filename))
    eval_env_loaded = VecNormalize.load(env_filename, eval_env.venv)
    eval_env_loaded.training = False  # set to evaluation mode

    # create an evaluation environment and launch evaluation
    evaluate_policy(model, eval_env_loaded, n_eval_episodes=n_eval_episodes)

    # extract our `MonitorEpisodes` wrapper from the SB3 vectorized env
    eval_env = eval_env.envs[0].env.env.env
    return {
        "name": os.path.splitext(filename)[0] + "_eval_learned",
        "X": np.asarray(eval_env.observations),
        "U": np.asarray(eval_env.actions),
        "R": np.asarray(eval_env.rewards),
        "d": np.asarray(eval_env.disturbance_profiles_all_episodes).transpose(0, 2, 1),
    }


# find in the given folder all agents' .zip files, and evaluate each of them - each must
# have a corresponding env's .pkl file with the same naming convention
data = [evaluate_single(fn) for fn in folder.glob("ddpg_*.zip")]

# print timings
mean = inference_time_rms.mean
std = np.sqrt(inference_time_rms.var)
count = int(inference_time_rms.count)
print(f"timings = {mean:e} +/- {std:e} seconds per iter (count={count})")

# storing and plotting
if STORE_DATA:
    for datum in data:
        with open(datum["name"] + ".pkl", "wb") as file:
            pickle.dump(datum, file)
    np.savez(folder / "timings_learned", mean=mean, std=std, count=count)
if PLOT:
    for datum in data:
        plot_greenhouse(datum["X"], datum["U"], datum["d"], datum["R"])
