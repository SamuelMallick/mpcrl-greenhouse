import pickle
from collections.abc import Collection, Iterator
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from mpcrl.wrappers.envs import MonitorEpisodes

from agents.ddpg_agent import train_ddpg
from utils.plot import plot_greenhouse


def do_training(
    episodes: int,
    days_per_episode: int,
    n_agents: int = 1,
    seed: int | None = None,
    devices: str | Collection[str] = "auto",
) -> Iterator[tuple[MonitorEpisodes, MonitorEpisodes]]:
    """Launches the training of `n_agents` DDPG agents in parallel."""
    if isinstance(devices, str):
        devices = (devices,)
    seeds = np.random.SeedSequence(seed).generate_state(n_agents)

    def fun(n: int):
        return train_ddpg(
            episodes=episodes,
            days_per_episode=days_per_episode,
            learning_rate=1e-3,
            gradient_threshold=1.0,
            l2_regularization=1e-5,
            batch_size=64,
            buffer_size=10_000,
            gamma=0.99,  # NOTE: different from Morcego et al.
            seed=int(seeds[n]),
            device=devices[n % len(devices)],
            verbose=1,
        )

    return Parallel(n_jobs=n_agents, return_as="generator_unordered")(
        delayed(fun)(i) for i in range(n_agents)
    )


def process_simulations(
    sims: Iterator[tuple[MonitorEpisodes, MonitorEpisodes]]
) -> dict[str, dict[str, np.ndarray]]:
    """Extracts the simulations' data into numpy arrays."""
    train_envs, eval_envs = zip(*sims)
    data = {}
    for env_type, envs in [("train", train_envs), ("eval", eval_envs)]:
        X = np.asarray([env.observations for env in envs])  # agents x ep x T x nx
        U = np.asarray([env.actions for env in envs])  # agents x ep x T x nu
        R = np.asarray([env.rewards for env in envs])  # agents x ep x T
        D = [env.get_wrapper_attr("disturbance_profiles_all_episodes") for env in envs]
        D = np.asarray(D).swapaxes(-1, -2)  # agents x ep x T x nd
        data[env_type] = {"X": X, "U": U, "R": R, "D": D}
    return data


def do_plotting(
    data: dict[str, dict[str, np.ndarray]],
    agent_to_plot: int,
    env_type: Literal["train", "eval"],
) -> None:
    """Plots the greenhouse state of the agent `agent_to_plot` in the environment
    `env_type`."""
    i = agent_to_plot
    d = data[env_type]
    plot_greenhouse(d["X"][i], d["U"][i], d["D"][i], d["R"][i], None)


def store_data(
    data: dict[str, dict[str, np.ndarray]], identifier: str = "ddpg"
) -> None:
    """Stores the simulation data to disk."""
    for env_type in ("train", "eval"):
        X = data[env_type]["X"]
        U = data[env_type]["U"]
        D = data[env_type]["D"]
        R = data[env_type]["R"]
        for i in range(X.shape[0]):
            fn = f"{identifier}_{env_type}_{i}"
            with open(f"{fn}.pkl", "wb") as file:
                pickle.dump(
                    {"name": fn, "X": X[i], "U": U[i], "d": D[i], "R": R[i]}, file
                )


if __name__ == "__main__":
    # launch training
    simulations = do_training(
        episodes=2000, days_per_episode=40, n_agents=1, seed=1, devices="cuda:2"
    )

    # process and plot or store data
    PLOT = False
    STORE_DATA = True
    if PLOT or STORE_DATA:
        simdata = process_simulations(simulations)

        if PLOT:
            # for now, can only plot one agent from one type
            do_plotting(data=simdata, agent_to_plot=0, env_type="train")

        if STORE_DATA:
            # for now, each agent from each type is saved to a separate file
            store_data(simdata)
