import matplotlib.pyplot as plt
import numpy as np

from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model

nx = LettuceGreenHouse.nx
u_min, u_max = Model.get_u_min(), Model.get_u_max()


def plot_greenhouse(X, U, y, d, TD, R, num_episodes, ep_len):
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(TD, "o", markersize=1)
    axs[1].plot(R, "o", markersize=1)
    axs[0].set_ylabel(r"$\tau$")
    axs[1].set_ylabel("$L$")

    # get bounds
    y_min = np.zeros((nx, num_episodes * ep_len))
    y_max = np.zeros((nx, num_episodes * ep_len))
    for t in range(num_episodes * ep_len):
        y_min[:, [t]] = Model.get_output_min(d[:, [t]])
        y_max[:, [t]] = Model.get_output_max(d[:, [t]])

    _, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
    for i in range(4):
        axs[i].plot(X[:, i])
    _, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
    for i in range(4):
        axs[i].plot(y[:, i])
        axs[i].plot(y_min[i, :], color="black")
        if i != 0:
            axs[i].plot(y_max[i, :], color="r")
    _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    for i in range(3):
        axs[i].plot(U[:, i])
        axs[i].axhline(u_min[i], color="black")
        axs[i].axhline(u_max[i], color="r")
    _, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
    for i in range(4):
        axs[i].plot(d[i, :])
    plt.show()
