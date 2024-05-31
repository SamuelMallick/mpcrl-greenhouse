import matplotlib.pyplot as plt
import numpy as np

from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model


def plot_greenhouse(
    X: np.ndarray,
    U: np.ndarray,
    d: np.ndarray,
    R: np.ndarray,
    TD: np.ndarray | None = None,
):
    """Plot the greenhouse data.

    Parameters
    ----------
    X : np.ndarray
        The state data.
    U : np.ndarray
        The action data.
    d : np.ndarray
        The disturbance data.
    R : np.ndarray
        The reward data.
    TD : np.ndarray | None, optional
        The td error data."""
    # generate output data from state data X
    y = np.empty((X.shape[0], X.shape[1] - 1, X.shape[2]), dtype=X.dtype)
    y_min = y.copy()
    y_max = y.copy()
    p = Model.get_true_parameters()
    for i in range(X.shape[0]):
        for j in range(X.shape[1] - 1):
            y[i, j, :] = Model.output(X[i, j, :], p)
            y_min[i, j, :] = Model.get_output_min(d[i, j, :])
            y_max[i, j, :] = Model.get_output_max(d[i, j, :])

    # plot TD and reward
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    if TD is not None:
        TD_flat = TD.reshape(-1)
        axs[0].plot(TD_flat, "o", markersize=1)
    R_flat = R.reshape(-1)
    axs[1].plot(R_flat, "o", markersize=1)
    axs[0].set_ylabel(r"$\tau$")
    axs[1].set_ylabel("$L$")

    nx = LettuceGreenHouse.nx
    _, axs = plt.subplots(nx, 1, constrained_layout=True, sharex=True)
    X_flat = X.reshape(-1, nx)
    for i in range(nx):
        axs[i].plot(X_flat[:, i])
    _, axs = plt.subplots(nx, 1, constrained_layout=True, sharex=True)
    y_flat = y.reshape(-1, nx)
    y_min_flat = y_min.reshape(-1, nx)
    y_max_flat = y_max.reshape(-1, nx)
    for i in range(nx):
        axs[i].plot(y_flat[:, i])
        axs[i].plot(y_min_flat[:, i], color="black")
        if i != 0:
            axs[i].plot(y_max_flat[:, i], color="r")

    nu = LettuceGreenHouse.nu
    _, axs = plt.subplots(nu, 1, constrained_layout=True, sharex=True)
    U_flat = U.reshape(-1, nu)
    u_min, u_max = Model.get_u_min(), Model.get_u_max()
    for i in range(nu):
        axs[i].plot(U_flat[:, i])
        axs[i].axhline(u_min[i], color="black")
        axs[i].axhline(u_max[i], color="r")

    nd = LettuceGreenHouse.nd
    _, axs = plt.subplots(nd, 1, constrained_layout=True, sharex=True)
    d_flat = d.reshape(-1, nd)
    for i in range(nd):
        axs[i].plot(d_flat[:, i])
    plt.show()
