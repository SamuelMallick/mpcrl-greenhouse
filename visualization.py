import pickle
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model

plt.style.use("bmh")


# load data if available
if len(sys.argv) <= 1:
    raise RuntimeError("Please provide a file name to visualize, you donkey")
with open(sys.argv[1], "rb") as file:
    data = pickle.load(file)

# define some constants
days = 40
ep_len = days * LettuceGreenHouse.steps_per_day
seconds_in_time_step = 15 * 60
nx = 4
nu = 3
nd = 4
p = Model.get_true_parameters()

# create axes for plotting each timestep's and each episode's quantities
TD_in_data = "TD" in data
t_fig, t_axs = plt.subplots(
    3 if TD_in_data else 2, 1, constrained_layout=True, sharex=True
)
e_fig, e_axs = plt.subplots(
    6 if TD_in_data else 5, 1, constrained_layout=True, sharex=True
)
t_fig.suptitle("Timestep-wise Learning")
e_fig.suptitle("Episode-wise Learning")
if TD_in_data:
    R_indx, TD_indx, VIOL_Y_indx, VIOL_U_indx, EPI_indx, YIELD_indx = range(6)
else:
    R_indx, VIOL_Y_indx, VIOL_U_indx, EPI_indx, YIELD_indx = range(5)

# plot environment rewards
if "R" in data:
    R = data["R"]
    t_axs[R_indx].plot(R.flatten(), "o", markersize=1)
    t_axs[R_indx].set_ylabel("$L_t$")
    e_axs[R_indx].plot(R.sum(1), "o", markersize=1)
    e_axs[R_indx].set_ylabel("$L_{ep}$")

# plot TD error - TODO: replace with something that captures the fact nan is not good
if TD_in_data:
    TD = data["TD"]
    t_axs[TD_indx].plot(TD.flatten(), "o", markersize=1)
    t_axs[TD_indx].set_ylabel(r"$\delta_t$")
    e_axs[TD_indx].plot(np.nansum(TD, 1), "o", markersize=1)
    e_axs[TD_indx].set_ylabel(r"$\delta_{ep}$")

# calculate and plot constraint violations
X = data["X"]
d = data["d"]
U = data["U"]
y = Model.output(X[:, :-1].transpose(2, 0, 1), p).transpose(1, 2, 0)
mask = d[: X.shape[0], ..., 0, None] < 10
y_min = np.where(mask, [[[0, 0, 10, 0]]], [[[0, 0, 15, 0]]])
y_max = np.where(mask, [[[1e6, 1.6, 15, 70]]], [[[1e6, 1.6, 20, 70]]])
viols_lb = np.maximum(0, (y_min - y) / (y_max - y_min))
viols_ub = np.maximum(0, (y - y_max) / (y_max - y_min))
viols = (viols_lb + viols_ub).sum(-1)
du_lim = Model.get_du_lim()
viols_du = np.maximum(0, (np.abs(U[:, 1:, :] - U[:, :-1, :]) - du_lim) / du_lim)
t_axs[VIOL_Y_indx].plot(viols.reshape(-1), "o", markersize=1)
t_axs[VIOL_Y_indx].set_ylabel("$viols$")
t_axs[VIOL_Y_indx].set_xlabel("Timestep")
e_axs[VIOL_Y_indx].plot(viols.sum(1), "o", markersize=1)
e_axs[VIOL_Y_indx].set_ylabel("$viols_{ep}$")
e_axs[VIOL_U_indx].plot(
    viols_du.reshape(viols_du.shape[0], -1).sum(1), "o", markersize=1
)
e_axs[VIOL_U_indx].set_ylabel("$viols^{du}_{ep}$")

# calculate and plot ecomonmic performance index
c_co2 = 42e-2
c_q = 6.35e-9
c_pri_1 = 1.8
c_pri_2 = 16
final_yields = y[:, -1, 0]
EPI = (
    c_pri_1
    + 1e-3 * c_pri_2 * final_yields
    - seconds_in_time_step * (c_q * U[..., 2].sum(1) + 1e-6 * c_co2 * U[..., 0].sum(1))
)
e_axs[EPI_indx].plot(EPI, "o", markersize=1)
e_axs[EPI_indx].set_ylabel("$EPI$")
e_axs[EPI_indx].set_xlabel("Episode")

# plot final yields
e_axs[YIELD_indx].plot(final_yields, "o", markersize=1)
e_axs[YIELD_indx].set_ylabel("$yield$")

# plot learnt parameters
if "param_dict" in data:
    param_dict = data["param_dict"]
    cost_params = [k for k in param_dict if not k.startswith("p")]
    dyn_params = [k for k in param_dict if k.startswith("p")]

    for group, name in (
        (cost_params, "Cost Parameters"),
        (dyn_params, "Dynamics Parameters"),
    ):
        N = len(group)
        if N <= 0:
            continue
        ncols = min(N, 4)
        nrows = int(np.ceil(N / ncols))
        fig, axs = plt.subplots(nrows, ncols, constrained_layout=True, sharex=True)
        fig.suptitle(name)
        for ax, key in zip(np.atleast_2d(axs).flat, group):
            ax.plot(param_dict[key])
            ax.set_ylabel(key)

# plot first and last episodes' outputs
fig, axs = plt.subplots(nx, 2, constrained_layout=True, sharex=True, sharey="row")
for (axs_, idx), j in product(zip(axs.T, [0, -1]), range(nx)):
    axs_[j].plot(y[idx, :, j])
    if j > 0:
        axs_[j].plot(y_min[idx, :, j], color="black")
        axs_[j].plot(y_max[idx, :, j], color="r")
fig.suptitle("Outputs")
axs[0, 0].set_title("First ep")
axs[0, 1].set_title("Last ep")
axs[-1, 0].set_xlabel("Timestep")
axs[-1, 1].set_xlabel("Timestep")

# plot first and last episodes' control inputs
U_min = np.tile(Model.get_u_min(), (U.shape[1], 1))
U_max = np.tile(Model.get_u_max(), (U.shape[1], 1))
fig, axs = plt.subplots(nu, 2, constrained_layout=True, sharex=True, sharey="row")
for (axs_, idx), j in product(zip(axs.T, [0, -1]), range(nu)):
    axs_[j].plot(U[idx, :, j])
    axs_[j].plot(U_min[..., j], color="black")
    axs_[j].plot(U_max[..., j], color="r")
fig.suptitle("Control Actions")
axs[0, 0].set_title("First ep")
axs[0, 1].set_title("Last ep")
axs[-1, 0].set_xlabel("Timestep")
axs[-1, 1].set_xlabel("Timestep")

# plot disturbance profiles
fig, axs = plt.subplots(nd, 1, constrained_layout=True, sharex=True)
for i in range(nd):
    # axs[i].plot(d.reshape(-1, d.shape[2])[:, i])
    axs[i].plot(d[0, :, i].reshape(-1))
fig.suptitle("Disturbances")

# # set axis ranges
# t_axs[VIOL_Y_indx].set_ylim(0, 2)
# e_axs[VIOL_Y_indx].set_ylim(0, 1200)
# t_axs[TD_indx].set_ylim(-5e7, 5e7)
# e_axs[TD_indx].set_ylim(-10e9, 10e9)
# t_axs[R_indx].set_ylim(0, 2e5)
# e_axs[R_indx].set_ylim(0, 2e8)

plt.show()
