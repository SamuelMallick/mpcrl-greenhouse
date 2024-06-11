import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from greenhouse.model import Model

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

days = 40
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps
seconds_in_time_step = 15 * 60
nx = 4
nu = 3
file_name = "results/nominal/nominal_greenhouse_rk4_True_19.pkl"
# file_name = "results/test_30_train.pkl"
p = Model.get_true_parameters()

with open(
    file_name,
    "rb",
) as file:
    data = pickle.load(file)

_, t_axs = plt.subplots(
    3 if "TD" in data else 2, 1, constrained_layout=True, sharex=True
)  # axes for plotting each timestep
_, ep_axs = plt.subplots(
    6 if "TD" in data else 5, 1, constrained_layout=True, sharex=True
)  # axes for plotting each episode
R_indx = 0
TD_indx = 1
VIOL_Y_indx = 2 if "TD" in data else 1
VIOL_U_indx = 3 if "TD" in data else 2
EPI_indx = 4 if "TD" in data else 3
YIELD_indx = 5 if "TD" in data else 4
# plot environment rewards
if "R" in data:
    R = data["R"]
    t_axs[R_indx].plot(
        R.flatten(), "o", markersize=1
    )  # plot the reward for each timestep of all episodes
    t_axs[R_indx].set_ylabel("$L_t$")
    ep_axs[R_indx].plot(
        np.sum(R, axis=1), "o", markersize=1
    )  # plot the total reward for each episode
    ep_axs[R_indx].set_ylabel("$L_{ep}$")

# plot TD error
if "TD" in data:
    TD = data["TD"]
    t_axs[TD_indx].plot(
        TD.flatten(), "o", markersize=1
    )  # plot the reward for each timestep of all episodes
    t_axs[TD_indx].set_ylabel(r"$\delta_t$")
    ep_axs[TD_indx].plot(
        np.nansum(TD, axis=1),
        "o",
        markersize=1,  # TODO replace nansum with something that captures the fact that nan is not good
    )  # plot the total reward for each episode
    ep_axs[TD_indx].set_ylabel(r"$\delta_{ep}$")

# calculate constraint violations
X = data["X"]
d = data["d"]
U = data["U"]
# generate output data from state data X
y = Model.output(X[:, :-1].transpose(2, 0, 1), p).transpose(1, 2, 0)
y_min = np.empty((X.shape[0], X.shape[1] - 1, y.shape[2]), dtype=X.dtype)
y_max = y_min.copy()
viols = y_min.copy()
for i, j in product(range(X.shape[0]), range(X.shape[1] - 1)):
    if i % 100 == 0 and j == 0:
        print(f"Calculating constraint violations for episode {i} timestep {j}")
    y_min[i, j, :] = Model.get_output_min(d[i, j, :])
    y_max[i, j, :] = Model.get_output_max(d[i, j, :])
    viol_lower = np.maximum(
        (y_min[i, j, :] - y[i, j, :]) / (y_max[i, j, :] - y_min[i, j, :]), 0
    )
    viol_upper = np.maximum(
        (y[i, j, :] - y_max[i, j, :]) / (y_max[i, j, :] - y_min[i, j, :]), 0
    )
    viols[i, j, :] = viol_lower.sum() + viol_upper.sum()

du_lim = Model.get_du_lim()
viols_du = np.maximum(0, (np.abs(U[:, 1:, :] - U[:, :-1, :]) - du_lim) / du_lim)

# plot constraint violations
t_axs[VIOL_Y_indx].plot(
    viols.reshape(-1, nx), "o", markersize=1
)  # plot the reward for each timestep of all episodes
t_axs[VIOL_Y_indx].set_ylabel("$viols$")
t_axs[VIOL_Y_indx].set_xlabel("Timestep")
ep_axs[VIOL_Y_indx].plot(
    np.sum(viols.reshape(-1, ep_len * nx), axis=1), "o", markersize=1
)  # plot the total reward for each episode
ep_axs[VIOL_Y_indx].set_ylabel("$viols_{ep}$")

ep_axs[VIOL_U_indx].plot(
    np.sum(viols_du.reshape(viols_du.shape[0], -1), axis=1), "o", markersize=1
)

# calculate ecomonmic performance index
c_co2 = 42e-2
c_q = 6.35e-9
c_pri_1 = 1.8
c_pri_2 = 16
EPI = np.zeros((X.shape[0]), dtype=float)
for i in range(X.shape[0]):
    final_yield = y[i, -1, 0] * 1e-3  # convert from g to kg
    EPI[i] = (
        c_pri_1
        + c_pri_2 * final_yield
        - seconds_in_time_step
        * (c_q * np.sum(U[i, :, 2]) + c_co2 * np.sum(U[i, :, 0]) * 1e-6)
    )  # converting co2 from mg to kg

# plot economic performance index
ep_axs[EPI_indx].plot(EPI, "o", markersize=1)  # plot the total reward for each episode
ep_axs[EPI_indx].set_ylabel("$EPI$")
ep_axs[EPI_indx].set_xlabel("Episode")

# plot yields
ep_axs[YIELD_indx].plot(
    y[:, -1, 0], "o", markersize=1
)  # plot the total lettuce yield for each episode
ep_axs[YIELD_indx].set_ylabel("$yield$")

# plot learnt parameters
if "param_dict" in data:
    param_dict = data["param_dict"]
    cost_keys = [x for x in list(param_dict.keys()) if x.startswith("p") is False]
    if len(cost_keys) > 0:
        _, axs = plt.subplots(len(cost_keys), 1, constrained_layout=True, sharex=True)
        if len(cost_keys) == 1:
            axs = [axs, None]
        for i in range(len(cost_keys)):
            axs[i].plot(
                [
                    (param_dict[cost_keys[i]][j]).squeeze()
                    for j in range(len(param_dict[cost_keys[i]]))
                ]
            )
            axs[i].set_ylabel(cost_keys[i])

        param_keys = [x for x in list(param_dict.keys()) if x.startswith("p") is True]
        num_figs = int(np.ceil(len(param_keys) / 5))
        for j in range(num_figs):
            _, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
            for i in range(5):
                if 5 * j + i < len(param_keys):
                    axs[i].plot(param_dict[param_keys[5 * j + i]])
                    axs[i].set_ylabel(param_keys[5 * j + i])

# plot first and last episodes
_, axs = plt.subplots(nx, 2, constrained_layout=True, sharex=True)
for i in range(2):
    for j in range(nx):
        axs[j, i].plot(y[0 if i == 0 else -1, :, j])
        if j > 0:
            axs[j, i].plot(y_min[0 if i == 0 else -1, :, j], color="black")
            axs[j, i].plot(y_max[0 if i == 0 else -1, :, j], color="r")
axs[0, 0].set_title("First ep")
axs[0, 1].set_title("Last ep")
axs[-1, 0].set_xlabel("Timestep")
axs[-1, 1].set_xlabel("Timestep")

U = data["U"]
u_min = Model.get_u_min()
u_max = Model.get_u_max()
U_min = np.tile(u_min, (U.shape[1], 1)).T
U_max = np.tile(u_max, (U.shape[1], 1)).T
_, axs = plt.subplots(nu, 2, constrained_layout=True, sharex=True)
for i in range(2):
    for j in range(nu):
        axs[j, i].plot(U[0 if i == 0 else -1, :, j])
        axs[j, i].plot(U_min[j, :], color="black")
        axs[j, i].plot(U_max[j, :], color="r")
axs[0, 0].set_title("First ep")
axs[0, 1].set_title("Last ep")
axs[-1, 0].set_xlabel("Timestep")
axs[-1, 1].set_xlabel("Timestep")

# plot disturbance profiles
_, axs = plt.subplots(nd, 1, constrained_layout=True, sharex=True)
for i in range(nd):
    axs[i].plot(d.reshape(-1, d.shape[2])[:, i])

plt.show()
