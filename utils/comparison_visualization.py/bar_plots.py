import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from greenhouse.model import Model

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

days = 40
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps
seconds_in_time_step = 15 * 60
nx = 4
nu = 3
file_names = [
    "results/nominal/nominal_greenhouse_rk4_True_18.pkl",
    # "results/nominal/nominal_greenhouse_rk4_False_0.pkl",
    "results/sample/sample_greenhouse_rk4_2_1_18.pkl",
    # "results/sample/sample_greenhouse_rk4_5_1_0.pkl",
    # "results/sample/sample_greenhouse_rk4_10_1_0.pkl",
    # "results/sample/sample_greenhouse_rk4_20_1_0.pkl",
]
p = Model.get_true_parameters()

data = []
for file_name in file_names:
    with open(
        file_name,
        "rb",
    ) as file:
        data.append(pickle.load(file))

_, ep_axs = plt.subplots(
    4, 1, constrained_layout=True, sharex=True
)  # axes for plotting each episode
R_indx = 0
VIOL_indx = 1
YIELD_indx = 2
EPI_indx = 3
# plot environment rewards
R = [o["R"] for o in data]
ep_axs[R_indx].boxplot(
    [np.sum(o, axis=1) for o in R],
)  # plot the total reward for each episode
ep_axs[R_indx].set_yscale("log")
ep_axs[R_indx].set_ylabel("$L$")

# calculate constraint violations
X = [o["X"] for o in data]
d = [o["d"] for o in data]
# generate output data from state data X
y = [
    np.empty((X[i].shape[0], X[i].shape[1] - 1, X[i].shape[2]), dtype=X[i].dtype)
    for i in range(len(data))
]
y_min = [y[i].copy() for i in range(len(data))]
y_max = [y[i].copy() for i in range(len(data))]
viols = [y[i].copy() for i in range(len(data))]
for i in range(y[0].shape[0]):
    for z in range(len(data)):
        for j in range(X[z].shape[1] - 1):
            y[z][i, j, :] = Model.output(X[z][i, j, :], p)
            y_min[z][i, j, :] = Model.get_output_min(d[z][i, j, :])
            y_max[z][i, j, :] = Model.get_output_max(d[z][i, j, :])
            viol_lower = np.maximum(
                (y_min[z][i, j, :] - y[z][i, j, :])
                / (y_max[z][i, j, :] - y_min[z][i, j, :]),
                0,
            )
            viol_upper = np.maximum(
                (y[z][i, j, :] - y_max[z][i, j, :])
                / (y_max[z][i, j, :] - y_min[z][i, j, :]),
                0,
            )
            viols[z][i, j, :] = viol_upper.sum() + viol_lower.sum()

# plot constraint violations
ep_axs[VIOL_indx].boxplot(
    [np.sum(viols[i].reshape(viols[i].shape[0], -1), axis=1) for i in range(len(data))]
)  # plot the total reward for each episode
ep_axs[VIOL_indx].set_yscale("log")
ep_axs[VIOL_indx].set_ylabel("$viols$")

# plot yields
ep_axs[YIELD_indx].boxplot(
    [y[i][:, -1, 0] for i in range(len(data))],
)  # plot the total reward for each episode
# ep_axs[YIELD_indx].set_yscale("log")
ep_axs[YIELD_indx].set_ylabel("$yield$")

# calculate ecomonmic performance index
U = [o["U"] for o in data]
c_co2 = 42e-2
c_q = 6.35e-9
c_pri_1 = 1.8
c_pri_2 = 16
final_yield = [o[:, -1, 0] * 1e-3 for o in y]  # convert from g to kg
EPI = [
    (
        c_pri_1
        + c_pri_2 * final_yield[i]
        - seconds_in_time_step
        * (
            c_q * np.sum(U[i][:, :, 2], axis=1)
            + c_co2 * np.sum(U[i][:, :, 0], axis=1) * 1e-6
        )
    )
    for i in range(len(data))
]  # converting co2 from mg to kg

# plot economic performance index
ep_axs[EPI_indx].boxplot(
    EPI,
    # tick_label=[
    #     "nom perfect",
    #     "nom",
    #     "scenario 2",
    #     "scenario 5",
    #     "scenario 10",
    #     "scenario 20",
    # ],
)  # plot the total reward for each episode
ep_axs[EPI_indx].set_ylabel("$EPI$")


plt.show()
