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
    "results/nominal/nominal_greenhouse_rk4_True.pkl",
    "results/nominal/nominal_greenhouse_rk4_False.pkl",
    "results/sample/sample_greenhouse_rk4_2_1.pkl",
    "results/sample/sample_greenhouse_rk4_5_1.pkl",
    "results/sample/sample_greenhouse_rk4_10_1.pkl",
]
p = Model.get_true_parameters()
ep_num = 0

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
ep_axs[R_indx].bar(
    list(range(len(R))),
    [np.sum(o[ep_num], axis=0) for o in R],
    color=[f"C{i}" for i in range(len(R))],
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
for z in range(len(data)):
    for j in range(X[z].shape[1] - 1):
        y[z][ep_num, j, :] = Model.output(X[z][ep_num, j, :], p)
        y_min[z][ep_num, j, :] = Model.get_output_min(d[z][ep_num, j, :])
        y_max[z][ep_num, j, :] = Model.get_output_max(d[z][ep_num, j, :])
        viols[z][ep_num, j, :] = np.maximum(
            np.maximum(y_min[z][ep_num, j, :] - y[z][ep_num, j, :], 0),
            np.maximum(y[z][ep_num, j, :] - y_max[z][ep_num, j, :], 0),
        )

# plot constraint violations
ep_axs[VIOL_indx].bar(
    list(range(len(X))),
    [np.sum(viols[i][ep_num].flatten()) for i in range(len(data))],
    color=[f"C{i}" for i in range(len(R))],
)  # plot the total reward for each episode
ep_axs[VIOL_indx].set_yscale("log")
ep_axs[VIOL_indx].set_ylabel("$viols$")

# plot yields
ep_axs[YIELD_indx].bar(
    list(range(len(y))),
    [y[i][ep_num, -1, 0] for i in range(len(data))],
    color=[f"C{i}" for i in range(len(R))],
)  # plot the total reward for each episode
# ep_axs[YIELD_indx].set_yscale("log")
ep_axs[YIELD_indx].set_ylabel("$yield$")

# calculate ecomonmic performance index
U = [o["U"] for o in data]
c_co2 = 42e-2
c_q = 6.35e-9
c_pri_1 = 1.8
c_pri_2 = 16
EPI = np.zeros((len(U)), dtype=float)
for i in range(len(U)):
    final_yield = y[i][ep_num, -1, 0] * 1e-3  # convert from g to kg
    EPI[i] = (
        c_pri_1
        + c_pri_2 * final_yield
        - seconds_in_time_step
        * (c_q * np.sum(U[i][ep_num, :, 2]) + c_co2 * np.sum(U[i][ep_num, :, 0]) * 1e-6)
    )  # converting co2 from mg to kg

# plot economic performance index
ep_axs[EPI_indx].bar(
    list(range(len(y))),
    EPI,
    tick_label=["nom perfect", "nom", "scenario 2", "scenario 5", "scenario 10"],
    color=[f"C{i}" for i in range(len(R))],
)  # plot the total reward for each episode
ep_axs[EPI_indx].set_ylabel("$EPI$")


plt.show()
