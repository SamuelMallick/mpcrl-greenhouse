import pickle

import matplotlib.pyplot as plt
import numpy as np

from envs.model import get_y_max, get_y_min

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

num_episodes = 100
days = 40
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps
nx = 4
nu = 3
allp = True
rk4 = True
lr = 0.0001

with open(
    "results/test_26.pkl",
    # "results/sample/sample_5.pkl",
    # "results/nominal/disturbance_profiles/nom_first_day_200.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
    y = pickle.load(file)
    d = pickle.load(file)
    R = pickle.load(file)
    TD = pickle.load(file)
    param_list = pickle.load(file)

R_eps = [sum(R[ep_len * i : ep_len * (i + 1)]) for i in range(num_episodes)]
TD_eps = [
    np.nansum(TD[ep_len * i : ep_len * (i + 1)]) / ep_len for i in range(num_episodes)
]

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD, "o", markersize=1)
axs[1].plot(R, "o", markersize=1)
axs[0].set_ylabel(r"$\tau$")
axs[1].set_ylabel("$L$")

_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(TD_eps, "o", markersize=1)
axs[1].plot(R_eps, "o", markersize=1)
axs[0].set_ylabel(r"$\tau (ep)$")
axs[1].set_ylabel("$L (ep)$")

# yields
yields = [y[(i + 1) * ep_len + i, 0] for i in range(num_episodes)]
_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(yields, "o", markersize=1)
axs[0].set_ylabel(r"$yield$")

num_cnstr_viols = np.zeros((num_episodes, 1))
mag_cnstr_viols = np.zeros((num_episodes, 1))
for i in range(num_episodes):
    for k in range(ep_len):
        y_max = get_y_max(d[:, [i * ep_len + k]])
        y_min = get_y_min(d[:, [i * ep_len + k]])
        # extra +i index in y because it has ep_len+1 entries for each ep
        if any(y[[i * (ep_len + 1) + k], :].reshape(4, 1) > y_max) or any(
            y[[i * (ep_len + 1) + k], :].reshape(4, 1) < y_min
        ):
            num_cnstr_viols[i, :] += 1
            y_below = y[[i * (ep_len + 1) + k], :].reshape(4, 1) - y_min
            y_below[y_below > 0] = 0
            y_above = y_max - y[[i * (ep_len + 1) + k], :].reshape(4, 1)
            y_above[y_above > 0] = 0
            mag_cnstr_viols[i, :] += np.linalg.norm(y_above + y_below, ord=2)
axs[1].plot(num_cnstr_viols, "o", markersize=1)
axs[1].set_ylabel(r"$num cnstr viols$")
axs[2].plot(mag_cnstr_viols, "o", markersize=1)
axs[2].set_ylabel(r"$mag cnstr viols$")

# states and input
# first ep
# get bounds
y_min = np.zeros((nx, ep_len))
y_max = np.zeros((nx, ep_len))
for k in range(ep_len):
    y_min[:, [k]] = get_y_min(d[:, [k]])
    y_max[:, [k]] = get_y_max(d[:, [k]])
_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
axs[0].set_title("First ep")
for i in range(4):
    axs[i].plot(y[:ep_len, i])
    axs[i].plot(y_min[i, :], color="black")
    if i != 0:
        axs[i].plot(y_max[i, :], color="r")

# last ep
# get bounds
y_min = np.zeros((nx, ep_len))
y_max = np.zeros((nx, ep_len))
for k in range(ep_len):
    y_min[:, [k]] = get_y_min(d[:, [-ep_len + k]])
    y_max[:, [k]] = get_y_max(d[:, [-ep_len + k]])
# _, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
axs[0].set_title("Last ep")
for i in range(4):
    axs[i].plot(y[-ep_len - 1 : -1, i])
    axs[i].plot(y_min[i, :], color="black")
    if i != 0:
        axs[i].plot(y_max[i, :], color="r")
# _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
# for i in range(3):
#    axs[i].plot(U[:, i])

# parameters - first cost params
cost_keys = [x for x in list(param_list.keys()) if x.startswith("p") is False]
if len(cost_keys) > 0:
    _, axs = plt.subplots(len(cost_keys), 1, constrained_layout=True, sharex=True)
    if len(cost_keys) == 1:
        axs = [axs, None]
    for i in range(len(cost_keys)):
        axs[i].plot(
            [
                (param_list[cost_keys[i]][j]).squeeze()
                for j in range(len(param_list[cost_keys[i]]))
            ]
        )
        axs[i].set_ylabel(cost_keys[i])

    param_keys = [x for x in list(param_list.keys()) if x.startswith("p") is True]
    num_figs = int(np.ceil(len(param_keys) / 5))
    for j in range(num_figs):
        _, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
        for i in range(5):
            if 5 * j + i < len(param_keys):
                axs[i].plot(param_list[param_keys[5 * j + i]])
                axs[i].set_ylabel(param_keys[5 * j + i])

plt.show()
