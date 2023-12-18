import pickle

import matplotlib.pyplot as plt
import numpy as np

from envs.model import get_y_max, get_y_min

plt.rc("text", usetex=True)

num_episodes = 50
days = 40
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps
nx = 4
nu = 3

with open(
    # "results/v_2_lr_1e-05_df_0_8.pkl",
    "green_V2_lr_1e-08_ne_50_df_0.826012504615801.pkl",
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
TD_eps = [sum(TD[ep_len * i : ep_len * (i + 1)]) / ep_len for i in range(num_episodes)]

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
_, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
axs[0].plot(yields, "o", markersize=1)
axs[0].set_ylabel(r"$yield$")
cnstr_viols = np.zeros((num_episodes, 1))
for i in range(num_episodes):
    for k in range(ep_len):
        y_max = get_y_max(d[:, [i * ep_len + k]])
        y_min = get_y_min(d[:, [i * ep_len + k]])
        # extra +i index in y because it has ep_len+1 entries for each ep
        if any(y[[i * (ep_len + 1) + k], :].reshape(4, 1) > y_max) or any(
            y[[i * (ep_len + 1) + k], :].reshape(4, 1) < y_min
        ):
            cnstr_viols[i, :] += 1
axs[1].plot(cnstr_viols, "o", markersize=1)
axs[1].set_ylabel(r"$cnstr viols$")

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
_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
axs[0].set_title("Last ep")
for i in range(4):
    axs[i].plot(y[-ep_len - 1 : -1, i])
    axs[i].plot(y_min[i, :], color="black")
    if i != 0:
        axs[i].plot(y_max[i, :], color="r")
# _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
# for i in range(3):
#    axs[i].plot(U[:, i])

# parameters
_, axs = plt.subplots(5, 2, constrained_layout=True, sharex=True)
axs[0, 0].plot(param_list["V0"])
axs[0, 0].set_ylabel(r"$V_0$")
axs[1, 0].plot([param_list["c_u"][k][0] for k in range(num_episodes)])
axs[1, 0].set_ylabel(r"$cu_1$")
axs[2, 0].plot([param_list["c_u"][k][1] for k in range(num_episodes)])
axs[2, 0].set_ylabel(r"$cu_2$")
axs[3, 0].plot([param_list["c_u"][k][2] for k in range(num_episodes)])
axs[3, 0].set_ylabel(r"$cu_3$")
axs[4, 0].plot(param_list["c_y"])
axs[4, 0].set_ylabel(r"$cy$")

axs[0, 1].plot(param_list["p_1"])
axs[0, 1].set_ylabel(r"$p_1$")
axs[1, 1].plot(param_list["p_2"])
axs[1, 1].set_ylabel(r"$p_2$")
axs[2, 1].plot(param_list["p_3"])
axs[2, 1].set_ylabel(r"$p_3$")
axs[3, 1].plot(param_list["p_0"])
axs[3, 1].set_ylabel(r"$p_0$")
if "c_dy" in param_list.keys():
    axs[4, 1].plot(param_list["c_dy"])
    axs[4, 1].set_ylabel(r"$c_dy$")


plt.show()
