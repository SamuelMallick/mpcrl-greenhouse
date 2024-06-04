import pickle

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
file_name = "results/nominal/nominal_greenhouse_rk4_False.pkl"
# file_name = "results/sample/sample_greenhouse_rk4_2_1.pkl"
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
    4 if "TD" in data else 3, 1, constrained_layout=True, sharex=True
)  # axes for plotting each episode
R_indx = 0
TD_indx = 1
VIOL_indx = 2 if "TD" in data else 1
EPI_indx = 3 if "TD" in data else 2
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
        np.sum(TD, axis=1), "o", markersize=1
    )  # plot the total reward for each episode
    ep_axs[TD_indx].set_ylabel(r"$\delta_{ep}$")

# calculate constraint violations
X = data["X"]
d = data["d"]
# generate output data from state data X
y = np.empty((X.shape[0], X.shape[1] - 1, X.shape[2]), dtype=X.dtype)
y_min = y.copy()
y_max = y.copy()
viols = y.copy()
for i in range(X.shape[0]):
    for j in range(X.shape[1] - 1):
        y[i, j, :] = Model.output(X[i, j, :], p)
        y_min[i, j, :] = Model.get_output_min(d[i, j, :])
        y_max[i, j, :] = Model.get_output_max(d[i, j, :])
        viols[i, j, :] = np.maximum(
            np.maximum(y_min[i, j, :] - y[i, j, :], 0),
            np.maximum(y[i, j, :] - y_max[i, j, :], 0),
        )

# plot constraint violations
t_axs[VIOL_indx].plot(
    viols.reshape(-1, nx), "o", markersize=1
)  # plot the reward for each timestep of all episodes
t_axs[VIOL_indx].set_ylabel("$viols$")
t_axs[VIOL_indx].set_xlabel("Timestep")
ep_axs[VIOL_indx].plot(
    np.sum(viols.reshape(-1, ep_len * nx), axis=1), "o", markersize=1
)  # plot the total reward for each episode
ep_axs[VIOL_indx].set_ylabel("$viols_{ep}$")

# calculate ecomonmic performance index
U = data["U"]
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

print(f"Yields: {y[:, -1, 0]}")

# # yields
# yields = [y[(i + 1) * ep_len + i, 0] for i in range(num_episodes)]
# _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
# axs[0].plot(yields, "o", markersize=1)
# axs[0].set_ylabel(r"$yield$")

# num_cnstr_viols = np.zeros((num_episodes, 1))
# mag_cnstr_viols = np.zeros((num_episodes, 1))
# for i in range(num_episodes):
#     for k in range(ep_len):
#         y_max = get_y_max(d[:, [i * ep_len + k]])
#         y_min = get_y_min(d[:, [i * ep_len + k]])
#         # extra +i index in y because it has ep_len+1 entries for each ep
#         if any(y[[i * (ep_len + 1) + k], :].reshape(4, 1) > y_max) or any(
#             y[[i * (ep_len + 1) + k], :].reshape(4, 1) < y_min
#         ):
#             num_cnstr_viols[i, :] += 1
#             y_below = y[[i * (ep_len + 1) + k], :].reshape(4, 1) - y_min
#             y_below[y_below > 0] = 0
#             y_above = y_max - y[[i * (ep_len + 1) + k], :].reshape(4, 1)
#             y_above[y_above > 0] = 0
#             mag_cnstr_viols[i, :] += np.linalg.norm(y_above + y_below, ord=2)
# axs[1].plot(num_cnstr_viols, "o", markersize=1)
# axs[1].set_ylabel(r"$num cnstr viols$")
# axs[2].plot(mag_cnstr_viols, "o", markersize=1)
# axs[2].set_ylabel(r"$mag cnstr viols$")

# # states and input
# # first ep
# # get bounds
# y_min = np.zeros((nx, ep_len))
# y_max = np.zeros((nx, ep_len))
# for k in range(ep_len):
#     y_min[:, [k]] = get_y_min(d[:, [k]])
#     y_max[:, [k]] = get_y_max(d[:, [k]])
# _, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
# axs[0].set_title("First ep")
# for i in range(4):
#     axs[i].plot(y[:ep_len, i])
#     axs[i].plot(y_min[i, :], color="black")
#     if i != 0:
#         axs[i].plot(y_max[i, :], color="r")

# # last ep
# # get bounds
# y_min = np.zeros((nx, ep_len))
# y_max = np.zeros((nx, ep_len))
# for k in range(ep_len):
#     y_min[:, [k]] = get_y_min(d[:, [-ep_len + k]])
#     y_max[:, [k]] = get_y_max(d[:, [-ep_len + k]])
# # _, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
# axs[0].set_title("Last ep")
# for i in range(4):
#     axs[i].plot(y[-ep_len - 1 : -1, i])
#     axs[i].plot(y_min[i, :], color="black")
#     if i != 0:
#         axs[i].plot(y_max[i, :], color="r")
# # _, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
# # for i in range(3):
# #    axs[i].plot(U[:, i])

# # parameters - first cost params
# cost_keys = [x for x in list(param_list.keys()) if x.startswith("p") is False]
# if len(cost_keys) > 0:
#     _, axs = plt.subplots(len(cost_keys), 1, constrained_layout=True, sharex=True)
#     if len(cost_keys) == 1:
#         axs = [axs, None]
#     for i in range(len(cost_keys)):
#         axs[i].plot(
#             [
#                 (param_list[cost_keys[i]][j]).squeeze()
#                 for j in range(len(param_list[cost_keys[i]]))
#             ]
#         )
#         axs[i].set_ylabel(cost_keys[i])

#     param_keys = [x for x in list(param_list.keys()) if x.startswith("p") is True]
#     num_figs = int(np.ceil(len(param_keys) / 5))
#     for j in range(num_figs):
#         _, axs = plt.subplots(5, 1, constrained_layout=True, sharex=True)
#         for i in range(5):
#             if 5 * j + i < len(param_keys):
#                 axs[i].plot(param_list[param_keys[5 * j + i]])
#                 axs[i].set_ylabel(param_keys[5 * j + i])

plt.show()
