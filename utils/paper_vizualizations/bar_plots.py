import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from greenhouse.model import Model
from utils.get_constraint_violations import get_constraint_violations
from utils.tikz import save2tikz

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

days = 40
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps
seconds_in_time_step = 15 * 60
nx = 4
nu = 3
nominal_files = [
    "results/nominal/nominal_greenhouse_rk4_True_0.pkl",
    "results/nominal/nominal_greenhouse_rk4_False_0.pkl",
]
sample_files = [
    "results/sample/sample_greenhouse_rk4_5_1_0.pkl",
    "results/sample/sample_greenhouse_rk4_10_1_0.pkl",
    "results/sample/sample_greenhouse_rk4_20_1_0.pkl",
]
mpc_rl_files = [
    "results/test_80_eval_final.pkl",
    "results/test_93_eval_final.pkl",
    "results/test_80_eval_final.pkl",
]
ddpg_files = [
    "results/ddpg_agent_0_eval_learned.pkl",
    "results/ddpg_agent_1_eval_learned.pkl",
    "results/ddpg_agent_2_eval_learned.pkl",
]

p = Model.get_true_parameters()

data = []
for file_name in nominal_files:
    with open(
        file_name,
        "rb",
    ) as file:
        data.append(pickle.load(file))
for file_name in sample_files:
    with open(
        file_name,
        "rb",
    ) as file:
        data.append(pickle.load(file))
mpc_rl_data = []
for file_name in mpc_rl_files:
    with open(
        file_name,
        "rb",
    ) as file:
        # stack the data from the different seeds
        mpc_rl_data.append(pickle.load(file))
data.append(
    {key: np.concatenate([o[key] for o in mpc_rl_data]) for key in ["X", "U", "R", "d"]}
)

# ddpg
ddpg_data = []
for file_name in ddpg_files:
    with open(file_name, "rb") as file:
        data_ = pickle.load(file)
        ddpg_ = {key: val for key, val in data_.items() if isinstance(val, np.ndarray)}
        ddpg_data.append(ddpg_)
data.append(
    {key: np.concatenate([o[key] for o in ddpg_data]) for key in ["X", "U", "R", "d"]}
)

labels = ["I-MPC", "N-MPC", "R-MPC-5", "R-MPC-10", "R-MPC-20", "our", "DDPG"]
colors = ["C0"] + ["C1"] + ["C2"] * 3 + ["C3"] + ["C4"]

_, ep_axs = plt.subplots(
    4, 1, constrained_layout=True, sharex=True
)  # axes for plotting each episode
R_indx = 0
VIOL_indx = 1
YIELD_indx = 2
EPI_indx = 3
# plot environment rewards
R = [np.sum(o["R"], axis=1) for o in data]
mean_R, std_R = zip(*((np.mean(r), np.std(r)) for r in R))
ep_axs[R_indx].bar(labels, mean_R, color=colors)
ep_axs[R_indx].errorbar(
    labels, mean_R, std_R, marker=None, ls="none", color="black", capsize=7
)
ep_axs[R_indx].set_yscale("log")
ep_axs[R_indx].set_ylabel(r"$L_{ep}$")

# calculate constraint violations
X_full = [o["X"] for o in data]
d_full = [o["d"] for o in data]
U_full = [o["U"] for o in data]
viols_full = []
y_full = []
for X, d, U in zip(X_full, d_full, U_full):
    # generate output data from state data X
    viols, y, _, _ = get_constraint_violations(X, U, d)
    viols_full.append(viols)
    y_full.append(y)

# plot constraint violations
tot_viols_full = [np.sum(viols, axis=1) for viols in viols_full]
mean_viols, std_viols = zip(
    *((np.mean(viols), np.std(viols)) for viols in tot_viols_full)
)
ep_axs[VIOL_indx].bar(labels, mean_viols, color=colors)
ep_axs[VIOL_indx].errorbar(
    labels, mean_viols, std_viols, marker=None, ls="none", color="black", capsize=7
)
ep_axs[VIOL_indx].set_yscale("log")
ep_axs[VIOL_indx].set_ylim(-10, ep_axs[VIOL_indx].get_ylim()[1])
ep_axs[VIOL_indx].set_ylabel(r"$\Psi$")

# plot yields
y_final = [y[:, -1, 0] for y in y_full]
mean_y, std_y = zip(*((np.mean(y), np.std(y)) for y in y_final))
ep_axs[YIELD_indx].bar(labels, mean_y, color=colors)
ep_axs[YIELD_indx].errorbar(
    labels, mean_y, std_y, marker=None, ls="none", color="black", capsize=7
)
# ep_axs[YIELD_indx].set_yscale("log")
ep_axs[YIELD_indx].set_ylabel(r"$y_1(N_s)$")
ep_axs[YIELD_indx].set_ylim([100, 200])

# calculate ecomonmic performance index
U = [o["U"] for o in data]
c_co2 = 42e-2
c_q = 6.35e-9
c_pri_1 = 1.8
c_pri_2 = 16
final_yield = [o[:, -1, 0] * 1e-3 for o in y_full]  # convert from g to kg
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
mean_EPI, std_EPI = zip(*((np.mean(epi), np.std(epi)) for epi in EPI))
ep_axs[EPI_indx].bar(labels, mean_EPI, color=colors)
ep_axs[EPI_indx].errorbar(
    labels, mean_EPI, std_EPI, marker=None, ls="none", color="black", capsize=7
)
ep_axs[EPI_indx].set_xticks(list(range(len(labels))), labels)
ep_axs[EPI_indx].set_ylabel(r"$P$")

# save2tikz(plt.gcf())


plt.show()
