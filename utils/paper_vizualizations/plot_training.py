import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.getcwd())
from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model
from utils.get_constraint_violations import get_constraint_violations
from utils.tikz import save2tikz

plt.style.use("bmh")

# load data
test_numbers = [80, 93, 94]
data = []
for test_number in test_numbers:
    with open(f"results/test_{test_number}_train.pkl", "rb") as file:
        data.append(pickle.load(file))

# define some constants
days = 40
ep_len = days * LettuceGreenHouse.steps_per_day
seconds_in_time_step = 15 * 60
nx = 4
nu = 3
nd = 4
p = Model.get_true_parameters()

R = np.asarray([o["R"] for o in data])
R_mean = R.mean(0)
R_std = R.std(0)
TD = np.asarray([o["TD"] for o in data])
TD_mean = TD.mean(0)
TD_std = TD.std(0)
# calculate constraint viols
viols = []
for o in data:
    X = o["X"]
    d = o["d"]
    U = o["U"]
    viols_, _, _, _ = get_constraint_violations(X, U, d)
    viols.append(viols_)
viols = np.asarray(viols)
viols_mean = viols.mean(0)
viols_std = viols.std(0)
# get parameters
param_dict_mean = {
    key: np.asarray([o["param_dict"][key] for o in data]).mean(0)
    for key in data[0]["param_dict"]
    if not key.startswith("o")
}
param_dict_std = {
    key: np.asarray([o["param_dict"][key] for o in data]).std(0)
    for key in data[0]["param_dict"]
    if not key.startswith("o")
}
most_changed_params_mean = sorted(
    param_dict_mean,
    key=lambda k: np.linalg.norm((param_dict_mean[k][-1] - param_dict_mean[k][0]) / 1),
    reverse=True,
)
most_changed_param_std = sorted(
    param_dict_std,
    key=lambda k: np.linalg.norm((param_dict_mean[k][-1] - param_dict_mean[k][0]) / 1),
    reverse=True,
)

# create axes for plotting each timestep's and each episode's quantities
e_fig, e_axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
R_indx, TD_indx, VIOL_Y_indx = range(3)

# plot environment rewards
e_axs[R_indx].plot(R_mean.sum(1))
e_axs[R_indx].fill_between(
    list(range(R_mean.shape[0])),
    R_mean.sum(1) - R_std.sum(1),
    R_mean.sum(1) + R_std.sum(1),
    alpha=0.5,
)
# e_axs[R_indx].set_yscale("log")
e_axs[R_indx].set_ylabel("$L_{ep}$")

e_axs[TD_indx].plot(np.nanmean(TD_mean, 1))
e_axs[TD_indx].fill_between(
    list(range(TD_mean.shape[0])),
    np.nanmean(TD_mean, 1) - np.nanmean(TD_std, 1),
    np.nanmean(TD_mean, 1) + np.nanmean(TD_std, 1),
    alpha=0.5,
)
# e_axs[TD_indx].set_yscale("log")
e_axs[TD_indx].set_ylabel(r"$\delta_{ep}$")

e_axs[VIOL_Y_indx].plot(viols_mean.sum(1))
e_axs[VIOL_Y_indx].fill_between(
    list(range(viols_mean.shape[0])),
    viols_mean.sum(1) - viols_std.sum(1),
    viols_mean.sum(1) + viols_std.sum(1),
    alpha=0.5,
)
e_axs[VIOL_Y_indx].set_ylabel(r"$\Psi$")

e_axs[VIOL_Y_indx].set_xlabel("Growth cycle")

save2tikz(plt.gcf())

# plot most changed parameters
num_params = 4
fig, axs = plt.subplots(num_params, 1, constrained_layout=True, sharex=True)
for i, key in enumerate(most_changed_params_mean[:num_params]):
    axs[i].plot(param_dict_mean[key])
    axs[i].fill_between(
        list(range(param_dict_mean[key].shape[0])),
        (param_dict_mean[key] - param_dict_std[key]).squeeze(1),
        (param_dict_mean[key] + param_dict_std[key]).squeeze(1),
        alpha=0.5,
    )
    axs[i].set_ylabel(key)
axs[-1].set_xlabel("Growth cycle")
# save2tikz(plt.gcf())

plt.show()
