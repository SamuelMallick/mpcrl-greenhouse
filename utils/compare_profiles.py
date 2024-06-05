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

num_episodes = 1
days = 40
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps
nx = 4
nu = 3

X = []
U = []
y = []
d = []
R = []
TD = []

num_tests = 324

for i in range(num_tests):
    with open(
        f"results/nominal/disturbance_profiles/nom_first_day_{i}.pkl",
        "rb",
    ) as file:
        X.append(pickle.load(file))
        U.append(pickle.load(file))
        y.append(pickle.load(file))
        d.append(pickle.load(file))
        R.append(pickle.load(file))
        TD.append(pickle.load(file))
        param_list = pickle.load(file)

R = [sum(R[i]) for i in range(num_tests)]
_, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
axs.plot(R, "o", markersize=1)
axs.set_ylabel(r"$\tau$")
axs.set_ylabel("$L$")

# yields
yields = [y[i][-1, 0] for i in range(num_tests)]
_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
axs[0].plot(yields, "o", markersize=1)
axs[0].set_ylabel(r"$yield$")

num_cnstr_viols = np.zeros((num_tests, 1))
mag_cnstr_viols = np.zeros((num_tests, 1))
for i in range(num_tests):
    print(f"test {i}")
    for k in range(ep_len):
        y_max = Model.get_output_max(d[i][:, [k]])
        y_min = Model.get_output_min(d[i][:, [k]])
        # extra +i index in y because it has ep_len+1 entries for each ep
        if any(y[i][k, :] > y_max) or any(y[i][k, :] < y_min):
            num_cnstr_viols[i, :] += 1
            y_below = y[i][[k], :].reshape(4, 1) - y_min
            y_below[y_below > 0] = 0
            y_above = y_max - y[i][[k], :].reshape(4, 1)
            y_above[y_above > 0] = 0
            mag_cnstr_viols[i, :] += np.linalg.norm(y_above + y_below, ord=2)
axs[1].plot(num_cnstr_viols, "o", markersize=1)
axs[1].set_ylabel(r"$num cnstr viols$")
axs[2].plot(mag_cnstr_viols, "o", markersize=1)
axs[2].set_ylabel(r"$mag cnstr viols$")

# disturbance profiles
_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
for i in range(4):
    axs[i].plot([sum(d[j][i, :]) for j in range(num_tests)])

_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
for i in range(4):
    axs[i].plot(np.concatenate([d[j][i, :] for j in range(num_tests)]))
plt.show()
