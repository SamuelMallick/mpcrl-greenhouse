import pickle

import matplotlib.pyplot as plt

plt.rc("text", usetex=True)

num_episodes = 50
days = 40
ep_len = days * 24 * 4  # 40 days of 15 minute timesteps

with open(
    "data/green_data/e2.pkl",
    "rb",
) as file:
    X = pickle.load(file)
    U = pickle.load(file)
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

_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
for i in range(4):
    axs[i].plot(X[:, i])
_, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
for i in range(3):
    axs[i].plot(U[:, i])

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


plt.show()
