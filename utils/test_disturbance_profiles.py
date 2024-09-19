import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from tikz import save2tikz

sys.path.append(os.getcwd())

from greenhouse.env import LettuceGreenHouse

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

# perturbed profiles
env = LettuceGreenHouse(growing_days=40, model_type="continuous")
d = env.pick_perturbed_disturbance_profile(initial_day=0, num_days=41, noise_scaling=0)
_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
for i in range(4):
    axs[i].plot(d[i, -2 * 96 :], color="C0")

TESTS = 5
seeds = np.random.SeedSequence(4).generate_state(TESTS)
for s in seeds:
    env.reset(seed=int(s))
    d_p = env.pick_perturbed_disturbance_profile(
        initial_day=0, num_days=41, noise_scaling=np.array([0.02, 0.01, 0.02, 0.01])
    )
    for i in range(4):
        axs[i].plot(d_p[i, -2 * 96 :], color="C1", linewidth=0.25)
axs[0].set_ylabel("Radiation")
axs[1].set_ylabel("CO2")
axs[2].set_ylabel("Temperature")
axs[3].set_ylabel("Humidity")
axs[3].set_xlabel("Time")
axs[0].legend(["Nominal", "Perturbed"])
save2tikz(plt.gcf())

# nominal profiles
_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
for k, day in enumerate([0, 160]):
    env.reset(seed=1)
    d_n = env.pick_perturbed_disturbance_profile(
        initial_day=day, num_days=41, noise_scaling=0
    )
    for i in range(4):
        axs[i].plot(d_n[i], color=f"C{k}", linewidth=1)
axs[0].set_ylabel("Radiation")
axs[1].set_ylabel("CO2")
axs[2].set_ylabel("Temperature")
axs[3].set_ylabel("Humidity")
axs[3].set_xlabel("Time")
axs[0].legend(["Start Day 0", "Start Day 160"])
plt.show()
