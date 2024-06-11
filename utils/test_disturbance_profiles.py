import sys
import os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from greenhouse.env import LettuceGreenHouse

env = LettuceGreenHouse(growing_days=40, model_type="continuous")
d = env.pick_perturbed_disturbance_profile(initial_day=0, num_days=41, noise_scaling=0)
_, axs = plt.subplots(4, 1, constrained_layout=True, sharex=True)
for i in range(4):
    axs[i].plot(d[i], color="black")
    
TESTS = 10
seeds = np.random.SeedSequence(4).generate_state(TESTS)
for s in seeds:
    env.reset(seed=int(s))
    d_p = env.pick_perturbed_disturbance_profile(initial_day=0, num_days=41, noise_scaling=np.array([0.02, 0.01, 0.025, 0.01]))
    for i in range(4):
        axs[i].plot(d_p[i], color="r", linewidth=0.1)
plt.show()

