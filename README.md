# Reinforcement Learning-Based Model Predicitive Control for Greenhouse Climate Control

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/SamuelMallick/dmpcrl-concept/blob/main/LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-green.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


This repository contains the source code used to produce the results obtained in [Reinforcement Learning-Based Model Predicitive Control for Greenhouse Climate Control](https://arxiv.org/abs/2409.12789) submitted to [Computers and Electronics in Agriculture](https://www.sciencedirect.com/journal/computers-and-electronics-in-agriculture).

In this work we propose an integrated model predictive control and reinforcement learning approach for greenhouse climate control.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{mallick2025reinforcement,
  title={Reinforcement learning-based model predictive control for greenhouse climate control},
  author={Mallick, Samuel and Airaldi, Filippo and Dabiri, Azita and Sun, Congcong and De Schutter, Bart},
  journal={Smart Agricultural Technology},
  volume={10},
  pages={100751},
  year={2025},
  publisher={Elsevier}
}
```

---

## Installation

The code was created with `Python 3.11`. To access it, clone the repository

```bash
git clone https://github.com/SamuelMallick/mpcrl-greenhouse
cd mpcrl-greenhouse
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`agents`** contains the classes defined for RL agents.
- **`data`** contains weather disturbance data.
- **`greenhouse`** contains the model and environments classes for the greenhouse system.
- **`mpcs`** contains the classes for all mpc controllers.
- **`sims/configs`** contains configuration files for simulations.
- **`utils`** contains plotting and evalation scripts used to generate images and data used in Reinforcement Learning-Based Model Predicitive Control for Greenhouse Climate Control
- **`nominal_greenhouse.py`** simulates the nominal mpc controller.
- **`sample_greenhouse.py`** simulates the sample based mpc controller.
- **`q_learning_greenhouse.py`** trains the RL-based mpc controller.
- **`train_ddpg.py`** trains the DDPG-based RL controller.
- **`visualization.py`** vizualizes data saved from simulations.
## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/SamuelMallick/dmpcrl-concept/blob/main/LICENSE) file included with this repository.

---

## Author

[Samuel Mallick](https://www.tudelft.nl/staff/s.h.mallick/), PhD Candidate [s.mallick@tudelft.nl | sam.mallick.97@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of a project that has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([Grant agreement No. 101018826 - CLariNet](https://cordis.europa.eu/project/id/101018826)).

Copyright (c) 2024 Samuel Mallick.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest in the program “mpcrl-greenouse” (Reinforcement Learning-Based Model Predicitive Control for Greenhouse Climate Control) written by the Author(s). Prof. Dr. Ir. Fred van Keulen, Dean of 3mE.