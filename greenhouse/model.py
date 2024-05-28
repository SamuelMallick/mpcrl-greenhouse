# model of lettuce greenhouse from van Henten thesis (1994)

from math import floor
from random import seed, shuffle
from typing import Literal

import casadi as cs
import numpy as np
from mpcrl.util.seeding import RngType

np.random.seed(1)
seed(1)


class Model:
    """Van Henten's Model of the greenhouse system. Contains the true parameters and
    methods to simulate the dynamics."""

    p_scale = np.asarray(
        [
            0.544,
            2.65e-7,
            53,
            3.55e-9,
            5.11e-6,
            2.3e-4,
            6.29e-4,
            5.2e-5,
            4.1,
            4.87e-7,
            7.5e-6,
            8.31,
            273.15,
            101325,
            0.044,
            3e4,
            1290,
            6.1,
            0.2,
            4.1,
            0.0036,
            9348,
            8314,
            273.15,
            17.4,
            239,
            17.269,
            238.3,
        ]
    )
    p_scale.flags.writeable = False
    n_params = p_scale.size
    p_true = np.ones(n_params, dtype=float)

    @staticmethod
    def get_true_parameters() -> np.ndarray:
        """Gets the true parameters of the model.

        Returns
        -------
        np.ndarray
            The true model's parameters.
        """
        return Model.p_true

    @staticmethod
    def get_perturbed_parameters(
        perturb_idx: list[int],
        perturb_percentage: float = 0.2,
        np_random: RngType = None,
    ) -> np.ndarray:
        """Gets a perturbed version of the true parameters.

        Parameters
        ----------
        perturb_idx : list[int]
            A list of indices of the parameters to perturb.
        perturb_percentage : float, optional
            The maximum percentage to perturb the parameters by, by default 0.2.
        np_random : RngType, optional
            The numpy random generator to use, by default None.

        Returns
        -------
        np.ndarray
            The perturbed parameters.
        """
        np_random = np.random.default_rng(np_random)
        p_perturbed = Model.p_true.copy()
        max_pert = Model.p_true[perturb_idx] * perturb_percentage
        p_perturbed[perturb_idx] += np_random.uniform(-max_pert, max_pert)
        return p_perturbed


M = Model

# model parameters
nx = 4
nu = 3
nd = 4
ts = 60 * 15  # 15 minute time steps
time_steps_per_day = 24 * 4  # how many 15 minute incrementes there are in a day

# u bounds
u_min = np.zeros((3, 1))
u_max = np.array([[1.2], [7.5], [150]])
du_lim = 0.1 * u_max

# noise terms for output measurement
mean = 0
sd = 0

# disturbance profile
d = np.load("data/disturbances.npy")
VIABLE_STARTING_IDX = [0, 1, 3, 4, 5]  # TODO make these legit
shuffle(VIABLE_STARTING_IDX)  # TODO: find alternative to shuffle that uses np_random
ratio = floor(0.8 * len(VIABLE_STARTING_IDX))
TRAIN_VIABLE_STARTING_IDX = VIABLE_STARTING_IDX[:ratio]
TEST_VIABLE_STARTING_IDX = VIABLE_STARTING_IDX[ratio:]


def get_model_details():
    return nx, nu, nd, ts, time_steps_per_day


def get_disturbance_profile(init_day: int, days_to_grow: int):
    # disturbance data has 324 days
    if init_day > 324:
        init_day = init_day % 324
    # return the disturbance vectors for the number of days requested, with one extra for the prediction horizon during the last day
    idx1 = init_day * time_steps_per_day
    idx2 = (init_day + days_to_grow + 1) * time_steps_per_day
    if idx2 > d.shape[1]:
        return np.hstack((d[:, idx1:], d[:, : idx2 % d.shape[1]]))
    else:
        return d[:, idx1:idx2]


def get_control_bounds():
    return u_min, u_max, du_lim


def get_y_min(d):
    if d[0] < 10:
        return np.array([[0], [0], [10], [0]])
    else:
        return np.array([[0], [0], [15], [0]])


def get_y_max(d):
    if d[0] < 10:
        return np.array([[1e6], [1.6], [15], [70]])  # 1e6 replaces infinity
    else:
        return np.array([[1e6], [1.6], [20], [70]])


# lower and upper bounds for parameters to be learned
p_bounds: dict = {}
for i in range(M.get_true_parameters().size):
    p_bounds[f"p_{i}"] = [
        0.5,
        1.5,
    ]  # all parameters are normalized, such that true value is when p_i = 1. These bounds hence represent +- 50%


def get_p_learn_bounds():
    return p_bounds


def generate_perturbed_p(percentage_perturb: float = 0.1):
    # cv = 0.05*np.eye(len(p_true))
    # chol = np.linalg.cholesky(cv)
    # rand_nums = np.random.randn(len(p_true), 1)
    # p_hat = chol@rand_nums + np.asarray(p_true).reshape(rand_nums.shape)
    # p_hat[p_hat < 0] = 0    # replace negative vals with zero
    # return p_hat[:, 0]

    # adding a perturbation of max 10% of the nominal value
    p_hat = p_true.copy()
    for i in range(len(p_hat)):
        max_pert = p_hat[i] * percentage_perturb
        p_hat[i] = p_hat[i] + np.random.uniform(-max_pert, max_pert)
    return p_hat


# generate a range of samples of perturbed parameters
p_hat_list = []


def generate_parameters(percentage_perturb: float = 0.1):
    for i in range(20):  # generate 100 randomly purturbed param options
        p_hat_list.append(generate_perturbed_p(percentage_perturb))


# continuos time model


# sub-functions within dynamics
def psi(x, d, p):
    return (p[3] * M.p_scale[3]) * d[0] + (
        -(p[4] * M.p_scale[4]) * x[2] ** 2
        + (p[5] * M.p_scale[5]) * x[2]
        - (p[6] * M.p_scale[6])
    ) * (x[1] - (p[7] * p_scale[7]))


def phi_phot_c(x, d, p):
    return (
        (1 - cs.exp(-(p[2] * p_scale[2]) * x[0]))
        * (
            (p[3] * p_scale[3])
            * d[0]
            * (
                -(p[4] * p_scale[4]) * x[2] ** 2
                + (p[5] * p_scale[5]) * x[2]
                - (p[6] * p_scale[6])
            )
            * (x[1] - (p[7] * p_scale[7]))
        )
    ) / (psi(x, d, p))


def phi_vent_c(x, u, d, p):
    return (u[1] * 1e-3 + (p[10] * p_scale[10])) * (x[1] - d[1])


def phi_vent_h(x, u, d, p):
    return (u[1] * 1e-3 + (p[10] * p_scale[10])) * (x[3] - d[3])


def phi_trasnp_h(x, p):
    return (
        (p[20] * p_scale[20])
        * (1 - cs.exp(-(p[2] * p_scale[2]) * x[0]))
        * (
            (
                (p[21] * p_scale[21])
                / ((p[22] * p_scale[22]) * (x[2] + (p[23] * p_scale[23])))
            )
            * (cs.exp(((p[24] * p_scale[24]) * x[2]) / (x[2] + (p[25] * p_scale[25]))))
            - x[3]
        )
    )


def df(x, u, d, p):
    """Continuous derivative of state d_dot = df(x, u, d)/dt"""
    dx1 = (p[0] * p_scale[0]) * phi_phot_c(x, d, p) - (p[1] * p_scale[1]) * x[
        0
    ] * 2 ** (x[2] / 10 - 5 / 2)
    dx2 = (p[8] / (p_scale[8])) * (
        -phi_phot_c(x, d, p)
        + (p[9] * p_scale[9]) * x[0] * 2 ** (x[2] / 10 - 5 / 2)
        + u[0] * 1e-6
        - phi_vent_c(x, u, d, p)
    )
    dx3 = (p[15] / (p_scale[15])) * (
        u[2]
        - ((p[16] * p_scale[16]) * u[1] * 1e-3 + (p[17] * p_scale[17])) * (x[2] - d[2])
        + (p[18] * p_scale[18]) * d[0]
    )
    dx4 = (p[19] / (p_scale[19])) * (phi_trasnp_h(x, p) - phi_vent_h(x, u, d, p))
    return cs.vertcat(dx1, dx2, dx3, dx4)


def euler_step(x, u, d, p):
    return x + ts * df(x, u, d, p)


def rk4_step(x, u, d, p):
    k1 = df(x, u, d, p)
    k2 = df(x + (ts / 2) * k1, u, d, p)
    k3 = df(x + (ts / 2) * k2, u, d, p)
    k4 = df(x + ts * k3, u, d, p)
    return x + (ts / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def output(x, p):
    """Output function of state y = output(x)"""
    y1 = 1e3 * x[0]
    y2 = (
        (1e3 * (p[13] * p[14]) * (p[11] * p_scale[11]) * (x[2] + (p[12] * p_scale[12])))
        / ((p_scale[13]) * (p_scale[14]))
    ) * x[1]
    y3 = x[2]
    y4 = (
        (1e2 * (p[11] * p_scale[11]) * (x[2] + (p[12] * p_scale[12])))
        / (11 * cs.exp(((p[26] * p_scale[26]) * x[2]) / (x[2] + (p[27] * p_scale[27]))))
    ) * x[3]

    # add noise to measurement
    # noise = np.random.normal(mean, sd, (nx, 1))
    return cs.vertcat(y1, y2, y3, y4)


# accurate dynamics
def df_true(x, u, d):
    """Get continuous differential equation for state with accurate parameters"""
    return df(x, u, d, p_true)


def euler_true(x, u, d):
    """Get euler equation for state update with accurate parameters"""
    return euler_step(x, u, d, p_true)


def rk4_true(x, u, d):
    """Get discrete RK4 difference equation for state with accurate parameters"""
    return rk4_step(x, u, d, p_true)


def output_true(x):
    return output(x, p_true)


# innacurate dynamics
def df_perturbed(x, u, d, perturb_list: list[int]):
    """Get continuous differential equation with a subset of parameters perturbed."""
    p = p_true.copy()
    for idx in perturb_list:
        p[idx] = p_hat_list[0][idx]
    return df(x, u, d, p)


def euler_perturbed(x, u, d, perturb_list: list[int]):
    """Get euler equation for state update with a subset of parameters perturbed"""
    p = p_true.copy()
    for idx in perturb_list:
        p[idx] = p_hat_list[0][idx]
    return euler_step(x, u, d, p)


def rk4_perturbed(x, u, d, perturb_list: list[int]):
    """Get discrete RK4 difference equation with a subset of parameters perturbed"""
    p = p_true.copy()
    for idx in perturb_list:
        p[idx] = p_hat_list[0][idx]
    return rk4_step(x, u, d, p)


def output_perturbed(x, perturb_list: list[int]):
    """Get output equation with a subset of parameters perturbed"""
    p = p_true.copy()
    for idx in perturb_list:
        p[idx] = p_hat_list[0][idx]
    return output(x, p)


# robust sample based dynamics and output - assumed all parameters are wrong
def multi_sample_step(x, u, d, n_samples: int, step_type: Literal["euler", "rk4"]):
    if len(p_hat_list) == 0:
        raise RuntimeError(
            "P samples must be generated before using multi_sample_output."
        )

    if step_type == "euler":
        step = euler_step
    elif step_type == "rk4":
        step = rk4_step
    else:
        raise RuntimeError(f"{step_type} is not a valid step_type.")

    x_plus = cs.SX.zeros(x.shape)
    for i in range(n_samples):
        x_i = x[nx * i : nx * (i + 1), :]  # pull out state for one sample
        x_i_plus = step(
            x_i, u, d, p_hat_list[i]
        )  # step it with the corresponding p values
        x_plus[nx * i : nx * (i + 1), :] = x_i_plus
    return x_plus


def multi_sample_output(x: cs.SX, n_samples: int) -> cs.SX:
    if len(p_hat_list) == 0:
        raise RuntimeError(
            "P samples must be generated before using multi_sample_output."
        )
    y = cs.SX.zeros(x.shape)
    for i in range(n_samples):
        x_i = x[nx * i : nx * (i + 1), :]
        y_i = output(x_i, p_hat_list[i])
        y[nx * i : nx * (i + 1), :] = y_i
    return y


# learning based dynamics
def learnable_func(
    x,
    u,
    d,
    perturb_list: list[int],
    p_learn_tuple: list[tuple[int, cs.SX]],
    func: Literal["euler", "rk4"],
):
    if len(p_hat_list) == 0:
        raise RuntimeError("P samples must be generated before use.")
    p = p_true.copy()
    for idx in perturb_list:
        p[idx] = p_hat_list[0][idx]
    for idx, param in p_learn_tuple:
        p[idx] = param
    if func == "euler":
        return euler_step(x, u, d, p)
    elif func == "rk4":
        return rk4_step(x, u, d, p)


def euler_learnable(x, u, d, perturb_list, p_learn_tuple: list[tuple[int, cs.SX]]):
    """Euler dynamics update with some parameters perturbed, and some learnable."""
    return learnable_func(x, u, d, perturb_list, p_learn_tuple, "euler")


def rk4_learnable(x, u, d, perturb_list, p_learn_tuple: list[tuple[int, cs.SX]]):
    """Rk4 dynamics update with some parameters perturbed, and some learnable."""
    return learnable_func(x, u, d, perturb_list, p_learn_tuple, "rk4")


def output_learnable(x, perturb_list, p_learn_tuple: list[tuple[int, cs.SX]]):
    """Output function with some parameters perturbed, and some learnable."""
    if len(p_hat_list) == 0:
        raise RuntimeError("P samples must be generated before use.")
    p = p_true.copy()
    for idx in perturb_list:
        p[idx] = p_hat_list[0][idx]
    for idx, param in p_learn_tuple:
        p[idx] = param
    return output(x, p)


def get_perturbed_p(perturb_list: list[int]):
    p = p_true.copy()
    for idx in perturb_list:
        p[idx] = p_hat_list[0][idx]


def get_initial_perturbed_p():
    return p_hat_list[0]
