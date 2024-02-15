# model of lettuce greenhouse from van Henten thesis (1994)
from typing import Literal

import casadi as cs
import numpy as np

np.random.seed(1)

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


def get_model_details():
    return nx, nu, nd, ts, time_steps_per_day


def get_disturbance_profile(init_day: int, days_to_grow: int):
    # an extra days worth added to the profile for the prediction horizon
    if init_day > 310:
        init_day = init_day % 310
    # return the disturbance vectors for the number of dyas requested, with one extra for the prediction horizon during the last day
    return d[
        :,
        init_day
        * time_steps_per_day : (init_day + days_to_grow + 1)
        * time_steps_per_day,
    ]


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


p_true = [
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


def generate_perturbed_p():
    # cv = 0.05*np.eye(len(p_true))
    # chol = np.linalg.cholesky(cv)
    # rand_nums = np.random.randn(len(p_true), 1)
    # p_hat = chol@rand_nums + np.asarray(p_true).reshape(rand_nums.shape)
    # p_hat[p_hat < 0] = 0    # replace negative vals with zero
    # return p_hat[:, 0]

    p_hat = p_true.copy()
    for i in range(len(p_hat)):
        max_pert = p_hat[i] * 0.1
        p_hat[i] = p_hat[i] + np.random.uniform(-max_pert, max_pert)
    return p_hat


# continuos time model


# sub-functions within dynamics
def psi(x, d, p):
    return p[3] * d[0] + (-p[4] * x[2] ** 2 + p[5] * x[2] - p[6]) * (x[1] - p[7])


def phi_phot_c(x, d, p):
    return (
        (1 - cs.exp(-p[2] * x[0]))
        * (p[3] * d[0] * (-p[4] * x[2] ** 2 + p[5] * x[2] - p[6]) * (x[1] - p[7]))
    ) / (psi(x, d, p))


def phi_vent_c(x, u, d, p):
    return (u[1] * 1e-3 + p[10]) * (x[1] - d[1])


def phi_vent_h(x, u, d, p):
    return (u[1] * 1e-3 + p[10]) * (x[3] - d[3])


def phi_trasnp_h(x, p):
    return (
        p[20]
        * (1 - cs.exp(-p[2] * x[0]))
        * (
            ((p[21]) / (p[22] * (x[2] + p[23])))
            * (cs.exp((p[24] * x[2]) / (x[2] + p[25])))
            - x[3]
        )
    )


def df(x, u, d, p):
    """Continuous derivative of state d_dot = df(x, u, d)/dt"""
    dx1 = p[0] * phi_phot_c(x, d, p) - p[1] * x[0] * 2 ** (x[2] / 10 - 5 / 2)
    dx2 = (1 / p[8]) * (
        -phi_phot_c(x, d, p)
        + p[9] * x[0] * 2 ** (x[2] / 10 - 5 / 2)
        + u[0] * 1e-6
        - phi_vent_c(x, u, d, p)
    )
    dx3 = (1 / p[15]) * (
        u[2] - (p[16] * u[1] * 1e-3 + p[17]) * (x[2] - d[2]) + p[18] * d[0]
    )
    dx4 = (1 / p[19]) * (phi_trasnp_h(x, p) - phi_vent_h(x, u, d, p))
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
    y2 = ((1e3 * p[11] * (x[2] + p[12])) / (p[13] * p[14])) * x[1]
    y3 = x[2]
    y4 = (
        (1e2 * p[11] * (x[2] + p[12])) / (11 * cs.exp((p[26] * x[2]) / (x[2] + p[27])))
    ) * x[3]

    # add noise to measurement
    noise = np.random.normal(mean, sd, (nx, 1))
    return cs.vertcat(y1, y2, y3, y4) + noise


def df_real(x, u, d):
    """Get continuous differential equation for state with accurate parameters"""
    return df(x, u, d, p_true)


def euler_real(x, u, d):
    """Get euler equation for state update with accurate parameters"""
    return x + ts * df_real(x, u, d)


def rk4_step_real(x, u, d):
    """Get discrete RK4 difference equation for state with accurate parameters"""
    return rk4_step(x, u, d, p_true)


def output_real(x):
    return output(x, p_true)


# robust sample based dynamics and output
p_hat_list = []
for i in range(20):  # generate 20 randomly purturbed param options
    p_hat_list.append(generate_perturbed_p())


def multi_sample_step(x, u, d, Ns, step_type: Literal["euler", "rk4"]):
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
    for i in range(Ns):
        x_i = x[nx * i : nx * (i + 1), :]  # pull out state for one sample
        x_i_plus = step(
            x_i, u, d, p_hat_list[i]
        )  # step it with the corresponding p values
        x_plus[nx * i : nx * (i + 1), :] = x_i_plus
    return x_plus


def multi_sample_euler_step(x, u, d, Ns):
    """Computes a euler dynamics update for Ns copies of the state x, with a different param sample for each"""
    return multi_sample_step(x, u, d, Ns, step_type="euler")


def multi_sample_rk4_step(x, u, d, Ns):
    """Computes an rk4 dynamics update for Ns copies of the state x, with a different param sample for each"""
    return multi_sample_step(x, u, d, Ns, step_type="rk4")


def multi_sample_output(x, Ns):
    if len(p_hat_list) == 0:
        raise RuntimeError(
            "P samples must be generated before using multi_sample_output."
        )
    y = cs.SX.zeros(x.shape)
    for i in range(Ns):
        x_i = x[nx * i : nx * (i + 1), :]
        y_i = output(x_i, p_hat_list[i])
        y[nx * i : nx * (i + 1), :] = y_i
    return y


# learning based dynamics
def rk4_learnable(x, u, d, p_learnable, p_indexes):
    if len(p_hat_list) == 0:
        raise RuntimeError("P samples must be generated before use.")
    param_counter = 0
    p_learnable_full = p_hat_list[
        0
    ].copy()  # use the first perturbed entry as the initial guess for the parameters
    for i in range(
        len(p_indexes)
    ):  # replace the ones that can be learned with symbolic params
        p_learnable_full[p_indexes[i]] = p_learnable[param_counter]
        param_counter += 1

    return rk4_step(x, u, d, p_learnable_full)


def euler_learnable(x, u, d, p_learnable, p_indexes):
    if len(p_hat_list) == 0:
        raise RuntimeError("P samples must be generated before use.")
    param_counter = 0
    p_learnable_full = p_hat_list[
        0
    ].copy()  # use the first perturbed entry as the initial guess for the parameters
    for i in range(
        len(p_indexes)
    ):  # replace the ones that can be learned with symbolic params
        p_learnable_full[p_indexes[i]] = p_learnable[param_counter]
        param_counter += 1

    return x + ts * df(x, u, d, p_learnable_full)


def output_learnable(x, p_learnable, p_indexes):
    if len(p_hat_list) == 0:
        raise RuntimeError("P samples must be generated before use.")
    param_counter = 0
    p_learnable_full = p_hat_list[0].copy()
    for i in range(len(p_indexes)):
        p_learnable_full[p_indexes[i]] = p_learnable[param_counter]
        param_counter += 1

    return output(x, p_learnable_full)


def get_initial_perturbed_p():
    return p_hat_list[0]
