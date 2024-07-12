# model of lettuce greenhouse from van Henten thesis (1994)


import casadi as cs
import numpy as np
from mpcrl.util.control import rk4
from mpcrl.util.seeding import RngType


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
    p_true.flags.writeable = False

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

    @staticmethod
    def get_learnable_parameter_bounds() -> dict:
        """Gets the lower and upper bounds for the learnable parameters.

        Returns
        -------
        dict
            The bounds for the learnable parameters.
        """
        p_bounds = {}
        for i in range(Model.n_params):
            p_bounds[f"p_{i}"] = [
                0.5,
                1.5,
            ]  # all parameters are normalized, such that true value is when p_i = 1. These bounds hence represent +- 50%
        return p_bounds

    @staticmethod
    def get_output_min(d: np.ndarray) -> np.ndarray:
        """Gets the minimum output values for the given disturbance.

        Parameters
        ----------
        d : np.ndarray
            The disturbance vector.

        Returns
        -------
        np.ndarray
            The minimum output values.
        """
        if d.shape[0] != 4:
            raise ValueError("Disturbance vector must have 4 elements.")
        y_min = np.zeros_like(d)
        y_min[2] = np.where(d[0] < 10, 10, 15)
        return y_min

    @staticmethod
    def get_output_max(d: np.ndarray) -> np.ndarray:
        """Gets the maximum output values for the given disturbance.

        Parameters
        ----------
        d : np.ndarray
            The disturbance vector.

        Returns
        -------
        np.ndarray
            The maximum output values."""
        if d.shape[0] != 4:
            raise ValueError("Disturbance vector must have 4 elements.")
        y_max = np.empty_like(d)
        y_max[0] = 1e6  # 1e6 replaces infinity
        y_max[1] = 1.6
        y_max[2] = np.where(d[0] < 10, 15, 20)
        y_max[3] = 70
        return y_max

    @staticmethod
    def get_output_range() -> np.ndarray:
        """Gets the range of output values.

        Returns
        -------
        np.ndarray
            The range of output values.
        """
        return np.array([1e6, 1.6, 5, 70])

    @staticmethod
    def get_u_min() -> np.ndarray:
        """Gets the minimum input values.

        Returns
        -------
        np.ndarray
            The minimum input values.
        """
        return np.zeros((3,))

    @staticmethod
    def get_u_max() -> np.ndarray:
        """Gets the maximum input values.

        Returns
        -------
        np.ndarray
            The maximum input values.
        """
        return np.array([1.2, 7.5, 150])

    @staticmethod
    def get_du_lim() -> np.ndarray:
        """Gets the input rate magnitude limit.

        Returns
        -------
        np.ndarray
            The input rate limits.
        """
        return 0.1 * Model.get_u_max()

    # sub-functions within dynamics
    @staticmethod
    def psi(
        x: np.ndarray | cs.SX, d: np.ndarray | cs.SX, p: np.ndarray | cs.SX
    ) -> np.ndarray | cs.SX:
        """Calculates the value of psi(x, d, p).

        Parameters
        ----------
        x : np.ndarray | cs.SX
            The state vector.
        d : np.ndarray | cs.SX
            The disturbance vector.
        p : np.ndarray | cs.SX
            The parameter vector.

        Returns
        -------
        np.ndarray | cs.SX
            The value of psi(x, d, p).
        """
        return (p[3] * M.p_scale[3]) * d[0] + (
            -(p[4] * M.p_scale[4]) * x[2] ** 2
            + (p[5] * M.p_scale[5]) * x[2]
            - (p[6] * M.p_scale[6])
        ) * (x[1] - (p[7] * M.p_scale[7]))

    @staticmethod
    def phi_phot_c(
        x: np.ndarray | cs.SX, d: np.ndarray | cs.SX, p: np.ndarray | cs.SX
    ) -> np.ndarray | cs.SX:
        """Calculates the value of phi_phot_c(x, d, p).

        Parameters
        ----------
        x : np.ndarray | cs.SX
            The state vector.
        d : np.ndarray | cs.SX
            The disturbance vector.
        p : np.ndarray | cs.SX
            The parameter vector.

        Returns
        -------
        np.ndarray | cs.SX
            The value of phi_phot_c(x, d, p).
        """
        return (
            (1 - cs.exp(-(p[2] * M.p_scale[2]) * x[0]))
            * (
                (p[3] * M.p_scale[3])
                * d[0]
                * (
                    -(p[4] * M.p_scale[4]) * x[2] ** 2
                    + (p[5] * M.p_scale[5]) * x[2]
                    - (p[6] * M.p_scale[6])
                )
                * (x[1] - (p[7] * M.p_scale[7]))
            )
        ) / (M.psi(x, d, p))

    @staticmethod
    def phi_vent_c(
        x: np.ndarray | cs.SX,
        u: np.ndarray | cs.SX,
        d: np.ndarray | cs.SX,
        p: np.ndarray | cs.SX,
    ) -> np.ndarray | cs.SX:
        """Calculates the value of phi_vent_c(x, u, d, p).

        Parameters
        ----------
        x : np.ndarray | cs.SX
            The state vector.
        u : np.ndarray | cs.SX
            The input vector.
        d : np.ndarray | cs.SX
            The disturbance vector.
        p : np.ndarray | cs.SX
            The parameter vector.

        Returns
        -------
        np.ndarray | cs.SX
            The value of phi_vent_c(x, u, d, p).
        """
        return (u[1] * 1e-3 + (p[10] * M.p_scale[10])) * (x[1] - d[1])

    @staticmethod
    def phi_vent_h(
        x: np.ndarray | cs.SX,
        u: np.ndarray | cs.SX,
        d: np.ndarray | cs.SX,
        p: np.ndarray | cs.SX,
    ) -> np.ndarray | cs.SX:
        """Calculates the value of phi_vent_h(x, u, d, p).

        Parameters
        ----------
        x : np.ndarray | cs.SX
            The state vector.
        u : np.ndarray | cs.SX
            The input vector.
        d : np.ndarray | cs.SX
            The disturbance vector.
        p : np.ndarray | cs.SX
            The parameter vector.

        Returns
        -------
        np.ndarray | cs.SX
            The value of phi_vent_h(x, u, d, p).
        """
        return (u[1] * 1e-3 + (p[10] * M.p_scale[10])) * (x[3] - d[3])

    @staticmethod
    def phi_trasnp_h(
        x: np.ndarray | cs.SX, p: np.ndarray | cs.SX
    ) -> np.ndarray | cs.SX:
        """Calculates the value of phi_trasnp_h(x, p).

        Parameters
        ----------
        x : np.ndarray | cs.SX
            The state vector.
        p : np.ndarray | cs.SX
            The parameter vector.

        Returns
        -------
        np.ndarray | cs.SX
            The value of phi_trasnp_h(x, p).
        """
        return (
            (p[20] * M.p_scale[20])
            * (1 - cs.exp(-(p[2] * M.p_scale[2]) * x[0]))
            * (
                (
                    (p[21] * M.p_scale[21])
                    / ((p[22] * M.p_scale[22]) * (x[2] + (p[23] * M.p_scale[23])))
                )
                * (
                    cs.exp(
                        ((p[24] * M.p_scale[24]) * x[2])
                        / (x[2] + (p[25] * M.p_scale[25]))
                    )
                )
                - x[3]
            )
        )

    @staticmethod
    def df(
        x: np.ndarray | cs.SX,
        u: np.ndarray | cs.SX,
        d: np.ndarray | cs.SX,
        p: np.ndarray | cs.SX,
    ) -> np.ndarray | cs.SX:
        """Get continuous differential equation for state dynamics.

        Parameters
        ----------
        x : np.ndarray | cs.SX
            The state vector.
        u : np.ndarray | cs.SX
            The input vector.
        d : np.ndarray | cs.SX
            The disturbance vector.
        p : np.ndarray | cs.SX
            The parameter vector.

        Returns
        -------
        np.ndarray | cs.SX
            The continuous differential equation for the state dynamics.
        """
        dx1 = (p[0] * M.p_scale[0]) * M.phi_phot_c(x, d, p) - (p[1] * M.p_scale[1]) * x[
            0
        ] * 2 ** (x[2] / 10 - 5 / 2)
        dx2 = (p[8] / (M.p_scale[8])) * (
            -M.phi_phot_c(x, d, p)
            + (p[9] * M.p_scale[9]) * x[0] * 2 ** (x[2] / 10 - 5 / 2)
            + u[0] * 1e-6
            - M.phi_vent_c(x, u, d, p)
        )
        dx3 = (p[15] / (M.p_scale[15])) * (
            u[2]
            - ((p[16] * M.p_scale[16]) * u[1] * 1e-3 + (p[17] * M.p_scale[17]))
            * (x[2] - d[2])
            + (p[18] * M.p_scale[18]) * d[0]
        )
        dx4 = (p[19] / (M.p_scale[19])) * (
            M.phi_trasnp_h(x, p) - M.phi_vent_h(x, u, d, p)
        )
        if isinstance(x, (cs.SX, cs.MX, cs.DM)):
            return cs.vertcat(dx1, dx2, dx3, dx4)
        return np.array([dx1, dx2, dx3, dx4])

    @staticmethod
    def output(x: np.ndarray | cs.SX, p: np.ndarray | cs.SX) -> np.ndarray | cs.SX:
        """Output function of state y = output(x).

        Parameters
        ----------
        x : np.ndarray | cs.SX
            The state vector.
        p : np.ndarray | cs.SX
            The parameter vector.

        Returns
        -------
        np.ndarray | cs.SX
            The output vector."""
        y1 = 1e3 * x[0]
        y2 = (
            (
                1e3
                * (p[13] * p[14])
                * (p[11] * M.p_scale[11])
                * (x[2] + (p[12] * M.p_scale[12]))
            )
            / ((M.p_scale[13]) * (M.p_scale[14]))
        ) * x[1]
        y3 = x[2]
        y4 = (
            (1e2 * (p[11] * M.p_scale[11]) * (x[2] + (p[12] * M.p_scale[12])))
            / (
                11
                * cs.exp(
                    ((p[26] * M.p_scale[26]) * x[2]) / (x[2] + (p[27] * M.p_scale[27]))
                )
            )
        ) * x[3]
        if isinstance(x, (cs.SX, cs.MX, cs.DM)):
            return cs.vertcat(y1, y2, y3, y4)
        return np.array([y1, y2, y3, y4])

    @staticmethod
    def euler_step(
        x: np.ndarray | cs.SX,
        u: np.ndarray | cs.SX,
        d: np.ndarray | cs.SX,
        p: np.ndarray | cs.SX,
        ts: float,
    ) -> np.ndarray | cs.SX:
        """Get discrete euler approximation for state update.

        Parameters
        ----------
        x : np.ndarray | cs.SX
            The state vector.
        u : np.ndarray | cs.SX
            The input vector.
        d : np.ndarray | cs.SX
            The disturbance vector.
        p : np.ndarray | cs.SX
            The parameter vector.
        ts : float | cs.SX
            The time step.

        Returns
        -------
        np.ndarray | cs.SX
            The state vector after one time step."""
        return x + ts * M.df(x, u, d, p)

    @staticmethod
    def rk4_step(
        x: np.ndarray | cs.SX,
        u: np.ndarray | cs.SX,
        d: np.ndarray | cs.SX,
        p: np.ndarray | cs.SX,
        ts: float,
        steps_per_ts: int = 1,
    ) -> np.ndarray | cs.SX:
        """Get discrete runge-kutter-4 approximation for state update.

        Parameters
        ----------
        x : np.ndarray | cs.SX
            The state vector.
        u : np.ndarray | cs.SX
            The input vector.
        d : np.ndarray | cs.SX
            The disturbance vector.
        p : np.ndarray | cs.SX
            The parameter vector.
        ts : float
            The time step.
        steps_per_ts : int, optional
            How many RK4 steps to take in one dt interval, by default 1

        Returns
        -------
        np.ndarray | cs.SX
            The state vector after one time step."""
        return rk4(lambda x_: M.df(x_, u, d, p), x, ts, steps_per_ts)


M = Model

# # model parameters
# nx = 4
# nu = 3
# nd = 4
# ts = 60 * 15  # 15 minute time steps
# time_steps_per_day = 24 * 4  # how many 15 minute incrementes there are in a day


# # noise terms for output measurement
# mean = 0
# sd = 0

# # disturbance profile
# d = np.load("data/disturbances.npy")
# VIABLE_STARTING_IDX = [0, 1, 3, 4, 5]
# shuffle(VIABLE_STARTING_IDX)
# ratio = floor(0.8 * len(VIABLE_STARTING_IDX))
# TRAIN_VIABLE_STARTING_IDX = VIABLE_STARTING_IDX[:ratio]
# TEST_VIABLE_STARTING_IDX = VIABLE_STARTING_IDX[ratio:]


# def get_model_details():
#     return nx, nu, nd, ts, time_steps_per_day


# def generate_perturbed_p(percentage_perturb: float = 0.1):
#     # cv = 0.05*np.eye(len(p_true))
#     # chol = np.linalg.cholesky(cv)
#     # rand_nums = np.random.randn(len(p_true), 1)
#     # p_hat = chol@rand_nums + np.asarray(p_true).reshape(rand_nums.shape)
#     # p_hat[p_hat < 0] = 0    # replace negative vals with zero
#     # return p_hat[:, 0]

#     # adding a perturbation of max 10% of the nominal value
#     p_hat = p_true.copy()
#     for i in range(len(p_hat)):
#         max_pert = p_hat[i] * percentage_perturb
#         p_hat[i] = p_hat[i] + np.random.uniform(-max_pert, max_pert)
#     return p_hat


# # generate a range of samples of perturbed parameters
# p_hat_list = []


# def generate_parameters(percentage_perturb: float = 0.1):
#     for i in range(20):  # generate 100 randomly purturbed param options
#         p_hat_list.append(generate_perturbed_p(percentage_perturb))


# # accurate dynamics
# def df_true(x, u, d):
#     """Get continuous differential equation for state with accurate parameters"""
#     return df(x, u, d, p_true)


# def euler_true(x, u, d):
#     """Get euler equation for state update with accurate parameters"""
#     return euler_step(x, u, d, p_true)


# def rk4_true(x, u, d):
#     """Get discrete RK4 difference equation for state with accurate parameters"""
#     return rk4_step(x, u, d, p_true)


# def output_true(x):
#     return output(x, p_true)


# # innacurate dynamics
# def df_perturbed(x, u, d, perturb_list: list[int]):
#     """Get continuous differential equation with a subset of parameters perturbed."""
#     p = p_true.copy()
#     for idx in perturb_list:
#         p[idx] = p_hat_list[0][idx]
#     return df(x, u, d, p)


# def euler_perturbed(x, u, d, perturb_list: list[int]):
#     """Get euler equation for state update with a subset of parameters perturbed"""
#     p = p_true.copy()
#     for idx in perturb_list:
#         p[idx] = p_hat_list[0][idx]
#     return euler_step(x, u, d, p)


# def rk4_perturbed(x, u, d, perturb_list: list[int]):
#     """Get discrete RK4 difference equation with a subset of parameters perturbed"""
#     p = p_true.copy()
#     for idx in perturb_list:
#         p[idx] = p_hat_list[0][idx]
#     return rk4_step(x, u, d, p)


# def output_perturbed(x, perturb_list: list[int]):
#     """Get output equation with a subset of parameters perturbed"""
#     p = p_true.copy()
#     for idx in perturb_list:
#         p[idx] = p_hat_list[0][idx]
#     return output(x, p)


# # robust sample based dynamics and output - assumed all parameters are wrong
# def multi_sample_step(x, u, d, n_samples: int, step_type: Literal["euler", "rk4"]):
#     if len(p_hat_list) == 0:
#         raise RuntimeError(
#             "P samples must be generated before using multi_sample_output."
#         )

#     if step_type == "euler":
#         step = euler_step
#     elif step_type == "rk4":
#         step = rk4_step
#     else:
#         raise RuntimeError(f"{step_type} is not a valid step_type.")

#     x_plus = cs.SX.zeros(x.shape)
#     for i in range(n_samples):
#         x_i = x[nx * i : nx * (i + 1), :]  # pull out state for one sample
#         x_i_plus = step(
#             x_i, u, d, p_hat_list[i]
#         )  # step it with the corresponding p values
#         x_plus[nx * i : nx * (i + 1), :] = x_i_plus
#     return x_plus


# def multi_sample_output(x: cs.SX, n_samples: int) -> cs.SX:
#     if len(p_hat_list) == 0:
#         raise RuntimeError(
#             "P samples must be generated before using multi_sample_output."
#         )
#     y = cs.SX.zeros(x.shape)
#     for i in range(n_samples):
#         x_i = x[nx * i : nx * (i + 1), :]
#         y_i = output(x_i, p_hat_list[i])
#         y[nx * i : nx * (i + 1), :] = y_i
#     return y


# # learning based dynamics
# def learnable_func(
#     x,
#     u,
#     d,
#     perturb_list: list[int],
#     p_learn_tuple: list[tuple[int, cs.SX]],
#     func: Literal["euler", "rk4"],
# ):
#     if len(p_hat_list) == 0:
#         raise RuntimeError("P samples must be generated before use.")
#     p = p_true.copy()
#     for idx in perturb_list:
#         p[idx] = p_hat_list[0][idx]
#     for idx, param in p_learn_tuple:
#         p[idx] = param
#     if func == "euler":
#         return euler_step(x, u, d, p)
#     elif func == "rk4":
#         return rk4_step(x, u, d, p)


# def euler_learnable(x, u, d, perturb_list, p_learn_tuple: list[tuple[int, cs.SX]]):
#     """Euler dynamics update with some parameters perturbed, and some learnable."""
#     return learnable_func(x, u, d, perturb_list, p_learn_tuple, "euler")


# def rk4_learnable(x, u, d, perturb_list, p_learn_tuple: list[tuple[int, cs.SX]]):
#     """Rk4 dynamics update with some parameters perturbed, and some learnable."""
#     return learnable_func(x, u, d, perturb_list, p_learn_tuple, "rk4")


# def output_learnable(x, perturb_list, p_learn_tuple: list[tuple[int, cs.SX]]):
#     """Output function with some parameters perturbed, and some learnable."""
#     if len(p_hat_list) == 0:
#         raise RuntimeError("P samples must be generated before use.")
#     p = p_true.copy()
#     for idx in perturb_list:
#         p[idx] = p_hat_list[0][idx]
#     for idx, param in p_learn_tuple:
#         p[idx] = param
#     return output(x, p)


# def get_perturbed_p(perturb_list: list[int]):
#     p = p_true.copy()
#     for idx in perturb_list:
#         p[idx] = p_hat_list[0][idx]


# def get_initial_perturbed_p():
#     return p_hat_list[0]
