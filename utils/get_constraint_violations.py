import os
import sys

import numpy as np

sys.path.append(os.getcwd())
from greenhouse.model import Model


def get_constraint_violations(
    X: np.ndarray, U: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates constraint violations for a given dataset.

    Parameters
    ----------
    X : np.ndarray
        The state dataset.
    U : np.ndarray
        The action dataset.
    d : np.ndarray
        The disturbance dataset.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        The constraint violations, the outputs, the minimum outputs, and the maximum outputs.
    """
    p = Model.get_true_parameters()
    y = Model.output(X[:, :-1].transpose(2, 0, 1), p).transpose(1, 2, 0)
    y_min = Model.get_output_min(d[: X.shape[0]].transpose()).transpose()
    y_max = Model.get_output_max(d[: X.shape[0]].transpose()).transpose()
    viols_lb = np.maximum(0, (y_min - y) / (y_max - y_min))
    viols_ub = np.maximum(0, (y - y_max) / (y_max - y_min))
    viols = (viols_lb + viols_ub).sum(-1)
    return viols, y, y_min, y_max
