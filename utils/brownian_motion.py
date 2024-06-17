import numpy as np


def brownian_bridge(
    steps: int, noise_width: float, np_random: np.random.Generator
) -> np.ndarray:
    """
    Generates a Brownian bridge.

    Parameters:
        steps (int): Number of steps.
        noise_width (float): Width of noise.

    Returns:
        np.ndarray: Brownian bridge values.
    """
    t = np.arange(0, steps)
    W = np_random.normal(0, noise_width, size=steps)
    W = np.cumsum(W)
    B = W - (t / steps) * W[-1]
    return B


def brownian_excursion(
    steps: int, noise_width: float, np_random: np.random.Generator
) -> np.ndarray:
    """
    Generates a Brownian excursion.

    Parameters:
        steps (int): Number of steps.
        noise_width (float): Width of noise.

    Returns:
        np.ndarray: Brownian excursion values.
    """
    B = brownian_bridge(steps, noise_width, np_random)
    return B
