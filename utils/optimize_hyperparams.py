import argparse
import os
import sys
import uuid
from copy import deepcopy

import numpy as np
import optuna

sys.path.append(os.getcwd())
from q_learning_greenhouse import load_test, run_q_learning
from sims.configs.default import DefaultTest


def compute_score(data: tuple[dict[str, str | np.ndarray], ...]) -> float:
    """Computes the score to be optimized, given the results of a simulation."""
    data_train = data[0]
    td_errors = data_train["TD"]
    td_errors_per_ep = np.sum(td_errors, axis=1)
    return np.mean(td_errors_per_ep) - np.mean(td_errors_per_ep[-5:])


def launch_tuning(
    n_trials: int, n_episodes_per_trial: int, base_test_name: str, seed: int
) -> None:
    """Launches the hyperparameter tuning process."""
    study_name = f"greenhouse-{uuid.uuid4()}"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),  # for reproducibility
        pruner=None,
        load_if_exists=False,
        storage=f"sqlite:///{study_name}.db",
    )

    # save simulation details
    study.set_user_attr("n_trials", n_trials)
    study.set_user_attr("n_episodes_per_trial", n_episodes_per_trial)
    study.set_user_attr("base_test", base_test_name)
    study.set_user_attr("seed", seed)

    # load the base test only once
    base_test = load_test(base_test_name)
    base_test.num_episodes = n_episodes_per_trial

    def objective(trial: optuna.Trial) -> float:
        """Defines the function to be optimized, i.e., the RL returns."""
        test: DefaultTest = deepcopy(base_test)
        test.discount_factor = 1 - 10 ** trial.suggest_float("gamma_exp10", -10.0, -1.0)
        test.learning_rate = trial.suggest_float("learning_rate", 0.05, 1.0, log=True)
        return compute_score(run_q_learning(test))

    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimizes the hyperparameters for the greenhouse control.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-trials",
        "--n_trials",
        type=int,
        default=20,
        help="Numer of tuning trials.",
    )
    parser.add_argument(
        "--n-episodes-per-trial",
        "--n_episodes_per_trial",
        type=int,
        default=100,
        help="Numer of episodes to simulate per tuning trial.",
    )
    parser.add_argument(
        "--base-test",
        "--base_test",
        type=str,
        help="Base configuration for non-optimized hyperparameters.",
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()
    launch_tuning(
        args.n_trials,
        args.n_episodes_per_trial,
        args.base_test,
        args.seed,
    )
