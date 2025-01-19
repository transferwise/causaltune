from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def bite(
    working_df: pd.DataFrame,
    treatment_name: str,
    outcome_name: str,
    min_N: int = 10,
    max_N: int = 1000,
    num_N: int = 20,
    N_values: Optional[List[int]] = None,
    clip_propensity: float = 0.05,
) -> float:
    max_N = int(min(max_N, len(working_df) / 10))
    if N_values is None:
        N_values = exponential_spacing(min_N, max_N, num_N)
    # Calculate weights with clipping to avoid extremes
    working_df["weights"] = np.where(
        working_df[treatment_name] == 1,
        1 / np.clip(working_df["propensity"], clip_propensity, 1 - clip_propensity),
        1 / np.clip(1 - working_df["propensity"], clip_propensity, 1 - clip_propensity),
    )

    kendall_tau_values = []

    for N in N_values:
        iter_df = working_df.copy()

        try:
            # Ensure enough unique values for binning
            unique_ites = np.unique(iter_df["estimated_ITE"])
            if len(unique_ites) < N:
                continue

            # Create bins
            iter_df["ITE_bin"] = pd.qcut(
                iter_df["estimated_ITE"], q=N, labels=False, duplicates="drop"
            )

            # Compute bin statistics
            bin_stats = []
            for bin_idx in iter_df["ITE_bin"].unique():
                bin_data = iter_df[iter_df["ITE_bin"] == bin_idx]

                # Skip if bin is too small
                if len(bin_data) < 2:
                    continue

                naive_est = compute_naive_estimate(bin_data, treatment_name, outcome_name)

                # Only compute average ITE if weights are valid
                bin_weights = bin_data["weights"].values
                if bin_weights.sum() > 0 and not np.isnan(naive_est):
                    try:
                        avg_est_ite = np.average(bin_data["estimated_ITE"], weights=bin_weights)
                        bin_stats.append(
                            {
                                "ITE_bin": bin_idx,
                                "naive_estimate": naive_est,
                                "average_estimated_ITE": avg_est_ite,
                            }
                        )
                    except ZeroDivisionError:
                        continue

            # Calculate Kendall's Tau if we have enough valid bins
            bin_stats_df = pd.DataFrame(bin_stats)
            if len(bin_stats_df) >= 2:
                tau, _ = kendalltau(
                    bin_stats_df["naive_estimate"],
                    bin_stats_df["average_estimated_ITE"],
                )
                if not np.isnan(tau):
                    kendall_tau_values.append(tau)

        except (ValueError, ZeroDivisionError):
            continue

    # Return final score
    if len(kendall_tau_values) == 0:
        return -np.inf  # Return -inf for failed computations

    # top_3_taus = sorted(kendall_tau_values, reverse=True)[:3]
    return np.mean(kendall_tau_values)


def compute_naive_estimate(
    group_data: pd.DataFrame, treatment_name: str, outcome_name: str
) -> float:
    """Compute naive estimate for a group with safeguards against edge cases."""
    treated = group_data[group_data[treatment_name] == 1]
    control = group_data[group_data[treatment_name] == 0]

    if len(treated) == 0 or len(control) == 0:
        return np.nan

    treated_weights = treated["weights"].values
    control_weights = control["weights"].values

    # Check if weights sum to 0 or if all weights are 0
    if (
        treated_weights.sum() == 0
        or control_weights.sum() == 0
        or not (treated_weights > 0).any()
        or not (control_weights > 0).any()
    ):
        return np.nan

    # Weighted averages with explicit handling of edge cases
    try:
        y1 = np.average(treated[outcome_name], weights=treated_weights)
        y0 = np.average(control[outcome_name], weights=control_weights)
        return y1 - y0
    except ZeroDivisionError:
        return np.nan


def exponential_spacing(start, end, num_points):
    """
    Generate approximately exponentially spaced integers between start and end.

    Parameters:
        start (int): The starting value.
        end (int): The ending value.
        num_points (int): Number of integers to generate.

    Returns:
        list: A list of approximately exponentially spaced integers.
    """
    # Use a logarithmic scale for exponential spacing
    log_start = np.log(start)
    log_end = np.log(end)
    log_space = np.linspace(log_start, log_end, num_points)

    # Exponentiate back and round to nearest integers
    spaced_integers = np.round(np.exp(log_space)).astype(int)

    # Ensure unique integers
    return list(np.unique(spaced_integers))
