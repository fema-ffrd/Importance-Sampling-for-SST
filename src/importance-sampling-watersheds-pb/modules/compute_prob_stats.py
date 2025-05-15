#region Libraries

#%%
import numpy as np
import pandas as pd

import geopandas as gpd

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% Print simulation statistics
def print_sim_stats(df_depths: pd.DataFrame, multiplier: float=1) -> None:
    '''Print various simulation statistics such as proportion of samples with non-zero depth, and mean and standard error of depths.

    Args:
        df_depths (pd.DataFrame): Dataframe of probabilities from 'compute_depths'.
        multiplier (float, optional): A multiplier for deth values. Defaults to 1.
    '''
    n_sim = df_depths.shape[0]
    n_sim_intersect = df_depths.loc[lambda _: _.intersected == 1].shape[0]
    rate_success = n_sim_intersect/n_sim*100

    prob_total = df_depths.prob.sum()
    prob_intersected = df_depths.loc[lambda _: _.intersected == 1].prob.sum()

    df_depths = \
    (df_depths
        .assign(x_px = lambda _: _.depth * _.prob)
    )
    mean = df_depths.x_px.sum()
    df_depths = \
    (df_depths
        .assign(x_mx_px = lambda _: ((_.depth - mean)**2) * _.prob)
    )
    std = np.sqrt(df_depths.x_mx_px.sum())
    standard_error = std/np.sqrt(n_sim)

    depth_weighted = df_depths.depth * df_depths.weight
    mean_estimate = np.mean(depth_weighted)
    std_estimate = np.std(depth_weighted, ddof=1) # Sample std dev of h(x)*w(x)
    standard_error_estimate = std_estimate / np.sqrt(n_sim)

    print(
        f'Intersected: {n_sim_intersect} out of {n_sim} ({rate_success:.2f}%)\n'
        + f'Total Weights: Total {prob_total: .2f}, Intersected: {prob_intersected:.2f}\n'
        + f'Depth: {mean*multiplier:.2f} ± {standard_error*multiplier:.2f}\n'
        + f'Depth Estimate: {mean_estimate*multiplier:.2f} ± {standard_error_estimate*multiplier:.2f}'
    )

#%% Create probability dataframe from depths (sorted) and weights (sorted)
def get_df_freq_curve(depths: list|np.ndarray|pd.Series, probs: list|np.ndarray|pd.Series) -> pd.DataFrame:
    '''Generate frequency distribution curve datafra.e

    Args:
        depths (list | np.ndarray | pd.Series): Vector of depths.
        probs (list | np.ndarray | pd.Series): Vector of probabilities.

    Returns:
        pd.DataFrame: Dataframe with inverse sorted depths and corresponding probabilities, exceedence probabilities, and return periods.
    '''
    # Table of depths and probabilities
    df_prob_mc = pd.DataFrame(dict(
        depth = depths,
        prob = probs
    ))

    # Exceedence probability
    df_prob_mc = \
    (df_prob_mc
        .sort_values('depth', ascending=False)
        .assign(prob_exceed = lambda _: _.prob.cumsum())
        .assign(return_period = lambda _: 1/_.prob_exceed)
    )

    return df_prob_mc

#%%
def get_prob(df_depths: pd.DataFrame, greater_than: list = None,  greater_than_incl = True, less_than: list = None,less_than_incl = True) -> pd.DataFrame:
    '''Calculates the probability of depth being greater than or less than a value or between two values.

    Args:
        df_depths (pd.DataFrame): Dataframe of probabilities from 'compute_depths'.
        greater_than: The threshold greater than which to calculate the probability.
        greater_than_incl: If True, condition is depth >= greater_than.
                           If False, condition is depth > greater_than.
        less_than: The threshold less than which to calculate the probability.
        less_than_incl: If True, condition is depth <= less_than.
                        If False, condition is depth < less_than.
    '''
    v_depth = df_depths.depth
    v_weight = df_depths.weight
    if greater_than is not None:
        if greater_than_incl:
            indicator = (v_depth >= greater_than)
        else:
            indicator = (v_depth > greater_than)
        prob_greater_than = np.mean(indicator * v_weight)
    if less_than is not None:
        if less_than_incl:
            indicator = (v_depth <= greater_than)
        else:
            indicator = (v_depth < greater_than)
        prob_less_than = np.mean(indicator * v_weight)

    if greater_than is not None and less_than is not None:
        prob = prob_less_than - prob_greater_than
    elif greater_than is not None:
        prob = prob_greater_than
    elif less_than is not None:
        prob = prob_less_than
    else:
        prob = 1

    return prob

#endregion -----------------------------------------------------------------------------------------
