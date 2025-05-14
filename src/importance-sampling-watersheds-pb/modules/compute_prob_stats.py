#region Libraries

#%%
import numpy as np
import pandas as pd

import geopandas as gpd

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% Print simulation statistics
def print_sim_stats(df_prob: pd.DataFrame, multiplier: float=1) -> None:
    '''Print various simulation statistics such as proportion of samples with non-zero depth, and mean and standard error of depths.

    Args:
        df_prob (pd.DataFrame): Dataframe of probabilities from 'compute_depths'.
        multiplier (float, optional): A multiplier for deth values. Defaults to 1.
    '''
    n_sim = df_prob.shape[0]
    n_sim_intersect = df_prob.loc[lambda _: _.intersected == 1].shape[0]
    rate_success = n_sim_intersect/n_sim*100

    prob_total = df_prob.prob.sum()
    prob_intersected = df_prob.loc[lambda _: _.intersected == 1].prob.sum()

    df_prob = \
    (df_prob
        .assign(x_px = lambda _: _.depth * _.prob)
    )
    mean = df_prob.x_px.sum()
    df_prob = \
    (df_prob
        .assign(x_mx_px = lambda _: ((_.depth - mean)**2) * _.prob)
    )
    std = np.sqrt(df_prob.x_mx_px.sum())
    standard_error = std/np.sqrt(n_sim)

    depth_weighted = df_prob.depth * df_prob.weight
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
def get_prob(df_depths: pd.DataFrame, greater_than: list = None, less_than: list = None, greater_than_equals: list = None, less_than_equals: list = None) -> pd.DataFrame:
    pass

#endregion -----------------------------------------------------------------------------------------
