#region Libraries

#%%
from typing import Literal

from tqdm import tqdm

from joblib import Parallel, delayed

import numpy as np
import pandas as pd

import geopandas as gpd

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from compute_raster_stats import sum_raster_values_in_polygon
from shift_storm_center import shift_gdf

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
def _compute_depth_single(sp_watershed: gpd.GeoDataFrame, path_storm: str, shift_x: float, shift_y: float) -> float:
    #TODO implement shift method
    sp_watershed_shifted = shift_gdf(sp_watershed, shift_x, shift_y)
    _depth = sum_raster_values_in_polygon(path_storm, sp_watershed_shifted)

    return _depth

#%%
def compute_depths(
    df_storm_sample: pd.DataFrame,
    sp_watershed: gpd.GeoDataFrame,
    shift: Literal['watershed', 'storm', 'best'] = 'watershed',
    parallel = True,
    n_jobs = -2,
) -> pd.DataFrame:
    '''Compute storm depths based on storm samples and watershed.

    Args:
        df_storm_sample (pd.DataFrame): Dataframe of storm samples obtained from 'sample_storms'.
        sp_watershed (gpd.GeoDataFrame): Watershed to compute depths in.
        shift (Literal['watershed', 'storm', 'best'], optional): Whether to shift the watershed or the storm. Defaults to 'watershed'.
        parallel (bool, optional): Whether to use parallel processing. Defaults to True.
        n_jobs (int, optional): Maximum number of concurrenlty running jobs. Only applicable if 'parallel' is True. If negative, (n_cpus + 1 + n_jobs) are used. Defaults to -2 (max cores - 1).

    Returns:
        pd.DataFrame: Updated 'df_storm_sample' with depths in 'depth' column.
    '''
    tqdm._instances.clear()
    if not parallel:
        pbar = tqdm(total=df_storm_sample.shape[0])

        _v_depth = []
        for i, row in df_storm_sample.iterrows():
            _depth = _compute_depth_single(sp_watershed, row.path, -row.x_del, -row.y_del)
            _v_depth.append(_depth)

            pbar.update(1)
    else:
        # Prepare arguments for each parallel task
        # tqdm can be wrapped around the iterable that Parallel consumes.
        tasks = (
            delayed(_compute_depth_single)(sp_watershed, row.path, -row.x_del, -row.y_del)
            for row in df_storm_sample[['path', 'x_del', 'y_del']].itertuples(index=False)
        )
    
        # Creating a list for tqdm to get total count
        task_list = list(tasks)
        
        _v_depth = Parallel(n_jobs=n_jobs, backend="loky")(
            tqdm(task_list, total=len(task_list), desc="Computing depths")
        )

    df_storm_sample = df_storm_sample.assign(depth = _v_depth)

    df_storm_sample = \
    (df_storm_sample
        .assign(intersected = lambda _: np.where(_.depth.isna(), 0, 1))
        .fillna({'depth': 0})
    )

    return df_storm_sample

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

#endregion -----------------------------------------------------------------------------------------
