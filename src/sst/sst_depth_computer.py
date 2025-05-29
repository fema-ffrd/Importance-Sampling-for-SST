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
from src.utils_spatial.zonal_stats import sum_raster_values_in_polygon
from src.sst.storm_center_shifter import shift_gdf

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% Shift storm and get zonal stat within watershed
def _shift_and_compute_depth_single(sp_watershed: gpd.GeoDataFrame, path_storm: str, shift_x: float, shift_y: float) -> float:
    #TODO implement shift method
    sp_watershed_shifted = shift_gdf(sp_watershed, shift_x, shift_y)
    _depth = sum_raster_values_in_polygon(path_storm, sp_watershed_shifted)

    return _depth

#%% For each row in df_storm_sample, compute depth (shift storm and get zonal stat within watershed)
def shift_and_compute_depth(
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
            _depth = _shift_and_compute_depth_single(sp_watershed, row.path, -row.x_del, -row.y_del)
            _v_depth.append(_depth)

            pbar.update(1)
    else:
        # Prepare arguments for each parallel task
        # tqdm can be wrapped around the iterable that Parallel consumes.
        tasks = (
            delayed(_shift_and_compute_depth_single)(sp_watershed, row.path, -row.x_del, -row.y_del)
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

#endregion -----------------------------------------------------------------------------------------
