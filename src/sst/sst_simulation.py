#region Libraries

#%%
import numpy as np
import pandas as pd

import geopandas as gpd

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.sst.storm_sampler import sample_storms
from src.sst.sst_depth_computer import shift_and_compute_depth
from src.stats.aep import get_return_period_langbein, get_aep_depths

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
#TODO
def simulate_sst(sp_watershed: gpd.GeoDataFrame, sp_domain: gpd.GeoDataFrame, df_storms: pd.DataFrame, dist_x=None, dist_y=None, num_simulations=10000, return_period: list|np.ndarray|pd.Series = [2, 2.5, 4, 5, 6.67, 10, 12.5, 20, 25, 33.33, 50, 75, 100, 150, 200, 250, 350, 500, 750, 1000, 2000]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_storm_sample = sample_storms(df_storms, sp_domain, dist_x, dist_y, num_simulations)
    df_depths = shift_and_compute_depth(df_storm_sample, sp_watershed)
    df_prob = get_return_period_langbein(df_depths)
    df_aep = get_aep_depths(df_prob)

    return df_depths, df_prob, df_aep

#endregion -----------------------------------------------------------------------------------------
