#region Libraries

#%%
from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy.stats import uniform

import geopandas as gpd

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% Get storm center samples
def sample_storm_centers(v_domain_stats: pd.Series, dist_x=None, dist_y=None, num_simulations=10000):
    if dist_x is None:
        method = 0

        dist_x = uniform(v_domain_stats.minx, v_domain_stats.range_x)
        dist_y = uniform(v_domain_stats.miny, v_domain_stats.range_y)
    else:
        method = 1

    # Get storm centers and weights
    v_centroid_x = dist_x.rvs(num_simulations)
    v_centroid_y = dist_y.rvs(num_simulations)

    if method == 1:
        v_weight = []
        pbar = tqdm(total=num_simulations)
        for i in range(num_simulations):
            # Sample centroid of storm from truncated normal distributions
            centroid_x = v_centroid_x[i]
            centroid_y = v_centroid_y[i]
            
            # Compute weight of each depth
            f_X_U = 1 / v_domain_stats.range_x
            f_Y_U = 1 / v_domain_stats.range_y
            f_X_TN = dist_x.pdf(centroid_x)
            f_Y_TN = dist_y.pdf(centroid_y)
            p = f_X_U * f_Y_U
            q = f_X_TN * f_Y_TN
            weight = p / q if q > 0 else 0
            v_weight.append(weight)

            # Update progress bar
            pbar.update(1)

            # Normalize weights
            v_weight_norm = np.array(v_weight)
            v_weight_norm /= v_weight_norm.sum()
    else:
        v_weight = 1
        v_weight_norm = 1/num_simulations

    # Dataframe of centroids, depths, and weights
    df = pd.DataFrame(dict(
        x_sampled = v_centroid_x,
        y_sampled = v_centroid_y,
        prob = v_weight_norm,
        weight = v_weight,
    ))

    return df

#%% Function to pass parameters to truncnorm
def truncnorm_params(mean, std_dev, lower, upper):
    d = dict(
        a = (lower - mean) / std_dev, 
        b = (upper - mean) / std_dev, 
        loc = mean, 
        scale = std_dev
    )

    return d    

#%%
def get_sp_stats(gdf: gpd.GeoDataFrame) -> pd.Series:
    v_sp_stats = \
    (pd.concat(
        [
            gdf.bounds,
            gdf.centroid.get_coordinates()
        ],
        axis=1,
    )
        .assign(range_x = lambda _: _.maxx - _.minx)
        .assign(range_y = lambda _: _.maxy - _.miny)
        .iloc[0]
    )

    return v_sp_stats    

#%% Get storm catalogue samples
def sample_storm_catalogues(df_storms: pd.DataFrame, num_simulations=10000) -> pd.DataFrame:
    return df_storms.sample(num_simulations, replace=True).reset_index(drop=True)

#%%
def sample_storms(df_storms: pd.DataFrame, v_domain_stats: pd.Series, dist_x=None, dist_y=None, num_simulations=10000):
    _df_storm_sample = sample_storm_catalogues(df_storms=df_storms, num_simulations=num_simulations)
    
    tqdm._instances.clear()
    _df_storm_centers = sample_storm_centers(v_domain_stats=v_domain_stats, dist_x=dist_x, dist_y=dist_y, num_simulations=num_simulations)
    
    df_storm_sample = \
    (pd.concat([_df_storm_sample, _df_storm_centers], axis=1)
        .assign(x_del = lambda _: _.x_sampled - _.x)
        .assign(y_del = lambda _: _.y_sampled - _.y)
    )

    return df_storm_sample

#endregion -----------------------------------------------------------------------------------------
