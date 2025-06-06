#region Libraries

#%%
from tqdm import tqdm

import numpy as np
import pandas as pd

from scipy.stats import uniform

import geopandas as gpd

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.utils_spatial.spatial_stats import get_sp_stats

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% Get storm catalogue samples
def sample_storm_catalogues(df_storms: pd.DataFrame, num_simulations=10000) -> pd.DataFrame:
    '''Get storm samples.

    Args:
        df_storms (pd.DataFrame): Storm catalogue dataframe (from "preprocess_storm_catalogue").
        num_simulations (int, optional): Number of simulations. Defaults to 10000.

    Returns:
        pd.DataFrame: Dataframe of sampled storms.
    '''
    return df_storms.sample(num_simulations, replace=True).reset_index(drop=True)

#%% Get storm center samples
#TODO Change dist_x, dist_y to dist that can take tuple of independent distributions or one single bivariate distribution
# def sample_storm_centers(v_domain_stats: pd.Series, dist_x=None, dist_y=None, num_simulations=10000) -> pd.DataFrame:
def sample_storm_centers(sp_domain: gpd.GeoDataFrame, dist_x=None, dist_y=None, num_simulations=10000) -> pd.DataFrame:
    '''Get storm center samples.

    Args:
        sp_domain (gpd.GeoDataFrame): Domain geodataframe.
        # v_domain_stats (pd.Series): Stats of the transposition domain from 'get_sp_stats'.
        dist_x (scipy.stats.*, optional): Scipy distribution object. Use None for Monte Carlo sampling. Defaults to None.
        dist_y (scipy.stats.*, optional): Scipy distribution object. Use None for Monte Carlo sampling. Defaults to None.
        num_simulations (int, optional): Number of simulations. Defaults to 10000.

    Returns:
        pd.DataFrame: Dataframe of storm centers with columns 'x_sampled' (sampled x), 'y_sampled' (sampled y), 'prob' (probabilities that sum to 1), and 'weight' (weights for importance sampling, 1 for Monte Carlo sampling).
    '''
    v_domain_stats = get_sp_stats(sp_domain)

    if dist_x is None:
        method = 0

        dist_x = uniform(v_domain_stats.minx, v_domain_stats.range_x)
        dist_y = uniform(v_domain_stats.miny, v_domain_stats.range_y)
    else:
        method = 1

    v_centroid_x = []
    v_centroid_y = []
    i = 0
    max_iter = 100
    while len(v_centroid_x) < num_simulations and i <= max_iter:

        # Get storm centers and weights
        _v_centroid_x = dist_x.rvs(num_simulations)
        _v_centroid_y = dist_y.rvs(num_simulations)

        # Filter points within domain
        sp_centers = gpd.GeoDataFrame(geometry=gpd.points_from_xy(_v_centroid_x, _v_centroid_y), crs=sp_domain.crs)

        sp_centers_within = sp_centers[sp_centers.within(sp_domain.geometry.iloc[0])]

        _v_centroid_x = sp_centers_within.geometry.x.to_list()
        _v_centroid_y = sp_centers_within.geometry.y.to_list()

        v_centroid_x += _v_centroid_x
        v_centroid_y += _v_centroid_y

        i += 1

    v_centroid_x = v_centroid_x[0:num_simulations]
    v_centroid_y = v_centroid_y[0:num_simulations]

    if method == 1:
        f_X_U = 1 / v_domain_stats.range_x
        f_Y_U = 1 / v_domain_stats.range_y
        f_X_TN = dist_x.pdf(v_centroid_x)
        f_Y_TN = dist_y.pdf(v_centroid_y)

        p = f_X_U * f_Y_U
        q = f_X_TN * f_Y_TN
        v_weight = p / q

        v_weight_norm = v_weight/v_weight.sum()
    else:
        v_weight = 1
        v_weight_norm = 1/num_simulations

    # Dataframe of centroids, depths, and weights
    df_storm_sample = pd.DataFrame(dict(
        x_sampled = v_centroid_x,
        y_sampled = v_centroid_y,
        prob = v_weight_norm,
        weight = v_weight,
    ))

    return df_storm_sample

#%%
#TODO Change dist_x, dist_y to dist that can take tuple of independent distributions or one single bivariate distribution
# def sample_storms(df_storms: pd.DataFrame, v_domain_stats: pd.Series, dist_x=None, dist_y=None, num_simulations=10000):
def sample_storms(df_storms: pd.DataFrame, sp_domain: gpd.GeoDataFrame, dist_x=None, dist_y=None, num_simulations=10000):
    '''Get storm samples and centers.

    Args:
        df_storms (pd.DataFrame): Storm catalogue dataframe (from "preprocess_storm_catalogue").
        sp_domain (gpd.GeoDataFrame): Domain geodataframe.
        # v_domain_stats (pd.Series): Stats of the transposition domain from 'get_sp_stats'.
        dist_x (scipy.stats.*, optional): Scipy distribution object. Use None for Monte Carlo sampling. Defaults to None.
        dist_y (scipy.stats.*, optional): Scipy distribution object. Use None for Monte Carlo sampling. Defaults to None.
        num_simulations (int, optional): Number of simulations. Defaults to 10000.

    Returns:
        pd.DataFrame: Dataframe of storm samples with columns 'x_sampled' (sampled x), 'y_sampled' (sampled y), 'prob' (probabilities that sum to 1), 'weight' (weights for importance sampling, 1 for Monte Carlo sampling), 'x_del' (x shift), and 'y_del' (y shift).
    '''
    _df_storm_sample = sample_storm_catalogues(df_storms=df_storms, num_simulations=num_simulations)

    tqdm._instances.clear()
    _df_storm_centers = sample_storm_centers(sp_domain=sp_domain, dist_x=dist_x, dist_y=dist_y, num_simulations=num_simulations)

    df_storm_sample = \
    (pd.concat([_df_storm_sample, _df_storm_centers], axis=1)
        .assign(x_del = lambda _: _.x_sampled - _.x)
        .assign(y_del = lambda _: _.y_sampled - _.y)
    )

    return df_storm_sample

#endregion -----------------------------------------------------------------------------------------
