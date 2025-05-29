#region Libraries

#%%
import os
import pathlib

import pandas as pd

import plotnine as pn

from scipy import stats

import geopandas as gpd

import platform

#%%
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.preprocessing.catalog_reader import read_catalog
from src.utils_spatial.spatial_stats import get_sp_stats
from src.sst.storm_sampler import sample_storms
from src.sst.sst_depth_computer import shift_and_compute_depth
from src.stats.distributions import TruncatedGeneralizedNormal, TruncatedDistribution, MixtureDistribution
from src.stats.distribution_helpers import truncnorm_params
from src.stats.aep import get_df_freq_curve
from src.evaluation.sampling_eval import print_sim_stats
from src.evaluation.plotting import plot_sample_centers, plot_xy_vs_depth, plot_freq_curve

#endregion -----------------------------------------------------------------------------------------
#region Select Watershed

#%% Select Watershed
name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][0]

#%% Working folder
os.chdir(rf'D:\Scripts\Python\FEMA_FFRD_Git_PB\Importance-Sampling-for-SST\data\1_interim\{name_watershed}')
cwd = pathlib.Path.cwd()

#endregion -----------------------------------------------------------------------------------------
#region Read Data

#%% Read watershed, domain, and storm catalogue
sp_watershed, sp_domain, df_storms = read_catalog(cwd/'data')

#%% Get polygon info (bounds, centroids, ranges)
v_watershed_stats = get_sp_stats(sp_watershed)
v_domain_stats = get_sp_stats(sp_domain)

#endregion -----------------------------------------------------------------------------------------
#region Set Number of Simulations

#%% Set number of simulations
n_sim_mc = 1_000_000
n_sim_is = 10_000

#endregion -----------------------------------------------------------------------------------------
#region Read Ground Truth

#%%
df_storm_sample_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_storm_sample_mc_0.pkl')
df_depths_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_depths_mc_0.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Read Monte Carlo for Comparison

#%%
df_storm_sample_mc_1: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_storm_sample_mc_1_n_{n_sim_is}.pkl')
df_depths_mc_1: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_depths_mc_1_n_{n_sim_is}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region 



#endregion -----------------------------------------------------------------------------------------
