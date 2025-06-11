#region Libraries

#%%
import os
import pathlib

import numpy as np
import pandas as pd

import platform

#%%
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.preprocessing.catalog_reader import read_catalog
from src.preprocessing.param_scheme_reader import get_dist_from_scheme
from src.sst.storm_sampler import sample_storms
from src.utils_spatial.spatial_stats import get_sp_stats
from src.stats.distribution_helpers import truncnorm_params, fit_rotated_normal_to_polygon
from src.stats.distributions import TruncatedGeneralizedNormal, TruncatedDistribution, MixtureDistribution, RotatedNormal
from src.sst.sst_simulation import simulate_sst
from src.evaluation.sampling_eval import print_sim_stats
from src.evaluation.plotting import plot_sample_centers, plot_xy_vs_depth, plot_xy_vs_depth_2d, plot_freq_curve
from src.evaluation.metrics import get_aep_rmse, get_aep_rmse_iter

#endregion -----------------------------------------------------------------------------------------
#region Set Watershed

#%% Select Watershed
name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][2]
folder_watershed = rf'D:\Scripts\Python\FEMA_FFRD_Git_PB\Importance-Sampling-for-SST\data\1_interim\{name_watershed}'

#endregion -----------------------------------------------------------------------------------------
#region Set Working Folder

#%% Working folder
os.chdir(folder_watershed)
cwd = pathlib.Path.cwd()

#endregion -----------------------------------------------------------------------------------------
#region Read Watershed and Storm Data

#%% Read watershed, domain, and storm catalogue
sp_watershed, sp_domain, df_storms = read_catalog(cwd)
df_storms = df_storms.assign(path = lambda _: _.path.apply(lambda _x: _x.absolute()))

#%% Get polygon info (bounds, centroids, ranges)
v_watershed_stats = get_sp_stats(sp_watershed)
v_domain_stats = get_sp_stats(sp_domain)

#endregion -----------------------------------------------------------------------------------------
#region Set Number of Simulations

#%% Set number of simulations
n_sim_mc = 1_000_000
n_sim_is = 100_000

#endregion -----------------------------------------------------------------------------------------
#region Run Simulations - Ground Truth

#%% Ground truth (Run once)
df_depths_mc_0, df_prob_mc_0, df_aep_mc_0 = simulate_sst(sp_watershed, sp_domain, df_storms, dist_x=None, dist_y=None, num_simulations=n_sim_mc)
df_depths_mc_0.to_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_mc}.pkl')
df_prob_mc_0.to_pickle(cwd/'pickle'/f'df_prob_mc_n_{n_sim_mc}.pkl')
df_aep_mc_0.to_pickle(cwd/'pickle'/f'df_aep_mc_n_{n_sim_mc}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Run Simulations - Monte Carlo for Comparison

#%% Run Monte Carlo for comparison
df_depths_mc, df_prob_mc, df_aep_mc = simulate_sst(sp_watershed, sp_domain, df_storms, dist_x=None, dist_y=None, num_simulations=n_sim_is)
df_depths_mc.to_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_is}.pkl')
df_prob_mc.to_pickle(cwd/'pickle'/f'df_prob_mc_n_{n_sim_is}.pkl')
df_aep_mc.to_pickle(cwd/'pickle'/f'df_aep_mc_n_{n_sim_is}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Set Importance Sampling Parameters

# #%%
# df_dist_params = pd.DataFrame(dict(
#     dist = ['Truncated Normal + Uniform'],
#     acronym = ['tnXu'],
#     param_1_name = ['std'],
#     param_1 = ['1'], # 1 for Duwamish, 0.5 for Kanahwa, 0.75 for Trinity
#     param_2_name = ['w1'],
#     param_2 = ['0.1'],
# ))

#%%
_coverage_factor = 0.8

#endregion -----------------------------------------------------------------------------------------
#region Update Importance Sampling Distribution Parameters

# #%%
# df_dist_params = \
# (df_dist_params
#     .assign(_p1 = lambda _: _.param_1.astype(str).str.replace(r'\.0$', '', regex=True))
#     .assign(_p2 = lambda _: _.param_2.astype(str).str.replace(r'\.0$', '', regex=True))
#     .assign(
#         name_file = lambda _: _.acronym + '_' + _.param_1_name + '_' + _._p1 +
#         np.where(_.param_2_name == '', '', '_' + _.param_2_name + '_' + _._p2)
#     )
#     .drop(columns=['_p1', '_p2'])
# )
# row_dist_params = df_dist_params.iloc[0]

# #%% Get distribution
# dist_x, dist_y = get_dist_from_scheme(row_dist_params, v_watershed_stats, v_domain_stats)
# dist_xy = None

#%%
_param_dist_xy = fit_rotated_normal_to_polygon(sp_domain.explode().geometry.iloc[0], coverage_factor=_coverage_factor)
row_dist_params = pd.Series(dict(name_file = 'RN_cf_08'))
dist_xy = RotatedNormal(mean=[v_watershed_stats.x, v_watershed_stats.y], stds=_param_dist_xy.get('stds'), angle_degrees=_param_dist_xy.get('angle_degrees'))
dist_x = None
dist_y = None

#endregion -----------------------------------------------------------------------------------------
#region Generate Samples and Evaluate

#%% Plot depth vs coordinates (for full MC)
g_x, g_y = plot_xy_vs_depth(df_depths_mc_0, v_watershed_stats=v_watershed_stats)
g_x.show()
g_y.show()
# g_x.save(cwd/'plots'/f'Check x vs depth for primary Monte Carlo.png', width=10, height=7)
# g_y.save(cwd/'plots'/f'Check y vs depth for primary Monte Carlo.png', width=10, height=7)

#%% Plot depth vs coordinates 2D (for full MC)
g_xy = plot_xy_vs_depth_2d(df_depths_mc_0, sp_watershed, sp_domain, 'std')
g_xy.show()
# g_xy.save(cwd/'plots'/f'Check xy vs depth for primary Monte Carlo.png', width=10, height=7)

#%% Distribution of sampled points
# df_storm_sample_is = sample_storms(df_storms, sp_domain, dist_x, dist_y, num_simulations=n_sim_is)
_param_dist_xy = fit_rotated_normal_to_polygon(sp_domain.explode().geometry.iloc[0], coverage_factor=0.8)
row = pd.Series(dict(name_file = 'RN_cf_08'))
dist_xy = RotatedNormal(mean=[v_watershed_stats.x, v_watershed_stats.y], stds=_param_dist_xy.get('stds'), angle_degrees=_param_dist_xy.get('angle_degrees'))
df_storm_sample_is = sample_storms(df_storms, sp_domain, dist_xy = dist_xy, num_simulations=n_sim_is)

g = plot_sample_centers(df_storm_sample_is, sp_watershed, sp_domain, v_domain_stats)
g.show()
# g.save(cwd/'plots'/f'XY n_{n_sim_is} {row.name_file}.png', width=10, height=7)

#endregion -----------------------------------------------------------------------------------------
#region Run Simulations - Importance Sampling

#%% Run Importance Sampling
df_depths_is, df_prob_is, df_aep_is = simulate_sst(sp_watershed, sp_domain, df_storms, dist_x=dist_x, dist_y=dist_y, dist_xy=dist_xy, num_simulations=n_sim_is)
df_depths_is.to_pickle(cwd/'pickle'/f'df_depths_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
df_prob_is.to_pickle(cwd/'pickle'/f'df_prob_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
df_aep_is.to_pickle(cwd/'pickle'/f'df_aep_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Read Ground Truth

#%%
df_depths_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_mc}.pkl')
df_prob_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_prob_mc_n_{n_sim_mc}.pkl')
df_aep_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_aep_mc_n_{n_sim_mc}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Read Monte Carlo for Comparison

#%%
df_depths_mc: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_is}.pkl')
df_prob_mc: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_prob_mc_n_{n_sim_is}.pkl')
df_aep_mc: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_aep_mc_n_{n_sim_is}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Read Importance Sampling

#%%
df_depths_is = pd.read_pickle(cwd/'pickle'/f'df_depths_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
df_prob_is = pd.read_pickle(cwd/'pickle'/f'df_prob_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
df_aep_is = pd.read_pickle(cwd/'pickle'/f'df_aep_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Evaluation

#%% Print some stats about the simulations
print_sim_stats(df_depths_mc_0)
print_sim_stats(df_depths_mc)
print_sim_stats(df_depths_is)

#%% Distribution of sampled points
g = plot_sample_centers(df_depths_is, sp_watershed, sp_domain, v_domain_stats)
g.show()
# g.save(cwd/'plots'/f'XY n_{n_sim_is} {row.name_file}.png', width=10, height=7)

#%% Plot frequency curves
g = plot_freq_curve([df_prob_mc_0, df_prob_mc, df_prob_is], [f'MC ({n_sim_mc/1000}k)', f'MC ({n_sim_is/1000}k)', f'IS ({n_sim_is/1000}k)'])
g.show()
# g.save(cwd/'plots'/f'Freq n_{n_sim_is} {row.name_file}.png', width=10, height=7)

#endregion -----------------------------------------------------------------------------------------
#region Ground Truth Evaluation

# #%%
# import scipy.stats as stats

#%%
_df_aep_mc_0_1 = pd.read_pickle(cwd/'pickle/df_aep_mc_n_1000000.pkl')
_df_aep_mc_0_2 = pd.read_pickle(cwd/'pickle/df_aep_mc_n_10000000.pkl')

get_aep_rmse(_df_aep_mc_0_1, _df_aep_mc_0_2)

#%%
df_aep_mc_0 = pd.read_pickle(cwd/'pickle/df_aep_mc_n_1000000.pkl')
# _df_aep_mc_0_2 = pd.read_pickle(cwd/'pickle/df_aep_summary_mc_iter_n_5000x100.pkl').loc[lambda _: _.type_val == 'median']
# df_aep_iter = pd.read_pickle(cwd/'pickle/df_aep_mc_iter_n_100000x10.pkl')
df_aep_iter = pd.read_pickle(cwd/'pickle/df_aep_is_iter_n_100000x10.pkl')

#%%
get_aep_rmse_iter(df_aep_mc_0, df_aep_iter)

#endregion -----------------------------------------------------------------------------------------
#region Read Storn Info

#%%
df_storm_stats = pd.read_pickle(cwd/'pickle'/'df_storm_stats.pkl')

#endregion -----------------------------------------------------------------------------------------
