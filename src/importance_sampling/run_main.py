#region Libraries

#%%
import os
import pathlib

import numpy as np
import pandas as pd

import plotnine as pn

from scipy import stats

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
from src.stats.aep import get_return_period_langbein
from src.evaluation.sampling_eval import print_sim_stats
from src.evaluation.plotting import plot_sample_centers, plot_xy_vs_depth, plot_freq_curve

#endregion -----------------------------------------------------------------------------------------
#region Set Watershed

#%% Select Watershed
name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][2]

#endregion -----------------------------------------------------------------------------------------
#region Set Working Folder

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
#region Set Importance Sampling Parameters

#%%
df_dist_params = pd.DataFrame(dict(
    dist = ['Truncated Normal + Uniform'],
    acronym = ['tnXu'],
    param_1_name = ['std'],
    param_1 = ['0.75'], # 1 for Duwamish, 0.75 for Trinity
    param_2_name = ['w1'],
    param_2 = ['0.1'],
))

#endregion -----------------------------------------------------------------------------------------
#region Preprocess Importance Sampling Paramters

df_dist_params = \
(df_dist_params
    .assign(_p1 = lambda _: _.param_1.astype(str).str.replace(r'\.0$', '', regex=True))
    .assign(_p2 = lambda _: _.param_2.astype(str).str.replace(r'\.0$', '', regex=True))
    .assign(
        name_file = lambda _: _.acronym + '_' + _.param_1_name + '_' + _._p1 +
        np.where(_.param_2_name == '', '', '_' + _.param_2_name + '_' + _._p2)
    )
    .drop(columns=['_p1', '_p2'])
)
row = df_dist_params.iloc[0]

#endregion -----------------------------------------------------------------------------------------
#region Update Importance Sampling Distribution Parameters

#%%
match row.dist:
    case 'Truncated Nornal':
        mult_std = row.param_1

        print (f'Running for mult_std = {mult_std}')
        dist_x = stats.truncnorm(**truncnorm_params(v_watershed_stats.x, v_watershed_stats.range_x*mult_std, v_domain_stats.minx, v_domain_stats.maxx))
        dist_y = stats.truncnorm(**truncnorm_params(v_watershed_stats.y, v_watershed_stats.range_y*mult_std, v_domain_stats.miny, v_domain_stats.maxy))
    case 'Truncated Generalized Normal':
        beta = row.param_1

        print (f'Running for beta = {beta}')
        dist_x = TruncatedGeneralizedNormal(
            beta=beta,
            loc=v_watershed_stats.x,
            scale=v_watershed_stats.range_x,
            lower_bound=v_domain_stats.minx,
            upper_bound=v_domain_stats.maxx,
        )
        dist_y = TruncatedGeneralizedNormal(
            beta=beta,
            loc=v_watershed_stats.y,
            scale=v_watershed_stats.range_y,
            lower_bound=v_domain_stats.miny,
            upper_bound=v_domain_stats.maxy,
        )
    case 'Truncated T':
        mult_std = row.param_1
        dof = float(row.param_2)

        print (f'Running for mult_std = {mult_std}, dof = {dof}')
        dist_x = TruncatedDistribution(stats.t(loc=v_watershed_stats.x, scale=v_watershed_stats.range_x*mult_std, df=dof), v_domain_stats.minx, v_domain_stats.maxx)
        dist_y = TruncatedDistribution(stats.t(loc=v_watershed_stats.y, scale=v_watershed_stats.range_y*mult_std, df=dof), v_domain_stats.miny, v_domain_stats.maxy)
    case 'Truncated Normal + Uniform':
        mult_std = float(row.param_1)
        w1 = float(row.param_2)

        print (f'Running for mult_std = {mult_std}, w1 = {w1}')
        dist_x = MixtureDistribution(
            stats.uniform(v_domain_stats.minx, v_domain_stats.range_x),
            stats.truncnorm(**truncnorm_params(v_watershed_stats.x, v_watershed_stats.range_x*mult_std, v_domain_stats.minx, v_domain_stats.maxx)),
            w1
        )
        dist_y = MixtureDistribution(
            stats.uniform(v_domain_stats.miny, v_domain_stats.range_y),
            stats.truncnorm(**truncnorm_params(v_watershed_stats.y, v_watershed_stats.range_y*mult_std, v_domain_stats.miny, v_domain_stats.maxy)),
            w1
        )
    
#endregion -----------------------------------------------------------------------------------------
#region Run Simulations - Ground Truth

#%% Ground truth (Run once)
# # Generate samples
# df_storm_sample_mc_0 = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_mc)
# df_storm_sample_mc_0.to_pickle(cwd/'pickle'/'df_storm_sample_mc_0.pkl')

# # Run simulations and get depths
# df_depths_mc_0 = shift_and_compute_depth(df_storm_sample_mc_0, sp_watershed)
# df_depths_mc_0.to_pickle(cwd/'pickle'/'df_depths_mc_0.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Run Simulations - Monte Carlo for Comparison

#%% Run Monte Carlo for comparison
# # Generate samples
# df_storm_sample_mc = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_is)
# df_storm_sample_mc.to_pickle(cwd/'pickle'/f'df_storm_sample_mc_n_{n_sim_is}.pkl')

# # Run simulations and get depths
# df_depths_mc = shift_and_compute_depth(df_storm_sample_mc, sp_watershed)
# df_depths_mc.to_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_is}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Run Simulations - Importance Sampling

#%% Run Importance Sampling
# # Generate samples
# df_storm_sample_is = sample_storms(df_storms, v_domain_stats, dist_x, dist_y, num_simulations=n_sim_is)
# df_storm_sample_is.to_pickle(cwd/'pickle'/f'df_storm_sample_is_n_{n_sim_is}_{row.name_file}.pkl')

# # Run simulations and get depths
# df_depths_is = shift_and_compute_depth(df_storm_sample_is, sp_watershed)
# df_depths_is.to_pickle(cwd/'pickle'/f'df_depths_is_n_{n_sim_is}_{row.name_file}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Read Ground Truth

#%%
df_storm_sample_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_storm_sample_mc_0.pkl')
df_depths_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_depths_mc_0.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Read Monte Carlo for Comparison

#%%
df_storm_sample_mc: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_storm_sample_mc_n_{n_sim_is}.pkl')
df_depths_mc: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_is}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Read Importance Sampling

#%%
df_storm_sample_is = pd.read_pickle(cwd/'pickle'/f'df_storm_sample_is_n_{n_sim_is}_{row.name_file}.pkl')
df_depths_is = pd.read_pickle(cwd/'pickle'/f'df_depths_is_n_{n_sim_is}_{row.name_file}.pkl')

#endregion -----------------------------------------------------------------------------------------
#region Get Table of Frequency Curves

#%% 
df_freq_curve_mc_0 = get_return_period_langbein(df_depths_mc_0.depth, df_depths_mc_0.prob)
df_freq_curve_mc = get_return_period_langbein(df_depths_mc.depth, df_depths_mc.prob)
df_freq_curve_is = get_return_period_langbein(df_depths_is.depth, df_depths_is.prob)

#endregion -----------------------------------------------------------------------------------------
#region Evaluation

#%% Print some stats about the simulations
print_sim_stats(df_depths_mc_0)
print_sim_stats(df_depths_mc)
print_sim_stats(df_depths_is)

#%% Plot depth vs coordinates (for full MC)
g_x, g_y = plot_xy_vs_depth(df_depths_mc_0, v_watershed_stats=v_watershed_stats)
g_x.show()
g_y.show()
# g_x.save(cwd/'plots'/f'Check x vs depth for primary Monte Carlo.png', width=10, height=7)
# g_y.save(cwd/'plots'/f'Check y vs depth for primary Monte Carlo.png', width=10, height=7)

#%% Distribution of sampled points
g = plot_sample_centers(df_storm_sample_is, sp_watershed, sp_domain, v_domain_stats)
g.show()
# g.save(cwd/'plots'/f'XY n_{n_sim_is} {row.name_file}.png', width=10, height=7)

#%% Plot frequency curves
g = plot_freq_curve([df_freq_curve_mc_0, df_freq_curve_mc, df_freq_curve_is], [f'MC ({n_sim_mc/1000}k)', f'MC ({n_sim_is/1000}k)', f'IS ({n_sim_is/1000}k)'])
g.show()
# g.save(cwd/'plots'/f'Freq n_{n_sim_is} {row.name_file}.png', width=10, height=7)

#endregion -----------------------------------------------------------------------------------------
#region Uncertainty Analysis

#%%
# df_depths_mc_u = pd.read_pickle(cwd/'pickle'/f'df_depths_mc_u_n_{n_sim_is}.pkl')
# df_depths_is_u = pd.read_pickle(cwd/'pickle'/f'df_depths_is_u_n_{n_sim_is}_{row.name_file}.pkl')
# df_freq_curve_mc_u = pd.read_pickle(cwd/'pickle'/f'df_freq_curve_mc_u_n_{n_sim_is}.pkl')
# df_freq_curve_is_u = pd.read_pickle(cwd/'pickle'/f'df_freq_curve_is_u_n_{n_sim_is}_{row.name_file}.pkl')

df_depths_mc_u = pd.DataFrame()
df_depths_is_u = pd.DataFrame()
df_freq_curve_mc_u = pd.DataFrame()
df_freq_curve_is_u = pd.DataFrame()
for i in range(0, 100):
    # Monte Carlo
    _df_storm_sample_mc = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_is)
    _df_depths_mc = shift_and_compute_depth(_df_storm_sample_mc, sp_watershed)

    # Importance sampling
    _df_storm_sample_is = sample_storms(df_storms, v_domain_stats, dist_x, dist_y, num_simulations=n_sim_is)
    _df_depths_is = shift_and_compute_depth(_df_storm_sample_is, sp_watershed)

    # Get Table of Frequency Curves
    _df_freq_curve_mc = get_return_period_langbein(_df_depths_mc.depth, _df_depths_mc.prob)
    _df_freq_curve_is = get_return_period_langbein(_df_depths_is.depth, _df_depths_is.prob)

    # Record results
    df_depths_mc_u = pd.concat([df_depths_mc_u, _df_depths_mc.assign(iter = i)])
    df_depths_is_u = pd.concat([df_depths_is_u, _df_depths_is.assign(iter = i)])
    df_freq_curve_mc_u = pd.concat([df_freq_curve_mc_u, _df_freq_curve_mc.assign(iter = i)])
    df_freq_curve_is_u = pd.concat([df_freq_curve_is_u, _df_freq_curve_is.assign(iter = i)])

#%%
df_depths_mc_u.to_pickle(cwd/'pickle'/f'df_depths_mc_u_n_{n_sim_is}.pkl')
df_depths_is_u.to_pickle(cwd/'pickle'/f'df_depths_is_u_n_{n_sim_is}_{row.name_file}.pkl')
df_freq_curve_mc_u.to_pickle(cwd/'pickle'/f'df_freq_curve_mc_u_n_{n_sim_is}.pkl')
df_freq_curve_is_u.to_pickle(cwd/'pickle'/f'df_freq_curve_is_u_n_{n_sim_is}_{row.name_file}.pkl')

#%%
# g = plot_freq_curve([df_freq_curve_mc_u.loc[lambda _: _.iter.isin([0])], df_freq_curve_is_u.loc[lambda _: _.iter.isin([0])]], [f'MC ({n_sim_is/1000}k)', f'IS ({n_sim_is/1000}k)'])
g = plot_freq_curve([df_freq_curve_mc_0, df_freq_curve_mc_u, df_freq_curve_is_u], [f'MC ({n_sim_mc/1000}k)', f'MC ({n_sim_is/1000}k)', f'IS ({n_sim_is/1000}k)'])
# g.show()
g.save(cwd/'plots'/f'Freq u n_{n_sim_is}x{i} {row.name_file}.png', width=10, height=7)

#endregion -----------------------------------------------------------------------------------------
