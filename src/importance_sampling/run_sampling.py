#region Libraries

#%%
import os
import pathlib

import numpy as np
import pandas as pd

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
#region Main

if __name__ == '__main__':
    #%% Select Watershed
    name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][1]

    #%% Read parameters
    df_dist_params = pd.read_csv(r'D:\Scripts\Python\FEMA_FFRD_Git_PB\Importance-Sampling-for-SST\src\importance_sampling\distribution_params.csv')

    #%% Working folder
    os.chdir(rf'D:\Scripts\Python\FEMA_FFRD_Git_PB\Importance-Sampling-for-SST\data\1_interim\{name_watershed}')
    cwd = pathlib.Path.cwd()
    
    #%% Read watershed, domain, and storm catalogue
    sp_watershed, sp_domain, df_storms = read_catalog(cwd/'data')

    #%% Get polygon info (bounds, centroids, ranges)
    v_watershed_stats = get_sp_stats(sp_watershed)
    v_domain_stats = get_sp_stats(sp_domain)

    #%% Preprocess parameter scheme
    df_dist_params = \
    (df_dist_params
        .ffill()
        .assign(param_2_name = lambda _: _.param_2_name.replace({'-': ''}))
        .assign(param_2 = lambda _: _.param_2.replace({'-': ''}))
        .assign(_p1 = lambda _: _.param_1.astype(str).str.replace(r'\.0$', '', regex=True))
        .assign(_p2 = lambda _: _.param_2.astype(str).str.replace(r'\.0$', '', regex=True))
        .assign(
            name_file = lambda _: _.acronym + '_' + _.param_1_name + '_' + _._p1 +
            np.where(_.param_2_name == '', '', '_' + _.param_2_name + '_' + _._p2)
        )
        .drop(columns=['_p1', '_p2'])
    )

    #%% Set number of simulations
    n_sim_mc = 1_000_000

    #%% Run simulations
    #%% Ground truth (Run once)
    # Generate samples
    df_storm_sample_mc_0 = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_mc)
    df_storm_sample_mc_0.to_pickle(cwd/'pickle'/'df_storm_sample_mc_0.pkl')

    # Run simulations and get depths
    df_depths_mc_0 = shift_and_compute_depth(df_storm_sample_mc_0, sp_watershed)
    df_depths_mc_0.to_pickle(cwd/'pickle'/'df_depths_mc_0.pkl')

    #%%
    for n_sim_is in [5_000, 10_000, 100_000]:
        #%% Run Monte Carlo for comparison
        # Generate samples
        df_storm_sample_mc_1 = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_is)
        df_storm_sample_mc_1.to_pickle(cwd/'pickle'/f'df_storm_sample_mc_1_n_{n_sim_is}.pkl')

        # Run simulations and get depths
        df_depths_mc_1 = shift_and_compute_depth(df_storm_sample_mc_1, sp_watershed)
        df_depths_mc_1.to_pickle(cwd/'pickle'/f'df_depths_mc_1_n_{n_sim_is}.pkl')

        #%% Run Importance Sampling
        for row in df_dist_params.itertuples():
            print (f'Running simulations for {row.name}')

            if row.name == 'Truncated Nornal':
                mult_std = row.param_1

                print (f'Running for mult_std = {mult_std}')
                dist_x = stats.truncnorm(**truncnorm_params(v_watershed_stats.x, v_watershed_stats.range_x*mult_std, v_domain_stats.minx, v_domain_stats.maxx))
                dist_y = stats.truncnorm(**truncnorm_params(v_watershed_stats.y, v_watershed_stats.range_y*mult_std, v_domain_stats.miny, v_domain_stats.maxy))
            elif row.name == 'TruncGenNorm':
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
            elif row.name == 'TruncT':
                mult_std = row.param_1
                dof = float(row.param_2)

                print (f'Running for mult_std = {mult_std}, dof = {dof}')
                dist_x = TruncatedDistribution(stats.t(loc=v_watershed_stats.x, scale=v_watershed_stats.range_x*mult_std, df=dof), v_domain_stats.minx, v_domain_stats.maxx)
                dist_y = TruncatedDistribution(stats.t(loc=v_watershed_stats.y, scale=v_watershed_stats.range_y*mult_std, df=dof), v_domain_stats.miny, v_domain_stats.maxy)
            elif row.name == 'TruncNorm_Unif':
                mult_std = float(row.param_1)
                w1 = float(row.param_2)

                print (f'Running for mult_std = {mult_std}, alpha = {w1}')
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
                
            df_storm_sample_is = sample_storms(df_storms, v_domain_stats, dist_x, dist_y, num_simulations=n_sim_is)
            df_storm_sample_is.to_pickle(cwd/'pickle'/f'df_storm_sample_is_n_{n_sim_is}_{row.name_file}.pkl')

            df_depths_is = shift_and_compute_depth(df_storm_sample_is, sp_watershed)
            df_depths_is.to_pickle(cwd/'pickle'/f'df_depths_is_n_{n_sim_is}_{row.name_file}.pkl')

    # #%%
    # n_sim_is = 10_000

    #%% Read simulations results
    #%% Read Ground Truth results
    df_storm_sample_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_storm_sample_mc_0.pkl')
    df_depths_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_depths_mc_0.pkl')

    #%%
    for n_sim_is in [5_000, 10_000, 100_000]:
        #%% Read Monte Carlo results
        df_storm_sample_mc_1: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_storm_sample_mc_1_n_{n_sim_is}.pkl')
        df_depths_mc_1: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_depths_mc_1_n_{n_sim_is}.pkl')

        #%% Plot depth vs coordinates (for full MC)
        g_x, g_y = plot_xy_vs_depth(df_depths_mc_0, v_watershed_stats=v_watershed_stats)
        # g_x.show()
        # g_y.show()
        g_x.save(cwd/'plots'/f'Check x vs depth for primary Monte Carlo.png', width=10, height=7)
        g_y.save(cwd/'plots'/f'Check y vs depth for primary Monte Carlo.png', width=10, height=7)

        #%% Read Importance Sampling results
        for row in df_dist_params.itertuples():
            print (f'Running simulations for {row.name}')

            df_storm_sample_is_1 = pd.read_pickle(cwd/'pickle'/f'df_storm_sample_is_n_{n_sim_is}_{row.name_file}.pkl')
            df_depths_is_1 = pd.read_pickle(cwd/'pickle'/f'df_depths_is_n_{n_sim_is}_{row.name_file}.pkl')
       
            #%% Print some stats about the simulations
            print_sim_stats(df_depths_mc_0)
            print_sim_stats(df_depths_mc_1)
            print_sim_stats(df_depths_is_1)

            #%% Distribution of sampled points
            g = plot_sample_centers(df_storm_sample_is_1, sp_watershed, sp_domain, v_domain_stats)
            # g.show()
            g.save(cwd/'plots'/f'XY n_{n_sim_is} {row.name_file}.png', width=10, height=7)

            #%% Get table of frequency curves
            df_freq_curve_mc_0 = get_df_freq_curve(df_depths_mc_0.depth, df_depths_mc_0.prob)
            df_freq_curve_mc_1 = get_df_freq_curve(df_depths_mc_1.depth, df_depths_mc_1.prob)
            df_freq_curve_is_1 = get_df_freq_curve(df_depths_is_1.depth, df_depths_is_1.prob)

            #%% Plot frequency curves
            g = plot_freq_curve([df_freq_curve_mc_0, df_freq_curve_mc_1, df_freq_curve_is_1], [f'MC ({n_sim_mc/1000}k)', f'MC ({n_sim_is/1000}k)', f'IS ({n_sim_is/1000}k)'])
            # g.show()
            g.save(cwd/'plots'/f'Freq n_{n_sim_is} {row.name_file}.png', width=10, height=7)

#endregion -----------------------------------------------------------------------------------------
