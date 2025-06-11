#region Libraries

#%%
import os
import pathlib
import platform

import numpy as np
import pandas as pd

from scipy import stats

#%%
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.preprocessing.catalog_reader import read_catalog
from src.preprocessing.param_scheme_reader import read_param_scheme, get_dist_from_scheme
from src.utils_spatial.spatial_stats import get_sp_stats
from src.sst.sst_simulation import simulate_sst
from src.evaluation.sampling_eval import print_sim_stats
from src.evaluation.plotting import plot_sample_centers, plot_xy_vs_depth, plot_xy_vs_depth_2d, plot_freq_curve

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%% Select Watershed
    name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][0]
    folder_watershed = rf'D:\Scripts\Python\FEMA_FFRD_Git_PB\Importance-Sampling-for-SST\data\1_interim\{name_watershed}'

    #%% Working folder
    os.chdir(folder_watershed)
    cwd = pathlib.Path.cwd()
    
    #%% Read watershed, domain, and storm catalogue
    sp_watershed, sp_domain, df_storms = read_catalog(folder_watershed)
    df_storms = df_storms.assign(path = lambda _: _.path.apply(lambda _x: _x.absolute()))

    #%% Get polygon info (bounds, centroids, ranges)
    v_watershed_stats = get_sp_stats(sp_watershed)
    v_domain_stats = get_sp_stats(sp_domain)

    #%% Set number of simulations
    n_sim_mc = 1_000_000
    v_n_sim_is = [5_000, 10_000, 100_000]

    #%% [markdown]
    # Run simulations

    #%% Ground truth (Run once)
    # Generate samples, run simulations and get depths
    df_depths_mc_0, df_prob_mc_0, df_aep_mc_0 = simulate_sst(sp_watershed, sp_domain, df_storms, dist_x=None, dist_y=None, num_simulations=n_sim_mc)
    df_depths_mc_0.to_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_mc}.pkl')
    df_prob_mc_0.to_pickle(cwd/'pickle'/f'df_prob_mc_n_{n_sim_mc}.pkl')
    df_aep_mc_0.to_pickle(cwd/'pickle'/f'df_aep_mc_n_{n_sim_mc}.pkl')

    #%% Read parameters
    df_dist_params = read_param_scheme(folder_watershed)

    #%% For different N
    for n_sim_is in v_n_sim_is:
        # Run Monte Carlo for comparison
        # Generate samples, run simulations and get depths
        df_depths_mc, df_prob_mc, df_aep_mc = simulate_sst(sp_watershed, sp_domain, df_storms, dist_x=None, dist_y=None, num_simulations=n_sim_is)
        df_depths_mc.to_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_is}.pkl')
        df_prob_mc.to_pickle(cwd/'pickle'/f'df_prob_mc_n_{n_sim_is}.pkl')
        df_aep_mc.to_pickle(cwd/'pickle'/f'df_aep_mc_n_{n_sim_is}.pkl')

        # Run different distributions
        for row_dist_params in df_dist_params.itertuples():
            print (f'Running simulations for {row_dist_params.dist}')

            # Get distribution
            dist_x, dist_y = get_dist_from_scheme(row_dist_params, v_watershed_stats, v_domain_stats) 
                
            # Generate samples, run simulations and get depths
            df_depths_is, df_prob_is, df_aep_is = simulate_sst(sp_watershed, sp_domain, df_storms, dist_x, dist_y, num_simulations=n_sim_is)
            df_depths_is.to_pickle(cwd/'pickle'/f'df_depths_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
            df_prob_is.to_pickle(cwd/'pickle'/f'df_prob_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
            df_aep_is.to_pickle(cwd/'pickle'/f'df_aep_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')

    #%% [markdown]
    # Read simulations results

    #%% Read Ground Truth results
    df_depths_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_mc}.pkl')
    df_prob_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_prob_mc_n_{n_sim_mc}.pkl')
    df_aep_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_aep_mc_n_{n_sim_mc}.pkl')
    
    #%% Distribution of sampled points
    g = plot_sample_centers(df_depths_mc_0, sp_watershed, sp_domain, v_domain_stats)
    # g.show()
    g.save(cwd/'plots'/f'XY mc.png', width=10, height=7)

    #%% Plot depth vs coordinates (for full MC)
    g_x, g_y = plot_xy_vs_depth(df_depths_mc_0, v_watershed_stats=v_watershed_stats)
    # g_x.show()
    # g_y.show()
    g_x.save(cwd/'plots'/f'Check x vs depth for primary Monte Carlo.png', width=10, height=7)
    g_y.save(cwd/'plots'/f'Check y vs depth for primary Monte Carlo.png', width=10, height=7)

    #%% Plot depth vs coordinates 2D (for full MC)
    for stat in ['sum', 'max', 'std']:
        g_xy = plot_xy_vs_depth_2d(df_depths_mc_0, sp_watershed, sp_domain, stat=stat)
        # g_xy.show()
        g_xy.save(cwd/'plots'/f'Check xy vs depth ({stat}) for primary Monte Carlo.png', width=10, height=7)

    #%% Read and evaluate results
    for n_sim_is in v_n_sim_is:
        #%% Read Monte Carlo results
        df_depths_mc: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_is}.pkl')
        df_prob_mc: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_prob_mc_n_{n_sim_is}.pkl')
        df_aep_mc: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_aep_mc_n_{n_sim_is}.pkl')

        #%% Read and evaluate Importance Sampling results
        for row_dist_params in df_dist_params.itertuples():
            print (f'Running simulations for {row_dist_params.dist}')

            #%% Read Importance Sampling results
            df_depths_is = pd.read_pickle(cwd/'pickle'/f'df_depths_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
            df_prob_is = pd.read_pickle(cwd/'pickle'/f'df_prob_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
            df_aep_is = pd.read_pickle(cwd/'pickle'/f'df_aep_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')

            #%% Print some stats about the simulations
            print_sim_stats(df_depths_mc_0)
            print_sim_stats(df_depths_mc)
            print_sim_stats(df_depths_is)

            #%% Distribution of sampled points
            g = plot_sample_centers(df_depths_is, sp_watershed, sp_domain, v_domain_stats)
            # g.show()
            g.save(cwd/'plots'/f'XY n_{n_sim_is} {row_dist_params.name_file}.png', width=10, height=7)

            #%% Plot frequency curves
            g = plot_freq_curve([df_prob_mc_0, df_prob_mc, df_prob_is], [f'MC ({n_sim_mc/1000}k)', f'MC ({n_sim_is/1000}k)', f'IS ({n_sim_is/1000}k)'])
            # g.show()
            g.save(cwd/'plots'/f'Freq n_{n_sim_is} {row_dist_params.name_file}.png', width=10, height=7)

#endregion -----------------------------------------------------------------------------------------
