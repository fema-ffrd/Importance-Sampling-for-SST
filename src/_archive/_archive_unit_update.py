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
from src.stats.aep import get_return_period_langbein, get_aep_depths
from src.evaluation.sampling_eval import print_sim_stats
from src.evaluation.plotting import plot_sample_centers, plot_xy_vs_depth, plot_freq_curve

#endregion -----------------------------------------------------------------------------------------
#region Main

#%% Select Watershed
name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][2]
folder_watershed = rf'D:\Scripts\Python\FEMA_FFRD_Git_PB\Importance-Sampling-for-SST\data\1_interim\{name_watershed}'

#%% Working folder
os.chdir(folder_watershed)
cwd = pathlib.Path.cwd()

#%% Read watershed, domain, and storm catalogue
sp_watershed, sp_domain, df_storms = read_catalog(folder_watershed)

#%% Read parameters
df_dist_params = read_param_scheme(folder_watershed)

#%% Get polygon info (bounds, centroids, ranges)
v_watershed_stats = get_sp_stats(sp_watershed)
v_domain_stats = get_sp_stats(sp_domain)

#%% Set number of simulations
n_sim_mc = 1_000_000
v_n_sim_is = [5_000, 10_000, 100_000]

#%% Read Ground Truth results
df_depths_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_depths_mc_0.pkl')

#%%
df_depths_mc_0 = df_depths_mc_0.assign(depth = lambda _: _.depth*1/25.4*1/(sp_watershed.area.iloc[0]/(4e3)**2))
df_prob_mc_0 = get_return_period_langbein(df_depths_mc_0)
df_aep_mc_0 = get_aep_depths(df_prob_mc_0)

df_depths_mc_0.to_pickle(cwd/'pickle'/f'df_depths_mc_0.pkl')
df_prob_mc_0.to_pickle(cwd/'pickle'/f'df_prob_mc_0.pkl')
df_aep_mc_0.to_pickle(cwd/'pickle'/f'df_aep_mc_0.pkl')

#%% Read and evaluate results
for n_sim_is in v_n_sim_is:
    #%% Read Monte Carlo results
    df_depths_mc: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_is}.pkl')

    #%%
    df_depths_mc = df_depths_mc.assign(depth = lambda _: _.depth*1/25.4*1/(sp_watershed.area.iloc[0]/(4e3)**2))
    df_prob_mc = get_return_period_langbein(df_depths_mc)
    df_aep_mc = get_aep_depths(df_prob_mc)
    
    df_depths_mc.to_pickle(cwd/'pickle'/f'df_depths_mc_n_{n_sim_is}.pkl')
    df_prob_mc.to_pickle(cwd/'pickle'/f'df_prob_mc_n_{n_sim_is}.pkl')
    df_aep_mc.to_pickle(cwd/'pickle'/f'df_aep_mc_n_{n_sim_is}.pkl')
    

    #%% Read and evaluate Importance Sampling results
    for row_dist_params in df_dist_params.itertuples():
        print (f'Running simulations for {row_dist_params.dist}')

        #%% Read Importance Sampling results
        df_depths_is = pd.read_pickle(cwd/'pickle'/f'df_depths_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')

        #%%
        df_depths_is = df_depths_is.assign(depth = lambda _: _.depth*1/25.4*1/(sp_watershed.area.iloc[0]/(4e3)**2))
        df_prob_is = get_return_period_langbein(df_depths_is)
        df_aep_is = get_aep_depths(df_prob_mc)
        
        df_depths_is.to_pickle(cwd/'pickle'/f'df_depths_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
        df_prob_is.to_pickle(cwd/'pickle'/f'df_prob_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
        df_aep_is.to_pickle(cwd/'pickle'/f'df_aep_is_n_{n_sim_is}_{row_dist_params.name_file}.pkl')
    


#endregion -----------------------------------------------------------------------------------------
