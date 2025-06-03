#region Libraries

#%%
import os
import pathlib

import numpy as np
import pandas as pd

from scipy import stats

import platform

import plotnine as pn

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
from src.stats.aep import get_return_period_langbein, get_aep_depths, get_aep_uncertainty
from src.evaluation.sampling_eval import print_sim_stats
from src.evaluation.plotting import plot_sample_centers, plot_xy_vs_depth, plot_freq_curve

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%% Select Watershed
    name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][2]

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
    v_n_sim_is = [10_000, 100_000]

    #%%
    match name_watershed:
        case 'Duwamish':
            row = df_dist_params.iloc[20]
        case 'Kanahwa':
            row = df_dist_params.iloc[18]
        case 'Trinity':
            row = df_dist_params.iloc[19]

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
    
    #%% Perform uncertainty analysis
    for n_sim_is in v_n_sim_is:
        df_depths_mc_u = pd.DataFrame()
        df_depths_is_u = pd.DataFrame()
        df_freq_curve_mc_u = pd.DataFrame()
        df_freq_curve_is_u = pd.DataFrame()
        for i in range(int(n_sim_mc/n_sim_is)):
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

        # Save uncertainty analysis results
        df_depths_mc_u.to_pickle(cwd/'pickle'/f'df_depths_mc_u_n_{n_sim_is}.pkl')
        df_depths_is_u.to_pickle(cwd/'pickle'/f'df_depths_is_u_n_{n_sim_is}_{row.name_file}.pkl')
        df_freq_curve_mc_u.to_pickle(cwd/'pickle'/f'df_freq_curve_mc_u_n_{n_sim_is}.pkl')
        df_freq_curve_is_u.to_pickle(cwd/'pickle'/f'df_freq_curve_is_u_n_{n_sim_is}_{row.name_file}.pkl')
    
    #%% Read uncertainty analysis results
    df_freq_curve_mc_u: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_freq_curve_mc_u_n_{n_sim_is}.pkl')
    df_freq_curve_is_u: pd.DataFrame = pd.read_pickle(cwd/'pickle'/f'df_freq_curve_is_u_n_{n_sim_is}_{row.name_file}.pkl')
    df_depths_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_depths_mc_0.pkl')
    df_freq_curve_mc_0 = get_return_period_langbein(df_depths_mc_0.depth, df_depths_mc_0.prob)
    
    #%% Plot uncertainty analysis results
    for n_sim_is in v_n_sim_is:
        #%%
        df_depths_is_u = pd.read_pickle(cwd/'pickle'/f'df_depths_is_u_n_{n_sim_is}_{row.name_file}.pkl')
        df_freq_curve_is_u = pd.read_pickle(cwd/'pickle'/f'df_freq_curve_is_u_n_{n_sim_is}_{row.name_file}.pkl')
    
        #%%
        df_aep_mc, df_aep_summary_mc = get_aep_uncertainty(df_freq_curve_mc_u)
        df_aep_is, df_aep_summary_is = get_aep_uncertainty(df_freq_curve_is_u)
        df_aep_mc_0, df_aep_summary_mc_0 = get_aep_uncertainty(df_freq_curve_mc_0.assign(iter = 0))
    
        #%%
        g = \
        (pn.ggplot(mapping = pn.aes(x = 'return_period', y = 'depth', group = 'type', linetype='type'))
            + pn.geom_line(data = df_aep_summary_mc, mapping = pn.aes(color='"MC"'))
            + pn.geom_line(data = df_aep_summary_is, mapping = pn.aes(color='"IS"'))
            + pn.geom_line(data = df_aep_summary_mc_0, mapping = pn.aes(color='"Truth"'))
            # + pn.geom_line(data = df_aep_mc_0, mapping = pn.aes(color='"Truth"'))
            + pn.scale_x_log10()
            + pn.labs(
                # x = 'Return Period',
                # x = 'Exceedence Probability',
                x = 'Return Period',
                y = 'Rainfall Depth',
                title = f'Uncertainty (N={n_sim_is}, iter={int(n_sim_mc/n_sim_is)})'
            )
            + pn.theme_bw()
            + pn.theme(
                title = pn.element_text(hjust = 0.5),
                # legend_position = 'bottom',
                legend_title = pn.element_blank(),
                legend_key = pn.element_blank(),
                axis_title_y = pn.element_text(ha = 'left'),
            )
        )
        # g.show()
        g.save(cwd/'plots'/f'Freq u n_{n_sim_is}x{int(n_sim_mc/n_sim_is)} {row.name_file}.png', width=10, height=7)
    
#endregion -----------------------------------------------------------------------------------------
