#region Libraries

#%%
import os
import pathlib
import platform

import numpy as np
import pandas as pd

import plotnine as pn

#%%
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.preprocessing.catalog_reader import read_catalog
from src.preprocessing.param_scheme_reader import read_param_scheme, get_dist_from_scheme
from src.utils_spatial.spatial_stats import get_sp_stats
from src.sst.sst_simulation import simulate_sst_iter
from src.evaluation.plotting import plot_sample_centers
from src.evaluation.metrics import get_aep_rmse
from src.stats.distribution_helpers import truncnorm_params, fit_rotated_normal_to_polygon
from src.stats.distributions import TruncatedGeneralizedNormal, TruncatedDistribution, MixtureDistribution, RotatedNormal

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%% Select Watershed
    name_watershed = ['Duwamish', 'Kanahwa', 'Trinity'][2]
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
    v_n_iter = [100, 100, 10]

    # #%% Read parameters
    # df_dist_params = pd.DataFrame(dict(
    #     dist = ['Truncated Normal + Uniform'],
    #     acronym = ['tnXu'],
    #     param_1_name = ['std'],
    #     param_1 = ['2.5'], # 1 for Duwamish, 0.5 for Kanahwa, 0.75 for Trinity
    #     param_2_name = ['w1'],
    #     param_2 = ['0.1'],
    # ))

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

    #%% Get distribution
    _param_dist_xy = fit_rotated_normal_to_polygon(sp_domain.explode().geometry.iloc[0], coverage_factor=0.8)
    row_dist_params = pd.Series(dict(name_file = 'RN_cf_08'))
    dist_xy = RotatedNormal(mean=[v_watershed_stats.x, v_watershed_stats.y], stds=_param_dist_xy.get('stds'), angle_degrees=_param_dist_xy.get('angle_degrees'))
    dist_x = None
    dist_y = None
    
    #%% Perform uncertainty analysis
    for n_sim_is, n_iter in zip(v_n_sim_is, v_n_iter):
        # Monte Carlo for comparison
        df_depths_mc_iter, df_prob_mc_iter, df_aep_mc_iter, df_aep_summary_mc_iter = simulate_sst_iter(sp_watershed, sp_domain, df_storms, dist_x=None, dist_y=None, num_simulations=n_sim_is, num_iter=n_iter)
        df_depths_mc_iter.to_pickle(cwd/'pickle'/f'df_depths_mc_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_prob_mc_iter.to_pickle(cwd/'pickle'/f'df_prob_mc_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_aep_mc_iter.to_pickle(cwd/'pickle'/f'df_aep_mc_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_aep_summary_mc_iter.to_pickle(cwd/'pickle'/f'df_aep_summary_mc_iter_n_{n_sim_is}x{n_iter}.pkl')

        # Importance Sampling
        df_depths_is_iter, df_prob_is_iter, df_aep_is_iter, df_aep_summary_is_iter = simulate_sst_iter(sp_watershed, sp_domain, df_storms, dist_x, dist_y, dist_xy, num_simulations=n_sim_is, num_iter=n_iter)
        df_depths_is_iter.to_pickle(cwd/'pickle'/f'df_depths_is_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_prob_is_iter.to_pickle(cwd/'pickle'/f'df_prob_is_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_aep_is_iter.to_pickle(cwd/'pickle'/f'df_aep_is_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_aep_summary_is_iter.to_pickle(cwd/'pickle'/f'df_aep_summaryisc_iter_n_{n_sim_is}x{n_iter}.pkl')
    
    #%% Read Ground Truth results
    df_depths_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_depths_mc_0.pkl')
    df_prob_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_prob_mc_0.pkl')
    df_aep_mc_0: pd.DataFrame = pd.read_pickle(cwd/'pickle'/'df_aep_mc_0.pkl')
        
    #%% Plot uncertainty analysis results
    for n_sim_is, n_iter in zip(v_n_sim_is, v_n_iter):
        #%% Read uncertainty analysis results
        df_depths_mc_iter = pd.read_pickle(cwd/'pickle'/f'df_depths_mc_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_prob_mc_iter = pd.read_pickle(cwd/'pickle'/f'df_prob_mc_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_aep_mc_iter = pd.read_pickle(cwd/'pickle'/f'df_aep_mc_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_aep_summary_mc_iter = pd.read_pickle(cwd/'pickle'/f'df_aep_summary_mc_iter_n_{n_sim_is}x{n_iter}.pkl')

        df_depths_is_iter = pd.read_pickle(cwd/'pickle'/f'df_depths_is_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_prob_is_iter = pd.read_pickle(cwd/'pickle'/f'df_prob_is_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_aep_is_iter = pd.read_pickle(cwd/'pickle'/f'df_aep_is_iter_n_{n_sim_is}x{n_iter}.pkl')
        df_aep_summary_is_iter = pd.read_pickle(cwd/'pickle'/f'df_aep_summaryisc_iter_n_{n_sim_is}x{n_iter}.pkl')

        # df_aep_summary_mc_iter.rename(columns={'type': 'type_val'}).to_pickle(cwd/'pickle'/f'df_aep_summary_mc_iter_n_{n_sim_is}x{n_iter}.pkl')
        # df_aep_summary_is_iter.rename(columns={'type': 'type_val'}).to_pickle(cwd/'pickle'/f'df_aep_summaryisc_iter_n_{n_sim_is}x{n_iter}.pkl')

        #%% Distribution of sampled points
        g = plot_sample_centers(df_depths_is_iter.loc[lambda _: _.iter == 0], sp_watershed, sp_domain, v_domain_stats)
        # g.show()
        g.save(cwd/'plots'/f'XY iter_n_{n_sim_is}x{n_iter} {row_dist_params.name_file}.png', width=10, height=7)

        #%%
        g = \
        (pn.ggplot(mapping = pn.aes(x = 'return_period', y = 'depth', group = 'type_val', linetype='type_val'))
            + pn.geom_line(data = df_aep_mc_0.assign(type_val = 'mean'), mapping = pn.aes(color='"Truth"'))
            + pn.geom_line(data = df_aep_summary_mc_iter, mapping = pn.aes(color='"MC"'))
            + pn.geom_line(data = df_aep_summary_is_iter, mapping = pn.aes(color='"IS"'))
            + pn.scale_x_log10()
            + pn.labs(
                # x = 'Return Period',
                # x = 'Exceedence Probability',
                x = 'Return Period',
                y = 'Rainfall Depth',
                title = f'Uncertainty (N={n_sim_is}, iter={n_iter})'
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
        g.save(cwd/'plots'/f'Freq u n_{n_sim_is}x{n_iter} {row_dist_params.name_file}.png', width=10, height=7)

        #%%
        df_rmse = \
        pd.DataFrame(dict(
            rmse_mc_mean = [get_aep_rmse(df_aep_mc_0, df_aep_summary_mc_iter.loc[lambda _: _.type_val == 'mean'])],
            rmse_mc_median = [get_aep_rmse(df_aep_mc_0, df_aep_summary_mc_iter.loc[lambda _: _.type_val == 'median'])],
            rmse_is_mean = [get_aep_rmse(df_aep_mc_0, df_aep_summary_is_iter.loc[lambda _: _.type_val == 'mean'])],
            rmse_is_median = [get_aep_rmse(df_aep_mc_0, df_aep_summary_is_iter.loc[lambda _: _.type_val == 'median'])],
        ))
        df_rmse.to_clipboard(index=False)
        # print (f'RMSE MC mean={get_aep_rmse(df_aep_mc_0, df_aep_summary_mc_iter.loc[lambda _: _.type_val == 'mean'])}')
        # print (f'RMSE MC median={get_aep_rmse(df_aep_mc_0, df_aep_summary_mc_iter.loc[lambda _: _.type_val == 'mean'])}')
        # print (f'RMSE MC mean={get_aep_rmse(df_aep_mc_0, df_aep_summary_mc_iter.loc[lambda _: _.type_val == 'mean'])}')
        # print (f'RMSE MC mean={get_aep_rmse(df_aep_mc_0, df_aep_summary_mc_iter.loc[lambda _: _.type_val == 'mean'])}')

#endregion -----------------------------------------------------------------------------------------
