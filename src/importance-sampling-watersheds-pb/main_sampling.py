#region Libraries

#%%
import os
import pathlib

import pandas as pd

import plotnine as pn

from scipy.stats import uniform, truncnorm

import geopandas as gpd

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from modules.compute_raster_stats import match_crs_to_raster
from modules.sample_storms import get_sp_stats, truncnorm_params, sample_storms
from modules.compute_depths import compute_depths, print_sim_stats, get_df_freq_curve
from modules.distributions import TruncatedGeneralizedNormal

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%% Set working folder
    # pls UPDATE this: folder to save outputs
    os.chdir(r'D:\FEMA Innovations\SO3.1\Py\Trinity')

    #%% Set location of storm catalogue (output from main_preprocess_storm_catalogue), watershed GIS file, and domain GIS file
    # pls UPDATE this: name for catalogue folder from main_preprocess_storm_catalogue
    path_storm = pathlib.Path('storm_catalogue_trinity')
    path_sp_watershed = r"D:\FEMA Innovations\SO3.1\Py\Trinity\watershed\trinity.geojson"
    path_sp_domain = r"D:\FEMA Innovations\SO3.1\Py\Trinity\watershed\trinity-transpo-area-v01.geojson"

    #%% Read storm catalogue
    df_storms = pd.read_pickle(path_storm/'catalogue.pkl')
    # df_storms = df_storms.iloc[[0]] # Only choose one storm from the catalogue for constrained analysis

    #%% Read watershed and domain
    sp_watershed = gpd.read_file(path_sp_watershed)
    sp_domain = gpd.read_file(path_sp_domain)

    #%% Match crs of watershed and domain to precipitation raster
    sp_watershed = match_crs_to_raster(sp_watershed, df_storms['path'].iloc[0])
    sp_domain = match_crs_to_raster(sp_domain, df_storms['path'].iloc[0])

    #%% Get polygon info (bounds, centroids, ranges)
    v_watershed_stats = get_sp_stats(sp_watershed)
    v_domain_stats = get_sp_stats(sp_domain)

    #%% Set distribution for x and y
    # pls UPDATE this: distribution details
    # dist_x = uniform(v_domain_stats.minx, v_domain_stats.range_x)
    # dist_y = uniform(v_domain_stats.miny, v_domain_stats.range_y)

    dist_x = truncnorm(**truncnorm_params(v_watershed_stats.x, v_watershed_stats.range_x*1.2, v_domain_stats.minx, v_domain_stats.maxx))
    dist_y = truncnorm(**truncnorm_params(v_watershed_stats.y, v_watershed_stats.range_y*1.2, v_domain_stats.miny, v_domain_stats.maxy))

    #%% Set number of simulations and get storm samples
    # pls UPDATE this: number of simulations for ground truth (n_sim_mc_0) and importance sampling (n_sim_is_1)
    n_sim_mc_0 = 1_000_000
    n_sim_is_1 = 100_000
    df_storm_sample_mc_0 = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_mc_0)
    df_storm_sample_mc_1 = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_is_1)
    # df_storm_sample_is_1 = sample_storms(df_storms, v_domain_stats, dist_x, dist_y, num_simulations=n_sim_is_1)

    df_storm_sample_mc_0.to_pickle('df_storm_sample_mc_0.pkl')
    df_storm_sample_mc_1.to_pickle('df_storm_sample_mc_1.pkl')
    # df_storm_sample_is_1.to_pickle('df_storm_sample_is_1.pkl')

    #%% Run simulations and get depths
    df_depths_mc_0 = compute_depths(df_storm_sample_mc_0, sp_watershed)
    df_depths_mc_1 = compute_depths(df_storm_sample_mc_1, sp_watershed)
    # df_depths_is_1 = compute_depths(df_storm_sample_is_1, sp_watershed)

    df_depths_mc_0.to_pickle('df_depths_mc_0.pkl')
    df_depths_mc_1.to_pickle('df_depths_mc_1.pkl')
    # df_depths_is_1.to_pickle('df_depths_is_1.pkl')



    #%% Truncated Normal Distribution
    # pls UPDATE this: standard deviations for importance sampling with TruncNorm
    n_sim_is = 100_000

    for mult_std in [0.25, 0.5, 0.75, 1, 1.2, 1.5]:
        print (f'Running for mult_std = {mult_std}')
        dist_x = truncnorm(**truncnorm_params(v_watershed_stats.x, v_watershed_stats.range_x*mult_std, v_domain_stats.minx, v_domain_stats.maxx))
        dist_y = truncnorm(**truncnorm_params(v_watershed_stats.y, v_watershed_stats.range_y*mult_std, v_domain_stats.miny, v_domain_stats.maxy))

        df_storm_sample_is = sample_storms(df_storms, v_domain_stats, dist_x, dist_y, num_simulations=n_sim_is)

        df_storm_sample_is.to_pickle(f'df_storm_sample_is_tn_std_{mult_std}.pkl')

        df_depths_is = compute_depths(df_storm_sample_is, sp_watershed)

        df_depths_is.to_pickle(f'df_depths_is_tn_std_{mult_std}.pkl')

    #%% Truncated Generalized Normal Distribution
    # pls UPDATE this: beta values for importance sampling with TruncGeoNorm
    n_sim_is = 100_000

    for beta in [5, 10]:
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

        df_storm_sample_is = sample_storms(df_storms, v_domain_stats, dist_x, dist_y, num_simulations=n_sim_is)

        df_storm_sample_is.to_pickle(f'df_storm_sample_is_tgn_beta_{beta}.pkl')

        df_depths_is = compute_depths(df_storm_sample_is, sp_watershed)

        df_depths_is.to_pickle(f'df_depths_is_tgn_beta_{beta}.pkl')



    #%% Read number of simulations
    n_sim_mc_0 = 1_000_000
    n_sim_is_1 = 100_000

    #%% Read Monte Carlo Results
    df_storm_sample_mc_0 = pd.read_pickle('df_storm_sample_mc_0.pkl')
    df_storm_sample_mc_1 = pd.read_pickle('df_storm_sample_mc_1.pkl')

    df_depths_mc_0 = pd.read_pickle('df_depths_mc_0.pkl')
    df_depths_mc_1 = pd.read_pickle('df_depths_mc_1.pkl')

    #%% Read IS Results
    # pls UPDATE this: which distribution/parameter combo to use
    # choice_dist = 'TruncNorm'
    # choice_param_value = 0.25
    # choice_param_name = 'std'
    choice_dist = 'TruncGenNorm'
    choice_param_value = 5
    choice_param_name = 'beta'

    if choice_dist == 'TruncNorm':
        mult_std = choice_param_value
        df_storm_sample_is_1 = pd.read_pickle(f'df_storm_sample_is_tn_std_{mult_std}.pkl')
        df_depths_is_1 = pd.read_pickle(f'df_depths_is_tn_std_{mult_std}.pkl')
    else:
        beta = choice_param_value
        df_storm_sample_is_1 = pd.read_pickle(f'df_storm_sample_is_tgn_beta_{beta}.pkl')
        df_depths_is_1 = pd.read_pickle(f'df_depths_is_tgn_beta_{beta}.pkl')

    #%% Print some stats about the simulations
    print_sim_stats(df_depths_mc_0)
    print_sim_stats(df_depths_mc_1)
    print_sim_stats(df_depths_is_1)

    #%% Distribution of sampled points
    g = \
    (pn.ggplot(df_storm_sample_is_1, pn.aes(x='x_sampled', y='y_sampled'))
        + pn.geom_bin2d(
            # bins=(20, 20),
            # drop=True by default, which means bins with zero count are not drawn
            # show_legend=True by default for the fill scale
        )
        + pn.geom_polygon(data = sp_watershed.get_coordinates(), mapping=pn.aes('x', 'y'), fill=None, color='red')
        + pn.geom_polygon(data = sp_domain.get_coordinates(), mapping=pn.aes('x', 'y'), fill=None, color='blue')
        + pn.coord_cartesian(
            xlim=(v_domain_stats.minx, v_domain_stats.maxx),
            ylim=(v_domain_stats.miny, v_domain_stats.maxy),
            # expand=False # prevents Plotnine from adding padding around limits
        )
        # + pn.scale_fill_continuous(low="lightblue", high="darkblue", name="Count")
        # + pn.scale_fill_cmap(cmap_name="cividis", name="Count")
        # from plotnine.scales import scale_fill_distiller
        + pn.scale_fill_distiller(type="seq", palette="Greens", direction=1, name="Count") # direction=1 is light to dark
        + pn.labs(
            title=f"Distribution of sampled points",
            x="x samples",
            y="y samples"
        )
        + pn.theme_bw()
    )
    # print(g)
    g.save(f'XY {choice_dist} {choice_param_name}_{choice_param_value}.png', width=10, height=7)

    #%% Get table of frequency curves
    df_freq_curve_mc_0 = get_df_freq_curve(df_depths_mc_0.depth, df_depths_mc_0.prob)
    df_freq_curve_mc_1 = get_df_freq_curve(df_depths_mc_1.depth, df_depths_mc_1.prob)
    df_freq_curve_is_1 = get_df_freq_curve(df_depths_is_1.depth, df_depths_is_1.prob)

    #%% Plot frequency curves
    g = \
    (pn.ggplot(mapping=pn.aes(x='prob_exceed', y='depth'))
        + pn.geom_point(data=df_freq_curve_mc_0, mapping=pn.aes(color=f'"MC ({n_sim_mc_0/1000}k)"'), size=0.1)
        + pn.geom_point(data=df_freq_curve_mc_1, mapping=pn.aes(color=f'"MC ({n_sim_is_1/1000}k)"'), size=0.1)
        + pn.geom_point(data=df_freq_curve_is_1, mapping=pn.aes(color=f'"IS ({n_sim_is_1/1000}k)"'), size=0.1)
        + pn.scale_x_log10()
        + pn.labs(
            x = 'Exceedence Probability',
            y = 'Rainfall Depth',
            title = 'Basic Monte Carlo vs Importance Sampling'
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
    # print(g)
    g.save(f'Freq {choice_dist} {choice_param_name}_{choice_param_value}.png', width=10, height=7)

#endregion -----------------------------------------------------------------------------------------

