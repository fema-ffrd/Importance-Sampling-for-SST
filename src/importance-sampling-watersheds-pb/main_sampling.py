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

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%% Set working folder
    os.chdir(r'D:\FEMA Innovations\SO3.1\Py')

    #%% Set location of storm catalogue (output from main_preprocess_storm_catalogue), watershed GIS file, and domain GIS file
    path_storm = pathlib.Path('temp_storm_catalogue_3')
    path_sp_watershed = r"D:\FEMA Innovations\SO3.1\Py\subham_sampling\example_files\watershed_18020126.geojson"
    path_sp_domain = r"D:\FEMA Innovations\SO3.1\Py\subham_sampling\example_files\domain.geojson"
    
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
    dist_x = uniform(v_domain_stats.minx, v_domain_stats.range_x)
    dist_y = uniform(v_domain_stats.miny, v_domain_stats.range_y)
    
    dist_x = truncnorm(**truncnorm_params(v_watershed_stats.x, v_watershed_stats.range_x*1.2, v_domain_stats.minx, v_domain_stats.maxx))
    dist_y = truncnorm(**truncnorm_params(v_watershed_stats.y, v_watershed_stats.range_y*1.2, v_domain_stats.miny, v_domain_stats.maxy))
    
    #%% Set number of simulations and get storm samples
    n_sim_mc_0 = 10_000
    n_sim_is_1 = 1_000
    df_storm_sample_mc_0 = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_mc_0)
    df_storm_sample_mc_1 = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_is_1)
    df_storm_sample_is_1 = sample_storms(df_storms, v_domain_stats, dist_x, dist_y, num_simulations=n_sim_is_1)

    #%% Run simulations and get depths
    df_depths_mc_0 = compute_depths(df_storm_sample_mc_0, sp_watershed)
    df_depths_mc_1 = compute_depths(df_storm_sample_mc_1, sp_watershed)
    df_depths_is_1 = compute_depths(df_storm_sample_is_1, sp_watershed)

    #%% Print some stats about the simulations
    print_sim_stats(df_depths_mc_0)
    print_sim_stats(df_depths_mc_1)
    print_sim_stats(df_depths_is_1)

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
    print(g)

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
    print(g)

#endregion -----------------------------------------------------------------------------------------
#region Tests

# #%%
# #%%
# (pn.ggplot(df_depths_is_1, mapping=pn.aes(x='depth', y='prob'))
#     + pn.geom_point(size=0.1)
# )

# #%%
# (pn.ggplot(df_freq_curve_is_1, mapping=pn.aes(x='depth', y='prob_exceed'))
#     + pn.geom_point(size=0.1)
# )

# #%%
# # df_prob = df_storm_sample_mc_0.copy()    
# df_prob = df_storm_sample_is_1.copy()    
# # df_prob = df_freq_curve_mc_1.copy()    
# # df_prob = df_freq_curve_is_1.copy()    
# df_prob = \
# (df_prob
#     .assign(depth_bin = lambda _: pd.cut(_.y_sampled, bins = 100))
#     .groupby('depth_bin')
#     .agg(prob_count = ('prob', 'size'),
#          prob_mean = ('prob', 'mean'),
#          prob_sum = ('prob', 'sum'))
#     .reset_index()
#     .assign(depth = lambda _: (_.depth_bin.apply(lambda _x: _x.left).astype(float)+_.depth_bin.apply(lambda _x: _x.right).astype(float))/2)
# )
# (pn.ggplot(mapping=pn.aes(x='depth'))
#     # + pn.geom_point(data=df_prob_mc_0, mapping=pn.aes(color=f'"MC ({n_sim_mc_0/1000}k)"'), size=0.1)
#     # + pn.geom_point(data=df_prob_mc_1, mapping=pn.aes(color=f'"MC ({n_sim_mc_1/1000}k)"'), size=0.1)
#     # + pn.geom_point(data=df_prob, mapping=pn.aes(y='prob_count', color=f'"IS ({n_sim_is_1/1000}k), count"'), size=0.1)
#     + pn.geom_point(data=df_prob, mapping=pn.aes(y='prob_sum', color=f'"IS ({n_sim_is_1/1000}k), sum"'), size=0.1)
#     # + pn.geom_point(data=df_prob, mapping=pn.aes(y='prob_mean', color=f'"IS ({n_sim_is_1/1000}k), mean"'), size=0.1)
#     # + pn.scale_x_log10()
#     + pn.labs(
#         x = 'value',
#         y = 'Probability',
#         title = 'Basic Monte Carlo vs Importance Sampling'
#     )
#     + pn.theme_bw()
#     + pn.theme(
#         title = pn.element_text(hjust = 0.5),
#         # legend_position = 'bottom',
#         legend_title = pn.element_blank(),
#         legend_key = pn.element_blank(),
#         axis_title_y = pn.element_text(ha = 'left'),
#     )
# )

#endregion -----------------------------------------------------------------------------------------
