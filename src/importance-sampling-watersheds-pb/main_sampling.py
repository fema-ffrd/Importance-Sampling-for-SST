#region Libraries

#%%
import os
import pathlib

from tqdm import tqdm

import pandas as pd

from scipy.stats import uniform, norm, multivariate_normal, truncnorm

import geopandas as gpd

#%%
from modules.compute_raster_stats import match_crs_to_raster
from modules.sample_storms import get_sp_stats, truncnorm_params, sample_storms
from modules.compute_depths import compute_depths, print_sim_stats, get_df_freq_curve

#endregion -----------------------------------------------------------------------------------------
#region Main

#%%
if __name__ == '__main__':
    #%%
    os.chdir(r'D:\FEMA Innovations\SO3.1\Py')

    #%%
    path_storm = pathlib.Path('temp_storm_catalogue')
    path_sp_watershed = r"D:\FEMA Innovations\SO3.1\Py\subham_sampling\example_files\watershed_18020126.geojson"
    path_sp_domain = r"D:\FEMA Innovations\SO3.1\Py\subham_sampling\example_files\domain.geojson"
    
    #%%
    df_storms = pd.read_pickle(path_storm/'catalogue.pkl')
    
    #%%
    sp_watershed = gpd.read_file(path_sp_watershed)
    sp_domain = gpd.read_file(path_sp_domain)
    
    #%%
    sp_watershed = match_crs_to_raster(sp_watershed, df_storms['path'].iloc[0])
    sp_domain = match_crs_to_raster(sp_domain, df_storms['path'].iloc[0])
    
    #%%
    v_watershed_stats = get_sp_stats(sp_watershed)
    v_domain_stats = get_sp_stats(sp_domain)
    
    #%%
    dist_x = truncnorm(**truncnorm_params(v_watershed_stats.x, v_watershed_stats.range_x*1.2, v_domain_stats.minx, v_domain_stats.maxx))
    dist_y = truncnorm(**truncnorm_params(v_watershed_stats.y, v_watershed_stats.range_y*1.2, v_domain_stats.miny, v_domain_stats.maxy))
    
    #%%
    n_sim_mc_0 = 1000
    n_sim_is_1 = 100
    df_storm_sample_mc_0 = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_mc_0)
    df_storm_sample_is_1 = sample_storms(df_storms, v_domain_stats, dist_x=None, dist_y=None, num_simulations=n_sim_is_1)
    # df_storm_sample_is_1 = sample_storms(df_storms, v_domain_stats, dist_x, dist_y, num_simulations=n_sim_is_1)

    #%%
    df_depths_mc_0 = compute_depths(df_storm_sample_mc_0, sp_watershed)
    df_depths_is_1 = compute_depths(df_storm_sample_is_1, sp_watershed)

    #%%
    print_sim_stats(df_depths_mc_0)
    print_sim_stats(df_depths_is_1)

    #%%
    df_freq_curve_mc_0 = get_df_freq_curve(df_depths_mc_0.depth, df_depths_mc_0.prob)
    df_freq_curve_is_1 = get_df_freq_curve(df_depths_is_1.depth, df_depths_mc_0.prob*10)

    #%% Plot frequency curves
    import plotnine as pn

    (pn.ggplot(mapping=pn.aes(x='prob_exceed', y='depth'))
        + pn.geom_point(data=df_freq_curve_mc_0, mapping=pn.aes(color=f'"MC ({n_sim_mc_0/1000}k)"'), size=0.1)
        # + pn.geom_point(data=df_prob_mc_1, mapping=pn.aes(color=f'"MC ({n_sim_mc_1/1000}k)"'), size=0.1)
        + pn.geom_point(data=df_freq_curve_is_1, mapping=pn.aes(color=f'"IS ({n_sim_is_1/1000}k)"'), size=0.1)
        + pn.scale_x_log10()
        + pn.labs(
            x = 'Return Period',
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

#endregion -----------------------------------------------------------------------------------------
