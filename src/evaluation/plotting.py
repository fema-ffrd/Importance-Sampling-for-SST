#region Libraries

#%%
import pandas as pd

import plotnine as pn

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.utils_spatial.spatial_stats import get_sp_stats

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
def plot_sample_centers(df_storm_sample, sp_watershed, sp_domain, v_domain_stats=None):
    if v_domain_stats is None:
        v_domain_stats = get_sp_stats(sp_domain)

    g = \
    (pn.ggplot(df_storm_sample, pn.aes(x='x_sampled', y='y_sampled'))
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
    
    return g

#%%
def plot_xy_vs_depth(df_depths, sp_watershed=None, v_watershed_stats=None):
    if v_watershed_stats is None:
        v_watershed_stats = get_sp_stats(sp_watershed)

    df_x_stats = \
    (df_depths
        .assign(x_sampled_bin = lambda _: pd.cut(_.x_sampled, bins=100))
        .groupby('x_sampled_bin')
        .agg(
            depth_min=('depth', 'min'), 
            depth_max=('depth', 'max'), 
            depth_mean=('depth', 'mean'), 
            depth_median=('depth', 'median')
        )
        .reset_index()
        .assign(x_sampled = lambda _: (_.x_sampled_bin.apply(lambda _x: _x.left).astype(float)+_.x_sampled_bin.apply(lambda _x: _x.right).astype(float))/2)
    )
    g_x = \
    (pn.ggplot(df_x_stats, pn.aes(x = 'x_sampled'))
        + pn.geom_point(pn.aes(y = 'depth_median', color='"median"'))
        + pn.geom_point(pn.aes(y = 'depth_mean', color='"mean"'))
        + pn.geom_point(pn.aes(y = 'depth_min', color='"min"'))
        + pn.geom_point(pn.aes(y = 'depth_max', color='"max"'))
        + pn.geom_vline(pn.aes(xintercept = v_watershed_stats.minx))
        + pn.geom_vline(pn.aes(xintercept = v_watershed_stats.maxx))
        + pn.labs(x = 'x sampled', y = 'depth values')
        + pn.theme_bw()
    )
    
    df_y_stats = \
    (df_depths
        .assign(y_sampled_bin = lambda _: pd.cut(_.y_sampled, bins=100))
        .groupby('y_sampled_bin')
        .agg(
            depth_min=('depth', 'min'), 
            depth_max=('depth', 'max'), 
            depth_mean=('depth', 'mean'), 
            depth_median=('depth', 'median')
        )
        .reset_index()
        .assign(y_sampled = lambda _: (_.y_sampled_bin.apply(lambda _x: _x.left).astype(float)+_.y_sampled_bin.apply(lambda _x: _x.right).astype(float))/2)
    )
    g_y = \
    (pn.ggplot(df_y_stats, pn.aes(x = 'y_sampled'))
        + pn.geom_point(pn.aes(y = 'depth_median', color='"median"'))
        + pn.geom_point(pn.aes(y = 'depth_mean', color='"mean"'))
        + pn.geom_point(pn.aes(y = 'depth_min', color='"min"'))
        + pn.geom_point(pn.aes(y = 'depth_max', color='"max"'))
        + pn.geom_vline(pn.aes(xintercept = v_watershed_stats.miny))
        + pn.geom_vline(pn.aes(xintercept = v_watershed_stats.maxy))
        + pn.labs(x = 'y sampled', y = 'depth values')
        + pn.theme_bw()
    )
    
    return g_x, g_y

#%%
def plot_freq_curve(l_df_depths, l_names, l_colors=None):
    g = pn.ggplot(mapping=pn.aes(x='prob_exceed', y='depth'))

        # + pn.geom_point(data=df_freq_curve_mc_0, mapping=pn.aes(color=f'"MC ({n_sim_mc/1000}k)"'), size=0.1)
        # + pn.geom_point(data=df_freq_curve_mc_1, mapping=pn.aes(color=f'"MC ({n_sim_is/1000}k)"'), size=0.1)
        # + pn.geom_point(data=df_freq_curve_is_1, mapping=pn.aes(color=f'"IS ({n_sim_is/1000}k)"'), size=0.1)

    for df_depths, name in zip(l_df_depths, l_names):
        g = g + pn.geom_point(data=df_depths, mapping=pn.aes(color=f'"{name}"'), size=0.1)
    
    g = \
    (g
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

    return g

#endregion -----------------------------------------------------------------------------------------
