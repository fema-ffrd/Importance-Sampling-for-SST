#region Libraries

#%%
from typing import Literal

import numpy as np
import pandas as pd

import plotnine as pn

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.utils_spatial.spatial_stats import get_sp_stats

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
def downscale_df_xy(df, var_x, var_y, x_res = 0.001, y_res = 0.001) -> pd.DataFrame:
    '''Downsamples data for plotting by keeping only points that are visually distinct based on the resolution of the plot.

    Args:
        df: DataFrame to downscale.
        var_x: Name of the x-column.
        var_y: Name of the y-column.
        x_res: The minimum proportion of maximum horizontal distance to be considered a new point.
        y_res: The minimum proportion of maximum vertical distance to be considered a new point.

    Returns:
        pd.DataFrame: Downscaled dataframe.
    '''
    # Set resolution
    x_range = df[var_x].max() - df[var_x].min()
    y_range = df[var_y].max() - df[var_y].min()
    
    x_res = x_range * x_res 
    y_res = y_range * y_res
    
    last_x, last_y = -np.inf, -np.inf
    keep_indices = []
    
    # Get numpy arrays for performance
    x_vals = df[var_x].values
    y_vals = df[var_y].values
    
    for i in range(len(df)):
        # Check if the point is far enough from the last kept point
        if abs(x_vals[i] - last_x) > x_res or abs(y_vals[i] - last_y) > y_res:
            keep_indices.append(i)
            last_x, last_y = x_vals[i], y_vals[i]
            
    return df.iloc[keep_indices]

#%%
#TODO
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
#TODO
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
#TODO
def plot_xy_vs_depth_2d(df_depths, sp_watershed, sp_domain, stat: Literal['sum', 'mean', 'max'] = 'sum', num_bins=30):
    # Create the bin edges for both x and y axes
    # This ensures our grid covers the entire data range.
    x_bins = np.linspace(df_depths['x_sampled'].min(), df_depths['x_sampled'].max(), num_bins + 1)
    y_bins = np.linspace(df_depths['y_sampled'].min(), df_depths['y_sampled'].max(), num_bins + 1)
    
    # Assign each point to an x and y interval (bin)
    df_depths['x_interval'] = pd.cut(df_depths['x_sampled'], bins=x_bins, include_lowest=True)
    df_depths['y_interval'] = pd.cut(df_depths['y_sampled'], bins=y_bins, include_lowest=True)
    
    # Group by the interval bins and sum the values
    df_summary = \
    (df_depths
        .groupby(['x_interval', 'y_interval'], observed=False)
        .agg(value=('depth', stat))
        .reset_index()
    )

    # ---- CRUCIAL STEP: Calculate numeric properties for plotting ----
    # Get the numeric center of each interval for the x and y coordinates
    df_summary['x_center'] = df_summary['x_interval'].apply(lambda i: i.mid).astype(float)
    df_summary['y_center'] = df_summary['y_interval'].apply(lambda i: i.mid).astype(float)
    
    # Get the width and height of each tile from the interval size
    df_summary['tile_width'] = df_summary['x_interval'].apply(lambda i: i.right - i.left)
    df_summary['tile_height'] = df_summary['y_interval'].apply(lambda i: i.right - i.left)

    df_summary = df_summary.loc[lambda _: _.value > 0]
    
    g = \
    (pn.ggplot(df_summary)  # Start with the summary dataframe
        + pn.geom_tile(pn.aes(
            x='x_center',
            y='y_center',
            fill='value',
        ))
        # + pn.geom_text(pn.aes(
        #     x='x_center',
        #     y='y_center',
        #     label='value',
        # ))
        + pn.geom_polygon(data = sp_watershed.get_coordinates(), mapping=pn.aes('x', 'y'), fill=None, color='red')
        + pn.geom_polygon(data = sp_domain.get_coordinates(), mapping=pn.aes('x', 'y'), fill=None, color='blue')
        + pn.scale_fill_distiller(type="seq", palette="Blues", direction=1, name=f"Total Depth") # direction=1 is light to dark
        + pn.labs(
            title=f"Distribution of {stat} of total depth in watershed",
            x="x samples",
            y="y samples",
            # fill="Sum of Values"   # The legend title
        )
        + pn.theme_bw()
    )
    
    return g

#%%
#TODO
def plot_freq_curve(l_df_prob, l_names, l_colors=None, var_x: Literal['return_period', 'prob_exceed']='return_period', downscale=True, downscale_prop=0.001):
    if downscale:
        for i in range(len(l_df_prob)):
            l_df_prob[i] = downscale_df_xy(l_df_prob[i], var_x, 'depth', x_res=downscale_prop, y_res=downscale_prop)
        
    g = pn.ggplot(mapping=pn.aes(x=var_x, y='depth'))

    for df_prob, name in zip(l_df_prob, l_names):
        print (len(df_prob))
        g = g + pn.geom_point(data=df_prob, mapping=pn.aes(color=f'"{name}"'), size=0.1)
    
    g = \
    (g
        + pn.scale_x_log10()
        + pn.labs(
            # x = 'Return Period',
            # x = 'Exceedence Probability',
            x = 'Return Period' if var_x == 'return_period' else 'Exceedence Probability',
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
