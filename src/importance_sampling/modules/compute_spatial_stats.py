#region Libraries

#%%
import pandas as pd

import geopandas as gpd

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
def get_sp_stats(gdf: gpd.GeoDataFrame) -> pd.Series:
    '''Get polygon info (bounds ("minx", "minxy", "maxx", "maxy"), centroid ("x", "y"), and range ("range_x", "range_y"))

    Args:
        gdf (gpd.GeoDataFrame): Geodataframe with polygon.

    Returns:
        pd.Series: Series with info.
    '''
    v_sp_stats = \
    (pd.concat(
        [
            gdf.bounds,
            gdf.centroid.get_coordinates()
        ],
        axis=1,
    )
        .assign(range_x = lambda _: _.maxx - _.minx)
        .assign(range_y = lambda _: _.maxy - _.miny)
        .iloc[0]
    )

    return v_sp_stats

#endregion -----------------------------------------------------------------------------------------
