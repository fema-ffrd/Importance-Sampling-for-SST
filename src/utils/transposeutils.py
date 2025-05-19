import geopandas as gpd
import pandas as pd

def get_sp_stats(gdf: gpd.GeoDataFrame) -> pd.Series:
    
    gdf_proj = gdf.to_crs("EPSG:3857")
    area_km2 = gdf_proj.area.iloc[0] / 1e6

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
    v_sp_stats["area_km2"] = area_km2

    return v_sp_stats    

# Function to pass parameters to truncnorm
def truncnorm_params(mean, std_dev, lower, upper):
    d = dict(
        a = (lower - mean) / std_dev, 
        b = (upper - mean) / std_dev, 
        loc = mean, 
        scale = std_dev
    )

    return d    