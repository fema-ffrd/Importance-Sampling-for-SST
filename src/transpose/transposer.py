import geopandas as gpd
import pandas as pd

def transpose_gdf(gdf: gpd.GeoDataFrame, x_del: float, y_del: float) -> gpd.GeoDataFrame:
    '''Shift a geodataframe in x and y directions.

    Args:
        gdf (gpd.GeoDataFrame): Geodataframe to shift.
        x_del (float): Shift in x direction.
        y_del (float): Shift in y direction.

    Returns:
        gpd.GeoDataFrame: Shifted geodataframe.
    '''
    gdf_shifted = gpd.GeoDataFrame(geometry=gdf.geometry.translate(-x_del, -y_del))

    return gdf_shifted

def transpose_storm(xr_dataset, x_del: float, y_del: float):
    '''Shift the storm in x and y directions.

    '''
    xr_dataset = xr_dataset.copy()
    xr_dataset_shifted = xr_dataset.assign_coords({
        "x": xr_dataset.x + x_del,
        "y": xr_dataset.y + y_del
    })
    return xr_dataset_shifted