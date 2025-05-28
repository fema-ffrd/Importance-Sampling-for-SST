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

def transpose_storms(
    df_storms: pd.DataFrame,
    v_domain_stats: pd.Series,
    num_simulations: int,
    method: str = "uniform",  # "uniform" or "truncnorm"
    dist_x=None,
    dist_y=None
) -> pd.DataFrame:

    if method == "truncnorm":
        if dist_x is None or dist_y is None:
            raise ValueError("dist_x and dist_y must be provided for truncated normal sampling.")
        df_centers = sample_truncated_normal_centers(v_domain_stats, dist_x, dist_y, num_simulations)
    elif method == "uniform":
        df_centers = sample_uniform_centers(v_domain_stats, num_simulations)
    else:
        raise ValueError("method must be either 'uniform' or 'truncnorm'.")

    df_transposed = pd.concat([df_storms.reset_index(drop=True), df_centers], axis=1)
    df_transposed['x_del'] = df_transposed['x_sampled'] - df_transposed['x']
    df_transposed['y_del'] = df_transposed['y_sampled'] - df_transposed['y']
    
    return df_transposed
