from typing import Literal
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd

from transpose import transpose_gdf
from utils import raster_zonal_mean

def compute_depths(df_storm_sample: pd.DataFrame, sp_watershed: gpd.GeoDataFrame, shift: Literal['watershed', 'storm', 'best'] = 'watershed') -> pd.DataFrame:
    '''Compute storm depths based on storm samples and watershed.

    Args:
        df_storm_sample (pd.DataFrame): Dataframe of storm samples obtained from 'sample_storms'.
        sp_watershed (gpd.GeoDataFrame): Watershed to compute depths in.
        shift (Literal['watershed', 'storm', 'best'], optional): Whether to shift the watershed or the storm. Defaults to 'watershed'.

    Returns:
        pd.DataFrame: Updated 'df_storm_sample' with depths in 'depth' column.
    '''
    tqdm._instances.clear()
    pbar = tqdm(total=df_storm_sample.shape[0])

    _v_depth = []
    for i, row in df_storm_sample.iterrows():
        #TODO implement shift method
        sp_watershed_shifted = transpose_gdf(sp_watershed, -row.x_del, -row.y_del)
        _depth = raster_zonal_mean(row.path, sp_watershed_shifted)
        _v_depth.append(_depth)

        pbar.update(1)

    df_storm_sample = df_storm_sample.assign(depth = _v_depth)

    df_storm_sample = \
    (df_storm_sample
        .assign(intersected = lambda _: np.where(_.depth.isna(), 0, 1))
        .fillna({'depth': 0})
    )

    return df_storm_sample