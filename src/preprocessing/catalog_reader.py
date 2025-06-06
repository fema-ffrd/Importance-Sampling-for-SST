#region Libraries

#%%
import os
import pathlib

import pandas as pd

import plotnine as pn

from scipy import stats

import geopandas as gpd

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.utils_spatial.crs_converter import match_crs_to_raster

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
def read_catalog(folder_watershed: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]:
    '''Read watershed, domain, and storm catalogue. The watershed and domain crs are matched to the crs of the storm catalogue.

    Args:
        folder_watershed (str): Watershed folder.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]: Tuple of watershed geodataframe, domain geodataframe, and storm catalogue dataframe.
    '''
    path_watershed = pathlib.Path(folder_watershed)
    path_data = path_watershed/'data'

    # Set paths
    path_storm = path_data/'storm_catalog'
    path_geojson = path_data/'geojson'
    path_sp_watershed = list(path_geojson.glob('BASIN_*'))[0]
    path_sp_domain = list(path_geojson.glob('DOMAIN_*'))[0]

    # Read storm catalogue
    df_storms = pd.read_pickle(path_storm/'catalog.pkl')
    
    # Read watershed and domain
    sp_watershed = gpd.read_file(path_sp_watershed)
    sp_domain = gpd.read_file(path_sp_domain)
    
    # Match crs of watershed and domain to precipitation raster
    sp_watershed = match_crs_to_raster(sp_watershed, df_storms['path'].iloc[0])
    sp_domain = match_crs_to_raster(sp_domain, df_storms['path'].iloc[0])

    return sp_watershed, sp_domain, df_storms

#endregion -----------------------------------------------------------------------------------------
