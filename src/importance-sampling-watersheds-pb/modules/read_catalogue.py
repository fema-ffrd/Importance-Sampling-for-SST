#region Modules

#%%
from modules.match_crs import match_crs_to_raster

#endregion -----------------------------------------------------------------------------------------
#region Libraries

#%%
import os
import pathlib

import pandas as pd

import plotnine as pn

from scipy import stats

import geopandas as gpd

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
def read_catalogue(path_sp_watershed: pathlib.Path, path_sp_domain: pathlib.Path, path_storm: pathlib.Path) -> tuple:    
    '''Read watershed, domain, and storm catalogue. The watershed and domain crs are matched to the crs of the storm catalogue.

    Args:
        path_sp_watershed (pathlib.Path): Path of watershed GIS file.
        path_sp_domain (pathlib.Path): Path of domain GIS file.
        path_storm (pathlib.Path): Path of storm catalogue pickel file.

    Returns:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame]: Tuple of watershed geodataframe, domain geodataframe, and storm catalogue dataframe.
    '''
    # Read storm catalogue
    df_storms = pd.read_pickle(path_storm/'catalogue.pkl')
    
    # Read watershed and domain
    sp_watershed = gpd.read_file(path_sp_watershed)
    sp_domain = gpd.read_file(path_sp_domain)
    
    # Match crs of watershed and domain to precipitation raster
    sp_watershed = match_crs_to_raster(sp_watershed, df_storms['path'].iloc[0])
    sp_domain = match_crs_to_raster(sp_domain, df_storms['path'].iloc[0])

    return sp_watershed, sp_domain, df_storms
    
#endregion -----------------------------------------------------------------------------------------
