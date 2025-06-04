#region Libraries

#%%
import pathlib
import platform

import numpy as np
import pandas as pd

import geopandas as gpd

import rasterio

from tqdm import tqdm

#%%
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

#endregion -----------------------------------------------------------------------------------------
#region Modules

#%%
from src.preprocessing.catalog_reader import read_catalog

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
#TODO
def get_raster_stats(path_raster: pathlib.Path, sp_domain: gpd.GeoDataFrame = None) -> dict:
    d_stats = {
        'name': path_raster.stem,
        'sum_val': np.nan,
        'max_val': np.nan,
        'mean_val': np.nan,
        'median_val': np.nan,
        'std_val': np.nan,
        'cv_val': np.nan,
        'non_zero_prop': np.nan,
        'mean_non_zero': np.nan,
        'median_non_zero': np.nan,
    }

    try:
        with rasterio.open(path_raster) as src:
            # Read the first band
            if sp_domain is None:
                data = src.read(1).astype(np.float64) # Ensure float for calculations
            else:
                data = rasterio.mask.mask(src, sp_domain.geometry, crop=True)[0]
            nodata_val = src.nodata

            # # Get raster dimensions
            # rows, cols = src.height, src.width

        # Handle NoData values by converting them to NaN
        if nodata_val is not None:
            data[data == nodata_val] = np.nan

        # Flatten row x columns to vector
        data_flat = data.flatten()

        # Remove NaN values for basic stats (std, cv)
        data_flat_non_na = data_flat[~np.isnan(data_flat)]

        # Non-zero values
        data_flat_non_zero = data_flat_non_na[~(data_flat_non_na == 0)]

        if data_flat_non_na.size < 2: # Need at least 2 valid points for std
            print(f"  Not enough valid data in {path_raster.stem} for basic stats. Skipping.")
            return d_stats

        # Stats
        sum_val = np.nansum(data_flat)

        max_val = np.max(data_flat)

        mean_val = np.nanmean(data_flat) # nanmean ignores NaNs
        median_val = np.median(data_flat)
        std_val = np.nanstd(data_flat) # nanstd ignores NaNs
        cv_val = (std_val/mean_val)*100 if mean_val != 0 and not np.isnan(mean_val) else np.nan # Avoid division by zero or if mean is NaN

        non_zero_prop = 1 - data_flat_non_zero.shape[0] / data_flat_non_na.shape[0]
        mean_val_non_zero = np.nanmean(data_flat_non_zero) # nanmean ignores NaNs
        median_val_non_zero = np.median(data_flat_non_zero)

        d_stats['sum_val'] = sum_val
        d_stats['max_val'] = max_val
        d_stats['mean_val'] = mean_val
        d_stats['median_val'] = median_val
        d_stats['std_val'] = std_val
        d_stats['cv_val'] = cv_val
        d_stats['non_zero_prop'] = non_zero_prop
        d_stats['mean_non_zero'] = mean_val_non_zero
        d_stats['median_non_zero'] = median_val_non_zero

        return d_stats

    except Exception as e:
        print(f"Error processing {path_raster.stem}: {e}")
        return d_stats

#%%
#TODO
def get_storm_stats(path_catalogue: pathlib.Path) -> pd.DataFrame:
    # Read watershed, domain, and storm catalogue
    sp_watershed, sp_domain, df_storms = read_catalog(path_catalogue/'data')

    # Get storm stats
    tif_files = df_storms.path
    pbar = tqdm(total=len(tif_files))
    d_stats = []
    for tif_file in tif_files:
        analysis_result = get_raster_stats(path_catalogue/tif_file, sp_domain)
        d_stats.append(analysis_result)
        pbar.update()
    _df_raster_stats = pd.DataFrame(d_stats)
    
    df_raster_stats = (df_storms.merge(_df_raster_stats, on='name', how='left'))

    return df_raster_stats


#endregion -----------------------------------------------------------------------------------------
