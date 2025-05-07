#region Libraries

#%%
from typing import Literal

import numpy as np

import geopandas as gpd

import rasterio as rio
import xarray as xr
import rioxarray as rxr # This import enables the .rio accessor on xarray objects
import rasterstats

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%% rasterio (fastest)
def _match_crs_to_raster_rasterio(gdf: gpd.GeoDataFrame, path_raster: str) -> gpd.GeoDataFrame:
    with rio.open(path_raster) as src:
        raster_crs = src.crs
    
        # 3. Ensure CRS match
        if gdf.crs is None:
            print("Warning: GeoJSON has no CRS defined. Assuming it matches raster CRS.")
            # Or raise an error: raise ValueError("GeoJSON CRS is undefined.")
        elif gdf.crs != raster_crs:
            print(f"Reprojecting polygon from {gdf.crs} to {raster_crs}...")
            gdf = gdf.to_crs(raster_crs)
    
#%% xarray
def _match_crs_to_raster_rasterio(gdf: gpd.GeoDataFrame, path_raster: str) -> gpd.GeoDataFrame:
    xr_dataset = rxr.open_rasterio(path_raster, masked=True)
    
    if gdf.crs is None and xr_dataset.rio.crs is not None:
        print(f"Warning: GeoJSON has no CRS. Assuming it matches raster CRS: {xr_dataset.rio.crs}")
        gdf.crs = xr_dataset.rio.crs
    elif gdf.crs != xr_dataset.rio.crs and xr_dataset.rio.crs is not None and gdf.crs is not None:
        print(f"Reprojecting polygon from {gdf.crs} to {xr_dataset.rio.crs} for rioxarray...")
        gdf = gdf.to_crs(xr_dataset.rio.crs)

    return gdf

#%%
def match_crs_to_raster(gdf: gpd.GeoDataFrame, path_raster: str, backend: Literal['rasterio', 'xarray']='rasterio') -> gpd.GeoDataFrame:
    match backend:
        case 'rasterio':
            return _match_crs_to_raster_rasterio(gdf, path_raster)
        case 'xarray':
            return _match_crs_to_raster_rasterio(gdf, path_raster)
        case _:
            pass # Deal with this (throw error? use rasterio? return null values?)            

#%% rasterio
def _sum_raster_values_in_polygon_rasterio(raster_path: str, gdf: gpd.geodataframe) -> float:
    """
    Calculates the sum of raster pixel values that intersect a single polygon
    defined in a GeoJSON file or string.

    Args:
        raster_path (str): Path to the input GeoTIFF raster file.
        geojson_path_or_string (str): Path to the GeoJSON file or a GeoJSON string
                                      containing a single polygon or multipolygon.

    Returns:
        float: The sum of the raster values within the polygon.
               Returns np.nan if no intersection or other issues.
    """
    try:
        # 1. Read the GeoJSON polygon
        if gdf.empty:
            print("Error: GeoJSON is empty or could not be read.")
            return np.nan

        if len(gdf) > 1:
            print("Warning: GeoJSON contains multiple features. Using the first one.")
            # Or, you could iterate or dissolve them:
            # polygon_geometry = gdf.geometry.unary_union # Dissolves into one geometry
            # gdf = gpd.GeoDataFrame(geometry=[polygon_geometry], crs=gdf.crs)
        
        # Ensure we have a valid geometry
        if gdf.geometry.iloc[0] is None or gdf.geometry.iloc[0].is_empty:
            print("Error: The geometry in the GeoJSON is invalid or empty.")
            return np.nan

        # 2. Open the raster file
        with rio.open(raster_path) as src:
            raster_crs = src.crs
            raster_transform = src.transform
            raster_nodata = src.nodata

            # 3. Ensure CRS match
            if gdf.crs is None:
                print("Warning: GeoJSON has no CRS defined. Assuming it matches raster CRS.")
                # Or raise an error: raise ValueError("GeoJSON CRS is undefined.")
            elif gdf.crs != raster_crs:
                print(f"Reprojecting polygon from {gdf.crs} to {raster_crs}...")
                gdf = gdf.to_crs(raster_crs)

            # Get the geometry in a format suitable for rasterio.mask
            # The mask function expects a list of GeoJSON-like geometry dictionaries
            geometries = [gdf.geometry.iloc[0].__geo_interface__] # Take the first geometry

            # 4. Mask the raster data
            # `crop=True` will crop the output array to the extent of the geometries
            # `all_touched=False` (default) includes pixels whose center is within the polygon.
            # `all_touched=True` includes all pixels touched by the geometry.
            # `invert=False` (default) means pixels *inside* the geometry are kept.
            # `nodata` can be specified; if None, it uses src.nodata. If src.nodata is None,
            # it will mask with 0. It's often good to use np.nan if possible for floats.
            # If raster is integer, NaN isn't directly possible, so it will use a fill_value.
            
            out_image, out_transform = rio.mask.mask(
                dataset=src,
                shapes=geometries,
                crop=True,  # Crop to the bounding box of the geometry for efficiency
                all_touched=False, # Or True, depending on your definition of "intersect"
                nodata=np.nan if src.dtypes[0] in ['float32', 'float64'] else raster_nodata
            )
            
            # The output `out_image` is a 3D array (bands, height, width)
            # If single band, it will be (1, height, width). Squeeze it.
            masked_data = out_image.squeeze()

            # If `nodata` was set to np.nan, NaNs are the masked values.
            # If `nodata` was an integer, those are the masked values.
            
            # 5. Sum the unmasked pixel values
            # If using np.nan for masking, np.nansum will sum non-NaN values.
            if src.dtypes[0] in ['float32', 'float64'] or raster_nodata is None and np.isnan(masked_data).any():
                total_sum = np.nansum(masked_data)
            else:
                # If integer type and a specific nodata value was used for masking
                # (or if rasterio used 0 by default for masking an integer array without nodata)
                # We need to exclude this nodata value from the sum.
                # The `mask` function should have set pixels outside the polygon to `nodata`.
                # So, we sum all pixels that are NOT the nodata value used by mask.
                # If `rasterio.mask` used `src.nodata` (or 0 if `src.nodata` was None for int types)
                # for pixels outside the polygon, then:
                
                # The `mask` function's `nodata` parameter sets the fill value for pixels *outside*
                # the mask. If the original raster also has a NoData value, we need to be careful.
                # The `out_image` from `mask` will have `nodata` (or np.nan) for areas outside the polygon.
                # Original NoData values *inside* the polygon will remain.
                
                # Let's refine:
                # Pixels outside the polygon are now `nodata_used_for_masking`.
                # Pixels inside the polygon retain their original values, including original NoData.

                nodata_used_for_masking = np.nan if src.dtypes[0] in ['float32', 'float64'] else \
                                          (raster_nodata if raster_nodata is not None else 0)

                # Create a boolean mask:
                # True for valid pixels (inside polygon AND not original NoData)
                valid_pixels_mask = np.ones(masked_data.shape, dtype=bool)

                if np.isnan(nodata_used_for_masking):
                    valid_pixels_mask &= ~np.isnan(masked_data)
                else:
                    valid_pixels_mask &= (masked_data != nodata_used_for_masking)
                
                # Additionally, if the original raster had a NoData value, exclude those too
                # from the pixels that *were* inside the polygon.
                if raster_nodata is not None and not np.isnan(raster_nodata):
                     valid_pixels_mask &= (masked_data != raster_nodata)

                total_sum = np.sum(masked_data[valid_pixels_mask])


            return float(total_sum)

    except FileNotFoundError:
        print(f"Error: Raster or GeoJSON file not found.")
        return np.nan
    except rio.errors.RasterioIOError as e:
        print(f"Rasterio Error: {e}")
        return np.nan
    except gpd.io.file.DriverError as e:
        print(f"GeoPandas Driver Error (check GeoJSON format/path): {e}")
        return np.nan
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # import traceback
        # traceback.print_exc() # For debugging
        return np.nan

#%% rasterstats
def _sum_raster_values_in_polygon_rasterstats(raster_path: str, gdf: gpd.geodataframe) -> float:
    try:
        if gdf.empty: return np.nan
        
        # rasterstats works with the vector's CRS, it will reproject raster data on the fly
        # if necessary, but it's often more efficient if they match.
        # Let's ensure polygon is in a projected CRS if raster is, or vice-versa for accuracy.
        # For simplicity here, we'll assume they will align or rasterstats handles it.
        # However, for best results and performance, ensure CRS alignment beforehand.
        with rio.open(raster_path) as src:
            affine_transform = src.transform
            if gdf.crs is None and src.crs is not None:
                print(f"Warning: GeoJSON has no CRS. Assuming it matches raster CRS: {src.crs}")
                gdf.crs = src.crs # Tentatively assign
            elif gdf.crs != src.crs and src.crs is not None and gdf.crs is not None:
                print(f"Reprojecting polygon from {gdf.crs} to {src.crs} for rasterstats...")
                gdf = gdf.to_crs(src.crs)

        # 2. Calculate zonal statistics
        # `stats` takes a list of statistics to compute.
        # `nodata` can be specified to ignore certain raster values.
        # `all_touched=False` (default) means pixel center must be in polygon.
        stats_results = rasterstats.zonal_stats(
            vectors=gdf, # Can be GeoDataFrame, path to shapefile, or GeoJSON features
            raster=raster_path,
            stats=["sum"],         # We only need the sum
            # nodata=None,           # Let rasterstats infer from raster or handle NaNs
            #                        # Or specify: e.g., if raster has -9999 as nodata
            # all_touched=False,
            # geojson_out=False      # We just want the stats, not new GeoJSON features
            # affine_transform = src.transform
        )

        if not stats_results or stats_results[0]['sum'] is None:
            # This can happen if no pixels intersect or all intersecting pixels are NoData
            print("No valid sum found by rasterstats (no intersection or all NoData).")
            return 0.0 # Or np.nan, depending on desired behavior for no data
        
        # stats_results is a list of dictionaries, one for each feature in 'vectors'
        # Since we assume one polygon:
        return float(stats_results[0]['sum'])

    except Exception as e:
        print(f"Error using rasterstats: {e}")
        # import traceback
        # traceback.print_exc()
        return np.nan

#%% xarray
def _sum_raster_values_in_polygon_xarray(raster_path: str, gdf: gpd.geodataframe) -> float:
    try:
        if gdf.empty: return np.nan

        # 1. Open raster with rioxarray
        # masked=True converts NoData to NaN
        rds = rxr.open_rasterio(raster_path, masked=True)
        
        # 2. If multi-band, select one. Squeeze if single band was read with band dim.
        if 'band' in rds.dims and rds.sizes['band'] > 1:
            rds = rds.sel(band=1) # Select first band
        elif 'band' in rds.dims:
             rds = rds.squeeze('band', drop=True)

        # 3. Ensure CRS match
        if gdf.crs is None and rds.rio.crs is not None:
            print(f"Warning: GeoJSON has no CRS. Assuming it matches raster CRS: {rds.rio.crs}")
            gdf.crs = rds.rio.crs
        elif gdf.crs != rds.rio.crs and rds.rio.crs is not None and gdf.crs is not None:
            print(f"Reprojecting polygon from {gdf.crs} to {rds.rio.crs} for rioxarray...")
            gdf = gdf.to_crs(rds.rio.crs)

        # 4. Clip/Mask the DataArray
        # `clip` expects a list of GeoJSON-like geometry dictionaries or GeoDataFrame
        # `all_touched` controls pixel inclusion criteria
        # `drop=True` (default) drops data outside of the clip geometry's bounding box
        clipped_da = rds.rio.clip(
            geometries=gdf.geometry, # Pass the GeoSeries
            all_touched=False,
            drop=True,
            # from_disk=True # If using dask for very large files
        )
        
        # `clipped_da` will have NaNs for pixels outside the polygon within the cropped extent.
        # Original NaNs (from NoData) within the polygon will also be NaNs.
        
        # 5. Sum the values
        # np.nansum is appropriate as masked values and original NoData are NaN
        total_sum = np.nansum(clipped_da.data) # .data gets the underlying numpy array

        return float(total_sum)

    except Exception as e:
        # print(f"Error using xarray/rioxarray: {e}")
        # import traceback
        # traceback.print_exc()
        return np.nan

#%%
def sum_raster_values_in_polygon(raster_path: str, gdf: gpd.geodataframe, backend: Literal['rasterio', 'rasterstats' 'xarray']='xarray') -> float:
    match backend:
        case 'rasterio':
            return _sum_raster_values_in_polygon_rasterio(raster_path, gdf)
        case 'rasterstats':
            return _sum_raster_values_in_polygon_rasterstats(raster_path, gdf)
        case 'xarray':
            return _sum_raster_values_in_polygon_xarray(raster_path, gdf)
        case _:
            pass # Deal with this (throw error? use xarray? return null values?)            

#endregion -----------------------------------------------------------------------------------------
