#region Modules

#%%
from process_files_folders import get_files_pathlib

#endregion -----------------------------------------------------------------------------------------
#region Libraries

#%%
import pathlib

from typing import Literal

import numpy as np
import pandas as pd

import pyproj

import rasterio as rio
import xarray as xr
import rioxarray as rxr # This import enables the .rio accessor on xarray objects

#endregion -----------------------------------------------------------------------------------------
#region Functions

#%%
def sum_netcdf_to_tif(
        netcdf_filepath: str,
        variable_name='band_data',
        stack_dimension_name='time',
        output_tif_filepath='nc_sum.tif',
        output_nodata: float=None) -> None:
    '''
    Sums raster layers from a NetCDF file along a specified dimension and saves
    the result as a GeoTIFF. Null values (NaNs) are treated as zero in the sum.

    Args:
        netcdf_filepath (str): Path to the input NetCDF file.
        variable_name (str): Name of the variable in the NetCDF to process.
        stack_dimension_name (str): Name of the dimension along which to sum
                                    (e.g., 'time', 'band', 'level').
        output_tif_filepath (str): Path to save the output GeoTIFF file.
        output_nodata (float, optional): NoData value to use in the output GeoTIFF.
                                        If None, rioxarray will try to infer or use
                                        a default. Common choices are np.nan or -9999.0.
                                        Note: NaNs in the input are treated as 0 for summation.
                                        This sets the NoData flag in the output TIF.
    '''
    try:
        print(f"Opening NetCDF file: {netcdf_filepath} with chunks='auto'")
        # Use chunks='auto' for dask-backed arrays, efficient for large files
        xr_dataset = xr.open_dataset(netcdf_filepath, chunks="auto")

        _crs = pyproj.CRS(xr_dataset['spatial_ref'].crs_wkt).to_epsg()
        xr_dataset = xr_dataset.rio.write_crs(_crs)

        if variable_name not in xr_dataset.data_vars:
            available_vars = list(xr_dataset.data_vars.keys())
            raise ValueError(
                f"Variable '{variable_name}' not found in NetCDF.\n"
                f"Available data variables are: {available_vars}"
            )

        data_array = xr_dataset[variable_name]

        if stack_dimension_name not in data_array.dims:
            available_dims = list(data_array.dims)
            raise ValueError(
                f"Stack dimension '{stack_dimension_name}' not found for variable '{variable_name}'.\n"
                f"Available dimensions are: {available_dims}"
            )

        print(f"Summing data variable '{variable_name}' along dimension '{stack_dimension_name}'...")
        # skipna=True: NaNs will be treated as 0 in the sum.
        # If all values along the stack_dimension are NaN for a pixel, the sum will be 0.
        summed_data = data_array.sum(dim=stack_dimension_name, skipna=True, dtype=np.float32) # Ensure float output

        # At this point, summed_data is an xarray.DataArray.
        # rioxarray will attempt to use existing CRS and transform information.
        # If your NetCDF is not CF-compliant or lacks explicit CRS, you might need to set it.
        # Example:
        # if summed_data.rio.crs is None:
        #     print("CRS not found, attempting to set to EPSG:4326 (WGS84). Adjust if needed.")
        #     # Guess common coordinate names, adjust if your file uses different ones
        #     if 'longitude' in summed_data.coords and 'latitude' in summed_data.coords:
        #         summed_data = summed_data.rio.set_spatial_dims(x_dim='longitude', y_dim='latitude')
        #     elif 'lon' in summed_data.coords and 'lat' in summed_data.coords:
        #         summed_data = summed_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
        #     elif 'x' in summed_data.coords and 'y' in summed_data.coords:
        #          summed_data = summed_data.rio.set_spatial_dims(x_dim='x', y_dim='y')
        #     else:
        #         print("Warning: Could not automatically determine spatial dimension names for CRS setting.")
        #     summed_data = summed_data.rio.write_crs("EPSG:4326", inplace=True)

        print(f"Writing summed raster to GeoTIFF: {output_tif_filepath}")
        # Use rio.to_raster for writing. It handles georeferencing.
        # Tiled and LZW compression are generally good for performance and size.
        summed_data.rio.to_raster(
            output_tif_filepath,
            driver="GTiff",
            nodata=output_nodata,
            tiled=True,
            compress='LZW',
            windowed=True # Good for dask arrays if they are large
        )

        print(f"Successfully created {output_tif_filepath}")

    except FileNotFoundError:
        print(f"Error: Input NetCDF file not found at {netcdf_filepath}")
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'ds' in locals() and xr_dataset is not None:
            xr_dataset.close()
            print("Closed NetCDF file.")

#%%
def _calculate_weighted_raster_centroid_rasterio(raster_path: str) -> tuple:
    try:
        with rio.open(raster_path) as src:
            # Read the raster data as a NumPy array
            # Assuming a single-band raster, or you want to use the first band
            data = src.read(1).astype(np.float64)  # Use float64 for precision in sums

            # Get geotransform (affine transformation)
            # transform[0] = top-left x
            # transform[1] = w-e pixel resolution / pixel width
            # transform[2] = rotation, 0 if image is "north up"
            # transform[3] = top-left y
            # transform[4] = rotation, 0 if image is "north up"
            # transform[5] = n-s pixel resolution / pixel height (negative)
            transform = src.transform

            # Get NoData value if it exists
            nodata_value = src.nodata
            if nodata_value is not None:
                # Create a mask for valid data (not NoData)
                valid_data_mask = (data != nodata_value)
                # Set NoData values to 0 so they don't contribute to the weighted sum
                # and their weight becomes 0.
                data_for_weights = np.where(valid_data_mask, data, 0)
            else:
                # If no NoData value, all data is considered valid for weighting
                valid_data_mask = np.ones_like(data, dtype=bool)
                data_for_weights = data.copy() # Use a copy

            # Calculate sum of weights (raster values)
            # Only sum valid data if nodata was present and masked
            sum_of_weights = np.sum(data_for_weights[valid_data_mask])

            if sum_of_weights == 0:
                print("Warning: Sum of raster values (weights) is zero. Cannot calculate weighted centroid.")
                # Optionally, return the geometric center instead, or handle as an error
                # Geometric center:
                # center_x = transform.c + (src.width / 2.0) * transform.a
                # center_y = transform.f + (src.height / 2.0) * transform.e
                # return center_x, center_y
                return None, None

            # Create coordinate grids for x and y
            # Pixel indices
            rows, cols = np.indices(data.shape)  # (0,0), (0,1)... (1,0), (1,1)...

            # Convert pixel indices to geographic coordinates (center of pixel)
            # x = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
            # y = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
            # Assuming no rotation (transform.b and transform.d are 0)
            # which is common for GeoTIFFs.
            # If rotation exists, the formula is more complex, but rasterio's transform handles it.

            # Get arrays of x and y coordinates for each pixel center
            # This uses the affine transform correctly for all pixels
            x_coords, y_coords = rio.transform.xy(transform, rows, cols, offset='center')
            x_coords = np.array(x_coords) # Ensure it's a numpy array
            y_coords = np.array(y_coords) # Ensure it's a numpy array


            # Calculate weighted sum for x and y coordinates
            # Only consider valid data points for these sums as well
            weighted_x_sum = np.sum(x_coords[valid_data_mask] * data_for_weights[valid_data_mask])
            weighted_y_sum = np.sum(y_coords[valid_data_mask] * data_for_weights[valid_data_mask])

            # Calculate centroid coordinates
            centroid_x = weighted_x_sum / sum_of_weights
            centroid_y = weighted_y_sum / sum_of_weights

            print(f"Raster CRS: {src.crs}")
            return centroid_x, centroid_y, sum_of_weights

    except rio.errors.RasterioIOError as e:
        print(f"Error opening or reading raster file: {e}")
        return None, None, None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None, None, None

# xarray (fastest)
def _calculate_weighted_raster_centroid_xarray(raster_path: str) -> tuple:
    try:
        # Open the raster using rioxarray, which returns an xarray.DataArray
        # masked=True will convert NoData values to NaN
        rds = rxr.open_rasterio(raster_path, masked=True, default_name="raster_values")

        # If it's a multi-band raster, select the first band or specify
        # For this example, assume single band or use the first.
        # .squeeze() removes dimensions of size 1 (like 'band' if only one band)
        data_array = rds.squeeze(drop=True) # drop=True removes the coord too

        # Ensure data_array is 2D (y, x)
        if data_array.ndim > 2:
            # This might happen if squeeze didn't remove all extra dims,
            # or if you need to select a specific band from a multi-band array
            # For simplicity, let's assume we want the first item along other dims
            # A more robust solution would be specific band selection if needed.
            print(f"Warning: Data array has more than 2 dimensions ({data_array.dims}). Attempting to use first slice.")
            # Example: if dims are ('band', 'y', 'x'), take first band
            if 'band' in data_array.dims and data_array.sizes['band'] > 1:
                data_array = data_array.isel(band=0) # Select first band
            else: # try to select first element along any extra dimension
                for dim_name in data_array.dims:
                    if dim_name not in ('y', 'x', data_array.rio.x_dim, data_array.rio.y_dim):
                        data_array = data_array.isel({dim_name: 0})

            if data_array.ndim > 2: # Check again
                raise ValueError(f"Could not reduce DataArray to 2D. Dimensions: {data_array.dims}")


        # Convert data to float64 for precision in sums
        data_array = data_array.astype(np.float64)

        # Handle NaNs (originally NoData values): treat them as 0 for weighting.
        # .fillna(0) replaces NaNs with 0. This resulting array contains the weights.
        weights = data_array.fillna(0)

        # Calculate sum of weights
        sum_of_weights = weights.sum()

        if sum_of_weights.item() == 0: # .item() to get scalar value from 0-dim xarray DataArray
            print("Warning: Sum of raster values (weights) is zero. Cannot calculate weighted centroid.")
            return None, None

        # xarray DataArrays have coordinate arrays (e.g., data_array.x, data_array.y)
        # These are 1D coordinate arrays. We need 2D arrays matching the data shape.
        # We can achieve this by broadcasting or by using xr.broadcast.
        # Alternatively, and more simply, xarray allows direct multiplication:

        # Weighted sum for x and y coordinates
        # (weights * data_array.x) will broadcast data_array.x across the y dimension
        # then sum over both x and y dimensions.
        weighted_x_sum = (weights * data_array[data_array.rio.x_dim]).sum()
        weighted_y_sum = (weights * data_array[data_array.rio.y_dim]).sum()

        # Calculate centroid coordinates
        centroid_x = (weighted_x_sum / sum_of_weights).item()
        centroid_y = (weighted_y_sum / sum_of_weights).item()

        print(f"Raster CRS (from xarray): {data_array.rio.crs}")
        return centroid_x, centroid_y, sum_of_weights.item()

    except Exception as e: # More general exception for xarray/rioxarray specific issues
        print(f"An unexpected error occurred with xarray: {e}")
        return None, None, None

#%%
def calculate_weighted_raster_centroid(raster_path: str, backend: Literal['rasterio', 'xarray']='xarray') -> tuple:
    '''
    Calculates the centroid of a raster weighted by its cell values. Also, provides the total of all cell values.

    Args:
        raster_path (str): Path to the input GeoTIFF raster file.
        backend (Literal[&#39;rasterio&#39;, &#39;xarray&#39;], optional): Backend to use. Defaults to 'xarray'.

    Returns:
        tuple: (centroid_x, centroid_y) in the raster's CRS, or (None, None) if the sum of weights is zero (e.g., all NoData or all zeros).
    '''
    match backend:
        case 'rasterio':
            return _calculate_weighted_raster_centroid_rasterio(raster_path)
        case 'xarray':
            return _calculate_weighted_raster_centroid_xarray(raster_path)
        case _:
            pass # Deal with this (throw error? use xarray? return null values?)

#%%
def preprocess_storm_catalogue(folder_storms, nc_data_name='APCP_surface', path_output='storm_catalogue') -> None:
    '''Preprocess storm data. This creates a tif of accumulated rasters and a pickle file with their centroids within 'path_output'.

    Args:
        folder_storms (str): Folder path containing the storm nc files.
        nc_data_name (str, optional): Data name within the nc file. Defaults to 'APCP_surface'.
        path_output (str, optional): Output folder path or folder name to create in current working directory. Defaults to 'storm_catalogue' (in current working directory).
    '''
    # Get list of nc files
    v_file_storm = get_files_pathlib(folder_storms)

    # Create accumulated raster
    path_storm = pathlib.Path(path_output)
    if not path_storm.exists():
        path_storm.mkdir()
    if not (path_storm/'tifs').exists():
        (path_storm/'tifs').mkdir()

    for _file_storm in v_file_storm:
        sum_netcdf_to_tif(_file_storm, variable_name=nc_data_name, output_tif_filepath=path_storm/f'tifs/{_file_storm.stem}.tif')

    # Calculate storm centroids
    df_storms = pd.DataFrame()
    for _f in (path_storm/'tifs').glob('*.tif'):
        _centroid = calculate_weighted_raster_centroid(_f)

        _df_storms = pd.DataFrame(dict(
            name = [_f.stem],
            path = [_f],
            x = [_centroid[0]],
            y = [_centroid[1]],
            total = [_centroid[2]]
        ))

        df_storms = pd.concat([df_storms, _df_storms], ignore_index=True)

    # Save pkl file
    df_storms.to_pickle(path_storm/'catalogue.pkl')

#endregion -----------------------------------------------------------------------------------------
