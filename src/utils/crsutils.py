from typing import Literal
import rasterio as rio
import rioxarray as rxr
import geopandas as gpd

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

        return gdf

def _match_crs_to_raster_xarray(gdf: gpd.GeoDataFrame, path_raster: str) -> gpd.GeoDataFrame:
    xr_dataset = rxr.open_rasterio(path_raster, masked=True)
    
    if gdf.crs is None and xr_dataset.rio.crs is not None:
        print(f"Warning: GeoJSON has no CRS. Assuming it matches raster CRS: {xr_dataset.rio.crs}")
        gdf.crs = xr_dataset.rio.crs
    elif gdf.crs != xr_dataset.rio.crs and xr_dataset.rio.crs is not None and gdf.crs is not None:
        print(f"Reprojecting polygon from {gdf.crs} to {xr_dataset.rio.crs} for rioxarray...")
        gdf = gdf.to_crs(xr_dataset.rio.crs)

    return gdf

def match_crs_to_raster(gdf: gpd.GeoDataFrame, path_raster: str, backend: Literal['rasterio', 'xarray']='rasterio') -> gpd.GeoDataFrame:
    match backend:
        case 'rasterio':
            return _match_crs_to_raster_rasterio(gdf, path_raster)
        case 'xarray':
            return _match_crs_to_raster_xarray(gdf, path_raster)
        case _:
            pass # Deal with this (throw error? use rasterio? return null values?) 