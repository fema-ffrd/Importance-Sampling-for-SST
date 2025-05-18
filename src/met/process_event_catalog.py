from typing import Literal
import pathlib
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import xarray as xr
import rioxarray as rxr  # This import enables the .rio accessor on xarray objects

from utils import get_files_pathlib


class EventCatalogProcessor:
    """
    A processor for storm event NetCDF files that:
    - Sums storm precipitation fields over time and saves as GeoTIFFs
    - Calculates weighted centroids of resulting rasters
    - Saves a catalog (as a pickle) containing filenames and centroid coordinates

    Attributes:
        folder_storms (Path): Path to folder containing NetCDF storm files.
        nc_data_name (str): Variable name to extract from each NetCDF file.
        output_dir (Path): Directory to save output TIFs and catalog.
        centroid_backend (str): Method used for centroid calculation ('xarray' or 'rasterio').
    """
    def __init__(
        self,
        folder_storms: str,
        nc_data_name: str = "APCP_surface",
        output_dir: str = "storm_catalog",
        centroid_backend: Literal["xarray", "rasterio"] = "xarray"
    ):
        self.folder_storms = pathlib.Path(folder_storms)
        self.nc_data_name = nc_data_name
        self.output_dir = pathlib.Path(output_dir)
        self.centroid_backend = centroid_backend
        self.output_tif_dir = self.output_dir / "tifs"

    def run(self):
        self._create_output_folders()
        self._convert_nc_to_tif()
        df_catalog = self._calculate_centroids()
        self._save_catalog(df_catalog)

    def _create_output_folders(self):
        self.output_dir.mkdir(exist_ok=True)
        self.output_tif_dir.mkdir(exist_ok=True)

    def _convert_nc_to_tif(self):
        files = get_files_pathlib(self.folder_storms, extension="nc")
        for nc_file in files:
            output_path = self.output_tif_dir / f"{nc_file.stem}.tif"
            self._sum_netcdf_to_tif(nc_file, output_path)

    def _sum_netcdf_to_tif(self, nc_path, tif_path):
        try:
            print(f"Opening NetCDF file: {nc_path}")
            ds = xr.open_dataset(nc_path, chunks="auto")

            crs = pyproj.CRS(ds["spatial_ref"].crs_wkt).to_epsg()
            ds = ds.rio.write_crs(crs)

            if self.nc_data_name not in ds:
                raise ValueError(f"Variable '{self.nc_data_name}' not found. Available: {list(ds.data_vars)}")

            arr = ds[self.nc_data_name]

            if "time" not in arr.dims:
                raise ValueError(f"Dimension 'time' not found in {arr.dims}")

            print(f"Summing '{self.nc_data_name}' over 'time'...")
            summed = arr.sum(dim="time", skipna=True, dtype=np.float32)

            print(f"Saving GeoTIFF to: {tif_path}")
            summed.rio.to_raster(
                tif_path,
                driver="GTiff",
                nodata=None,
                tiled=True,
                compress="LZW",
                windowed=True
            )
        except Exception as e:
            print(f"Failed to process {nc_path.name}: {e}")
        finally:
            ds.close()
            print("Closed NetCDF.")

    def _calculate_centroids(self) -> pd.DataFrame:
        tif_files = list(self.output_tif_dir.glob("*.tif"))
        records = []

        for tif in tif_files:
            x, y = self._calculate_weighted_raster_centroid(tif)
            records.append({
                "name": tif.stem,
                "path": tif,
                "x": x,
                "y": y
            })

        return pd.DataFrame(records)

    def _calculate_weighted_raster_centroid(self, raster_path):
        match self.centroid_backend:
            case "rasterio":
                return self._centroid_rasterio(raster_path)
            case "xarray":
                return self._centroid_xarray(raster_path)
            case _:
                raise ValueError("Unsupported backend")

    def _centroid_rasterio(self, raster_path):
        try:
            with rio.open(raster_path) as src:
                data = src.read(1).astype(np.float64)
                nodata = src.nodata

                if nodata is not None:
                    mask = (data != nodata)
                    weights = np.where(mask, data, 0)
                else:
                    weights = data
                    mask = np.ones_like(data, dtype=bool)

                total_weight = np.sum(weights[mask])
                if total_weight == 0:
                    print("Zero-weight raster")
                    return None, None

                rows, cols = np.indices(data.shape)
                x_coords, y_coords = rio.transform.xy(src.transform, rows, cols, offset="center")
                x_coords = np.array(x_coords)
                y_coords = np.array(y_coords)

                x = np.sum(x_coords[mask] * weights[mask]) / total_weight
                y = np.sum(y_coords[mask] * weights[mask]) / total_weight

                return x, y
        except Exception as e:
            print(f"Error processing {raster_path}: {e}")
            return None, None

    def _centroid_xarray(self, raster_path):
        try:
            rds = rxr.open_rasterio(raster_path, masked=True, default_name="raster_values")
            data = rds.squeeze(drop=True).astype(np.float64)

            if data.ndim > 2:
                for dim in data.dims:
                    if dim not in (data.rio.x_dim, data.rio.y_dim):
                        data = data.isel({dim: 0})
                if data.ndim > 2:
                    raise ValueError(f"Could not reduce to 2D. Dims: {data.dims}")

            weights = data.fillna(0)
            total_weight = weights.sum()

            if total_weight.item() == 0:
                print("Zero-weight raster")
                return None, None

            x = (weights * data[data.rio.x_dim]).sum() / total_weight
            y = (weights * data[data.rio.y_dim]).sum() / total_weight

            return x.item(), y.item()
        except Exception as e:
            print(f"xarray centroid error: {e}")
            return None, None

    def _save_catalog(self, df: pd.DataFrame):
        df.to_pickle(self.output_dir / "catalog.pkl")
        print(f"Catalog saved to {self.output_dir / 'catalog.pkl'}")