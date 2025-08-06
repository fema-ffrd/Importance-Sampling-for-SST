"""
Module to preprocess watershed and storm catalog data for use in SST Importance Sampling.

Use :class:`Preprocessor` to preprocess your data.
Call :meth:`Preprocessor.run` after initialization.
You can reuse a processed dataset using :meth:`Preprocessor.load`.

This module handles:
- Reading and projecting watershed/domain geometries
- Reading DSS storm files and computing cumulative precipitation
- Computing optimal storm transposition centers
- Saving preprocessed outputs

See the usage guide for a full example.
"""

import os
import sys
import json
import glob
import contextlib
from typing import Dict, Union
from pathlib import Path

import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.affinity import translate
from affine import Affine
from rasterio.features import rasterize
from pyproj import CRS
from tqdm import tqdm
from hecdss import HecDss

SHG_WKT = (
    'PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",'
    'GEOGCS["GCS_North_American_1983",'
    'DATUM["D_North_American_1983",'
    'SPHEROID["GRS_1980",6378137.0,298.257222101]],'
    'PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Albers"],'
    'PARAMETER["False_Easting",0.0],'
    'PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",-96.0],'
    'PARAMETER["Standard_Parallel_1",29.5],'
    'PARAMETER["Standard_Parallel_2",45.5],'
    'PARAMETER["Latitude_Of_Origin",23.0],'
    'UNIT["Meter",1.0]]'
)


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Internal context manager to silence stdout/stderr. Not part of the public API."""
    with open(os.devnull, "w") as devnull:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)

class Preprocessor:
    """
    Main class to handle preprocessing of storm catalogs and watershed data.

    Attributes:
        config_path: Path to the input config.json file.
        output_folder: Directory to store processed outputs.
        shg_crs: SHG projection as a pyproj.CRS object.
        watershed_gdf: GeoDataFrame of projected watershed.
        domain_gdf: GeoDataFrame of projected domain.
        cumulative_precip: Dict of cumulative precipitation DataArrays.
        storm_centers: DataFrame of storm centers (name, x, y).
    """

    def __init__(self, config_path: str, output_folder: str):
        self.config_path = config_path
        self.output_folder = output_folder
        self.shg_crs = CRS.from_wkt(SHG_WKT)
        os.makedirs(self.output_folder, exist_ok=True)

        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.watershed_gdf = None
        self.domain_gdf = None
        self.cumulative_precip: Dict[str, xr.DataArray] = {}
        self.storm_centers = pd.DataFrame(columns=["storm_path", "x", "y"])

    def run(self):
        """
        Executes the full preprocessing workflow.

        This includes:
        - Reprojecting geometries
        - Reading DSS files and calculating cumulative precipitation
        - Identifying storm centers
        - Saving results to NetCDF, Parquet, and GeoPackage
        - Updating the config file with output paths
        """
        self._process_geometries()
        self._process_dss_files()
        self._save_outputs()
        self._update_config()

    def _process_geometries(self):
        """Reads and reprojects watershed and domain GeoJSONs to SHG, then saves as GPKG."""
        watershed_path = self.config["watershed"]["geometry_file"]
        domain_path = self.config["transposition_region"]["geometry_file"]

        self.watershed_gdf = gpd.read_file(watershed_path).to_crs(self.shg_crs)
        self.domain_gdf = gpd.read_file(domain_path).to_crs(self.shg_crs)

        self.watershed_path_out = os.path.join(self.output_folder, "watershed_projected.gpkg")
        self.domain_path_out = os.path.join(self.output_folder, "domain_projected.gpkg")

        self.watershed_gdf.to_file(self.watershed_path_out, driver="GPKG")
        self.domain_gdf.to_file(self.domain_path_out, driver="GPKG")

    def _process_dss_files(self):
        """Reads DSS files, computes cumulative precipitation, and determines storm center."""
        dss_folder = self.config["catalog"]["catalog_folder"]
        dss_files = sorted(glob.glob(os.path.join(dss_folder, "*.dss")))

        for dss_file in tqdm(dss_files, desc="Processing DSS files"):
            name = os.path.splitext(os.path.basename(dss_file))[0]

            with _suppress_stdout_stderr():
                da = self._read_dss_cumulative(dss_file)

            self.cumulative_precip[name] = da
            center = self._compute_storm_center(da)
            self.storm_centers.loc[len(self.storm_centers)] = {
                "storm_path": name,
                "x": center.x,
                "y": center.y,
            }

    def _read_dss_cumulative(self, dss_path: str) -> xr.DataArray:
        """Reads a DSS file and returns cumulative precipitation as an xarray.DataArray."""
        dss = HecDss(dss_path)
        catalog = dss.get_catalog()
        paths = [p for p in catalog.uncondensed_paths if "PRECIPITATION" in p and "SHG" in p]
        data_list = []
        sample_grid = None
        cell_size = ll_x = ll_y = None

        for path in paths:
            record = dss.get(path)
            grid = record.data.astype(np.float32)
            grid[grid == -3.4028235e38] = np.nan
            data_list.append(grid)

            if sample_grid is None:
                sample_grid = grid
                cell_size = record.cellSize
                ll_x = record.lowerLeftCellX * cell_size
                ll_y = record.lowerLeftCellY * cell_size

        dss.close()

        data_stack = np.stack(data_list, axis=0)
        cumulative = np.nansum(data_stack, axis=0)

        rows, cols = cumulative.shape
        x_coords = ll_x + np.arange(cols) * cell_size
        y_coords = ll_y + np.arange(rows) * cell_size

        return xr.DataArray(
            cumulative,
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords},
            attrs={"units": "mm", "cell_size": cell_size, "crs": self.shg_crs.to_string()},
        )

    def _compute_storm_center(self, da: xr.DataArray) -> Point:
        """Finds best storm center position by maximizing precipitation over watershed."""
        precip = da.values
        y_coords = da["y"].values
        x_coords = da["x"].values
        dx = np.mean(np.diff(x_coords))
        dy = np.mean(np.diff(y_coords))
        transform = Affine.translation(x_coords[0] - dx / 2, y_coords[0] - dy / 2) * Affine.scale(
            dx, dy
        )

        rows, cols = precip.shape
        watershed_poly = self.watershed_gdf.geometry.iloc[0]
        domain_poly = self.domain_gdf.geometry.iloc[0]

        watershed_mask = rasterize([(watershed_poly, 1)], out_shape=(rows, cols), transform=transform)
        domain_mask = rasterize([(domain_poly, 1)], out_shape=(rows, cols), transform=transform)

        w_inds = np.argwhere(watershed_mask)
        h_w, w_w = np.ptp(w_inds, axis=0) + 1

        best_total = -np.inf
        best_shift = (0, 0)

        for i in range(rows - h_w):
            for j in range(cols - w_w):
                sub_d = domain_mask[i : i + h_w, j : j + w_w]
                sub_p = precip[i : i + h_w, j : j + w_w]
                w_mask = watershed_mask[
                    w_inds[:, 0].min() : w_inds[:, 0].min() + h_w,
                    w_inds[:, 1].min() : w_inds[:, 1].min() + w_w,
                ]
                if sub_d.shape != w_mask.shape or np.any((w_mask == 1) & (sub_d == 0)):
                    continue
                total = np.nansum(sub_p[w_mask == 1])
                if total > best_total:
                    best_total = total
                    best_shift = (j - w_inds[:, 1].min(), i - w_inds[:, 0].min())

        dx_m = best_shift[0] * dx
        dy_m = best_shift[1] * dy
        shifted_poly = translate(watershed_poly, xoff=dx_m, yoff=dy_m)
        return shifted_poly.centroid

    def _save_outputs(self):
        """Saves NetCDF and Parquet output files to output folder."""
        # Add stacking to make a 3D DataArray
        da_stack = xr.concat(
            list(self.cumulative_precip.values()),
            dim=pd.Index(self.cumulative_precip.keys(), name="storm_path")
        )
        self.nc_path = os.path.join(self.output_folder, "cumulative_precip.nc")
        da_stack.to_netcdf(self.nc_path)

        self.storm_center_path = os.path.join(self.output_folder, "storm_centers.pq")
        self.storm_centers.to_parquet(self.storm_center_path, index=False)

    def _update_config(self):
        """Updates the config.json file with processed output file paths."""
        self.config["preprocessed"] = {
            "watershed_projected": self.watershed_path_out,
            "domain_projected": self.domain_path_out,
            "cumulative_precip": self.nc_path,
            "storm_centers": self.storm_center_path,
        }

        out_config_path = os.path.join(self.output_folder, "config.json")
        with open(out_config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    @classmethod
    def load(cls, output_folder: Union[str, Path]) -> "Preprocessor":
        """
        Load a preprocessed Preprocessor instance from disk using the config file.

        This method reads the `config.json` file located in the specified output folder,
        loads the projected watershed and domain geometries, the cumulative precipitation
        NetCDF dataset, and the storm center locations stored in a Parquet file.

        Parameters
        ----------
        output_folder : str or Path
            Path to the folder containing the `config.json` file and all preprocessed outputs.

        Returns
        -------
        Preprocessor
            A `Preprocessor` instance with all data loaded and ready for use.

        Raises
        ------
        FileNotFoundError
            If the `config.json` file is missing in the provided folder.
        ValueError
            If the `preprocessed` section is missing from the config file.
        """
        output_folder = Path(output_folder)
        config_path = output_folder / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found in {output_folder}")

        obj = cls(config_path=config_path, output_folder=str(output_folder))

        with open(config_path, "r") as f:
            obj.config = json.load(f)

        pre = obj.config.get("preprocessed", {})
        if not pre:
            raise ValueError("Missing 'preprocessed' section in config.json")

        obj.watershed_gdf = gpd.read_file(pre["watershed_projected"])
        obj.domain_gdf = gpd.read_file(pre["domain_projected"])
        obj.cumulative_precip = xr.open_dataarray(pre["cumulative_precip"])
        obj.storm_centers = pd.read_parquet(pre["storm_centers"])

        return obj
