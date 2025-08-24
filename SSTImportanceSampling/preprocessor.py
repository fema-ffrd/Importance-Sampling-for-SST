"""
Module to preprocess watershed and storm catalog data for use in SST Importance Sampling.

Use :class:`Preprocessor` to preprocess your data.
Call :meth:`Preprocessor.run` after initialization.
You can reuse a processed dataset using :meth:`Preprocessor.load`.

This module handles:
- Reading and projecting watershed/domain geometries
- Computing basic spatial stats (bounds, centroid, ranges) for watershed & domain
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
from affine import Affine

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


def _grid_transform_from_coords(x_coords: np.ndarray, y_coords: np.ndarray) -> Affine:
    """
    Build an Affine transform assuming x_coords/y_coords are cell centers on a regular grid.
    """
    dx = float(np.mean(np.diff(x_coords)))
    dy = float(np.mean(np.diff(y_coords)))
    # shift to upper-left corner of the (0,0) cell
    return Affine.translation(x_coords[0] - dx / 2.0, y_coords[0] - dy / 2.0) * Affine.scale(dx, dy)


# ---------- NEW: spatial stats helper ----------
def get_sp_stats(gdf: gpd.GeoDataFrame) -> pd.Series:
    """
    Compute polygon spatial stats:
      - bounds: minx, miny, maxx, maxy
      - centroid: x, y (of unary union)
      - ranges: range_x, range_y (extent width/height)

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame

    Returns
    -------
    pandas.Series
    """
    minx, miny, maxx, maxy = gdf.total_bounds
    cen = gdf.geometry.unary_union.centroid
    return pd.Series({
        "minx": float(minx),
        "miny": float(miny),
        "maxx": float(maxx),
        "maxy": float(maxy),
        "x": float(cen.x),
        "y": float(cen.y),
        "range_x": float(maxx - minx),
        "range_y": float(maxy - miny),
    })


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
        watershed_stats: Series of watershed spatial stats (bounds, centroid, ranges).
        domain_stats: Series of domain spatial stats (bounds, centroid, ranges).
        spatial_stats_path: Path to saved JSON with both stats.
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

        # NEW: stats placeholders + output path
        self.watershed_stats: pd.Series | None = None
        self.domain_stats: pd.Series | None = None
        self.spatial_stats_path: str | None = None

    def run(self):
        """
        Executes the full preprocessing workflow.

        This includes:
        - Reprojecting geometries
        - Computing & saving watershed/domain spatial stats (JSON)
        - Reading DSS files and calculating cumulative precipitation
        - Identifying storm centers
        - Saving results to NetCDF, Parquet, and GeoPackage
        - Updating the config file with output paths
        """
        self._process_geometries()
        self._compute_and_save_spatial_stats()   # NEW
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

    # ---------- NEW: compute + save stats ----------
    def _compute_and_save_spatial_stats(self) -> None:
        """
        Compute spatial stats for watershed and domain and save them to JSON.
        The JSON contains:
          {"watershed_stats": {...}, "domain_stats": {...}}
        """
        if self.watershed_gdf is None or self.domain_gdf is None:
            raise ValueError("Geometries must be processed before computing spatial stats.")

        self.watershed_stats = get_sp_stats(self.watershed_gdf)
        self.domain_stats = get_sp_stats(self.domain_gdf)

        stats = {
            "watershed_stats": self.watershed_stats.to_dict(),
            "domain_stats": self.domain_stats.to_dict(),
        }
        self.spatial_stats_path = os.path.join(self.output_folder, "spatial_stats.json")
        with open(self.spatial_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

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
            pmax = float(np.nanmax(self.cumulative_precip[name].values))
            self.storm_centers.loc[len(self.storm_centers)] = {
                "storm_path": name,
                "x": center.x,
                "y": center.y,
                "pmax_mm": pmax,
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
        # Adjust to cell center instead of corner
        x_coords = ll_x + (0.5 + np.arange(cols)) * cell_size
        y_coords = ll_y + (0.5 + np.arange(rows)) * cell_size

        return xr.DataArray(
            cumulative,
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords},
            attrs={"units": "mm", "cell_size": cell_size, "crs": self.shg_crs.to_string()},
        )

    def _compute_storm_center(self, da: xr.DataArray, restrict_to_domain: bool = True) -> Point:
        """
        Storm center = cell center (x,y) of the maximum cumulative precipitation.

        Args:
            da: 2D DataArray with dims ('y','x') and coords 'x','y' (cell centers).
            restrict_to_domain: If True, find the max only within the transposition domain polygon.

        Returns:
            shapely Point at the cell center of the max value.
        """
        arr = da.values.astype(float)
        x_coords = da["x"].values
        y_coords = da["y"].values

        # Optionally mask to domain polygon
        if restrict_to_domain and self.domain_gdf is not None and len(self.domain_gdf) > 0:
            rows, cols = arr.shape
            transform = _grid_transform_from_coords(x_coords, y_coords)
            domain_poly = self.domain_gdf.geometry.unary_union
            dom_mask = rasterize(
                [(domain_poly, 1)],
                out_shape=(rows, cols),
                transform=transform,
                fill=0,
                dtype="uint8",
            ).astype(bool)
            # mask out-of-domain cells
            arr = np.where(dom_mask, arr, np.nan)

        # Guard: if all NaN after masking, fall back to global max
        if np.isnan(arr).all():
            arr = da.values

        # Index of the maximum (ignoring NaNs)
        flat_idx = np.nanargmax(arr)
        i, j = np.unravel_index(flat_idx, arr.shape)   # i=row (y), j=col (x)

        # Map to cell-center coordinates
        x = float(x_coords[j])
        y = float(y_coords[i])
        return Point(x, y)

    def _save_outputs(self):
        """Saves NetCDF and Parquet output files to output folder."""
        # storm_path dimension matches DSS file names
        storm_names = list(self.cumulative_precip.keys())
        da_stack = xr.concat(
            [self.cumulative_precip[name] for name in storm_names],
            dim=xr.DataArray(storm_names, dims="storm_path", name="storm_path")
        )
        da_stack.name = "cumulative_precip"

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
            # NEW: include spatial stats JSON
            "spatial_stats": self.spatial_stats_path,
        }

        out_config_path = os.path.join(self.output_folder, "config.json")
        with open(out_config_path, "w") as f:
            json.dump(self.config, f, indent=4)

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> "Preprocessor":
        """
        Load a preprocessed Preprocessor instance from disk using a config file.

        This method reads a `config.json` file (either passed directly or
        found inside the given folder), parses the `preprocessed` section,
        and loads:

        - The projected watershed and domain geometries (GeoPackage files)
        - The cumulative precipitation dataset (NetCDF/DataArray)
        - The storm center locations (Parquet file)

        File paths are taken directly from the `config.json` entries.
        They may reside in any directory, not necessarily alongside
        the config file itself.

        Parameters
        ----------
        config_path : str or Path
            Path to the `config.json` file or to a folder containing it.

        Returns
        -------
        Preprocessor
            A `Preprocessor` instance with all referenced data loaded.

        Raises
        ------
        FileNotFoundError
            If the `config.json` file is missing at the provided location.
        ValueError
            If the `preprocessed` section is missing from the config file.
        """
        config_path = Path(config_path)
        if config_path.is_dir():
            config_path = config_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"No config.json found at {config_path}")

        obj = cls(config_path=config_path, output_folder=str(config_path.parent))

        with open(config_path, "r") as f:
            obj.config = json.load(f)

        pre = obj.config.get("preprocessed", {})
        if not pre:
            raise ValueError("Missing 'preprocessed' section in config.json")

        obj.watershed_gdf = gpd.read_file(pre["watershed_projected"])
        obj.domain_gdf = gpd.read_file(pre["domain_projected"])
        obj.cumulative_precip = xr.open_dataarray(pre["cumulative_precip"])
        obj.storm_centers = pd.read_parquet(pre["storm_centers"])

        # Optionally load back stats if present
        stats_path = pre.get("spatial_stats")
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            obj.watershed_stats = pd.Series(stats.get("watershed_stats", {}))
            obj.domain_stats = pd.Series(stats.get("domain_stats", {}))
            obj.spatial_stats_path = stats_path

        return obj
