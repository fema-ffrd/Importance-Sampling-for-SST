"""
Module for computing watershed-averaged precipitation and exceedance probability from
sampled storm center transpositions

This module provides the `StormDepthProcessor` class which handles:
- Shifting storm footprints to new sampled storm center locations
- Masking shifted precipitation arrays with watershed geometry
- Computing watershed-averaged precipitation depths
- Estimating exceedance probabilities using arrival rate theory
- Efficient parallelized processing with Joblib

Typical usage example::

    processor = StormDepthProcessor(precip_cube, storm_centers, watershed_gdf)
    results_df = processor.shift_and_extract_precip(df_samples)
    results_df = processor.add_exc_prb(results_df)    
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Union
from joblib import Parallel, delayed
from shapely.geometry import mapping
from rasterio.features import geometry_mask
from affine import Affine
import geopandas as gpd


class StormDepthProcessor:
    """
    Computes watershed-averaged precipitation for sampled storm centers.

    This class shifts storm footprints over sampled storm centers, computes
    precipitation depth within a watershed, and derives return periods based on an arrival rate.
    """

    def __init__(
        self,
        precip_cube: xr.DataArray,
        storm_centers: pd.DataFrame,
        watershed_gdf: gpd.GeoDataFrame,
    ) -> None:
        """
        Initialize the StormDepthProcessor.

        Args:
            precip_cube (xr.DataArray): Aggregated precipitation with dims (storm_path, y, x).
            storm_centers (pd.DataFrame): DataFrame with columns ['storm_path', 'x', 'y'].
            watershed_gdf (gpd.GeoDataFrame): Watershed polygon in the same CRS as the cube.
        """
        self.precip_cube = precip_cube
        self.storm_centers = storm_centers.set_index("storm_path")

        self.x_coords = self.precip_cube.coords["x"].values
        self.y_coords = self.precip_cube.coords["y"].values
        self.dx = float(np.mean(np.diff(self.x_coords)))
        self.dy = float(np.mean(np.diff(self.y_coords)))

        self.transform = Affine.translation(self.x_coords[0] - self.dx/2.0,
                                            self.y_coords[0] - self.dy/2.0) * Affine.scale(self.dx, self.dy)

        self.watershed_mask = geometry_mask(
            geometries=[mapping(geom) for geom in watershed_gdf.geometry],
            out_shape=(len(self.y_coords), len(self.x_coords)),
            transform=self.transform,
            invert=True,
        )

    def _process_single(
        self,
        storm_path: str,
        storm_index: int,
        row: pd.Series,
    ) -> Union[dict, None]:
        """
        Process a single storm center shift and compute average watershed precipitation.

        Args:
            storm_path (str): Identifier of the storm event.
            storm_index (int): Index of the storm in the precipitation cube.
            row (pd.Series): Row from sampled storm centers containing x, y, weight.

        Returns:
            dict or None: Dictionary of computed values or None if storm_path is invalid.
        """
        if storm_path not in self.storm_centers.index:
            return None

        x_orig, y_orig = self.storm_centers.loc[storm_path, ["x", "y"]]
        x_new, y_new = row["x"], row["y"]

        dx_cells = int(round((x_new - x_orig) / self.dx))
        dy_cells = int(round((y_new - y_orig) / self.dy))

        precip = self.precip_cube.isel(storm_path=storm_index)
        shifted = np.roll(precip.values, shift=(dy_cells, dx_cells), axis=(0, 1))

        if dy_cells > 0:
            shifted[:dy_cells, :] = 0
        elif dy_cells < 0:
            shifted[dy_cells:, :] = 0

        if dx_cells > 0:
            shifted[:, :dx_cells] = 0
        elif dx_cells < 0:
            shifted[:, dx_cells:] = 0

        masked_precip = np.where(self.watershed_mask, shifted, 0)
        masked_precip = np.where(np.isnan(masked_precip), 0, masked_precip)
        precip_avg = masked_precip.sum() / self.watershed_mask.sum()

        return {
            "event_id": str(int(row["event_id"])) if not isinstance(row["event_id"], str) else row["event_id"],
            "x": x_new,
            "y": y_new,
            "weight": row["weight"],
            "storm_path": storm_path,
            "precip_avg_mm": precip_avg,
        }

    def add_exc_prb(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute exceedance probabilities from precipitation depths.

        Args:
            df (pd.DataFrame): DataFrame with column 'precip_avg_mm'.

        Returns:
            pd.DataFrame: DataFrame with added 'exc_prb' column.
        """
        df_sorted = df.sort_values("precip_avg_mm", ascending=False).reset_index(drop=True).copy()
        df_sorted["exc_prb"] = df_sorted["weight"].cumsum()
        return df_sorted

    def shift_and_extract_precip(
        self,
        df_storms: pd.DataFrame,
        n_jobs: int = -1,
        seed: int = None,
    ) -> pd.DataFrame:
        """
        Apply shifts to all sampled storm centers and compute precipitation metrics.

        Args:
            df_storms (pd.DataFrame): DataFrame with sampled storm centers (columns: rep, event_id, x, y, weight).
            n_jobs (int): Number of parallel jobs for processing.
            seed (int, optional): Random seed for reproducibility.

        Returns:
            pd.DataFrame: Aggregated results including average precipitation and return periods.
        """
        if seed is not None:
            np.random.seed(seed)

        if "rep" not in df_storms.columns:
            df_storms["rep"] = 1

        storm_paths = self.precip_cube.coords["storm_path"].values
        all_results = []

        for rep_id, df_rep in df_storms.groupby("rep"):
            selected_storms = np.random.choice(storm_paths, size=len(df_rep), replace=True)

            tasks = [
                (storm_path, np.where(storm_paths == storm_path)[0][0], row)
                for storm_path, (_, row) in zip(selected_storms, df_rep.iterrows())
            ]

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._process_single)(storm_path, idx, row)
                for storm_path, idx, row in tasks
            )

            df_result = pd.DataFrame([r for r in results if r is not None])
            if df_result.empty:
                continue

            df_result["rep"] = rep_id
            if "weight" in df_result.columns:
                df_result = self.add_exc_prb(df_result)
            all_results.append(df_result)

        return pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()