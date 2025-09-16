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
    Works with cumulative_precip as:
      - xr.DataArray with dims ('storm_path','y','x'), or
      - xr.Dataset containing var 'cumulative_precip', or
      - dict[str, xr.DataArray(y,x)] mapping storm_path -> 2D field.

    Expects the samples DataFrame to include:
      required: ['storm_path','newx','newy','weight','event_id']
      optional: ['realization','realization_seed']

    Outputs per row:
      ['event_id','storm_path','x','y','weight','precip_avg_mm','exc_prb','realization','realization_seed?']
    """

    def __init__(self, data):
        raw = data.cumulative_precip

        # --- normalize cumulative_precip into either a 3D DataArray or a dict ---
        self._cube: xr.DataArray | None = None
        self._per_storm: dict[str, xr.DataArray] | None = None

        if isinstance(raw, xr.Dataset):
            self._cube = raw["cumulative_precip"]
        elif isinstance(raw, xr.DataArray):
            self._cube = raw
        elif isinstance(raw, dict):
            self._per_storm = raw
        else:
            raise TypeError("Unsupported type for cumulative_precip.")

        # coords/x-y & storm list
        if self._cube is not None:
            self.x_coords = self._cube.coords["x"].values
            self.y_coords = self._cube.coords["y"].values
            cube_paths = pd.Index(self._cube.coords["storm_path"].values.astype(str))
            self._path_to_idx = {sp: i for i, sp in enumerate(cube_paths)}
        else:
            any_da = next(iter(self._per_storm.values()))
            self.x_coords = any_da.coords["x"].values
            self.y_coords = any_da.coords["y"].values
            self._path_to_idx = {}

        self.dx = float(np.mean(np.diff(self.x_coords)))
        self.dy = float(np.mean(np.diff(self.y_coords)))

        # grid->affine (centers -> upper-left origin)
        self.transform = (
            Affine.translation(self.x_coords[0] - self.dx/2.0,
                               self.y_coords[0] - self.dy/2.0)
            * Affine.scale(self.dx, self.dy)
        )

        # watershed mask in grid space
        watershed_gdf: gpd.GeoDataFrame = data.watershed_gdf
        self.watershed_mask = geometry_mask(
            geometries=[mapping(geom) for geom in watershed_gdf.geometry],
            out_shape=(len(self.y_coords), len(self.x_coords)),
            transform=self.transform,
            invert=True,
        )

        # storm centers table
        self.storm_centers: pd.DataFrame = data.storm_centers.set_index("storm_path")

    # unified getter for a storm's 2D precip field
    def _get_precip_by_path(self, storm_path: str) -> np.ndarray:
        if self._cube is not None:
            idx = self._path_to_idx.get(storm_path)
            if idx is None:
                raise KeyError(f"storm_path '{storm_path}' not found in cube coords.")
            return self._cube.isel(storm_path=idx).values
        else:
            da = self._per_storm.get(storm_path)
            if da is None:
                raise KeyError(f"storm_path '{storm_path}' not found in dict.")
            return da.values

    # -------- public: single runner --------
    def run(self, df_storms: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
        if df_storms.empty:
            return pd.DataFrame()

        needed = {"storm_path","newx","newy","weight","event_id"}
        missing = needed - set(df_storms.columns)
        if missing:
            raise ValueError(f"`df_storms` missing required columns: {missing}")

        # Ensure realization column exists; carry realization_seed through if present
        if "realization" not in df_storms.columns:
            df_storms = df_storms.assign(realization=1)

        rows = df_storms.to_dict(orient="records")

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._process_single)(pd.Series(r)) for r in rows
        )
        out = pd.DataFrame([r for r in results if r is not None])
        if out.empty:
            return out

        # exceedance probability per realization
        return self._add_exc_prb(out)

    # -------- internals --------
    def _process_single(self, row: pd.Series) -> Union[dict, None]:
        sp = str(row["storm_path"])
        if sp not in self.storm_centers.index:
            return None

        x_orig, y_orig = self.storm_centers.loc[sp, ["x", "y"]]
        x_new, y_new = float(row["newx"]), float(row["newy"])

        dx_cells = int(round((x_new - x_orig) / self.dx))
        dy_cells = int(round((y_new - y_orig) / self.dy))

        precip = self._get_precip_by_path(sp)           # (y,x) ndarray
        shifted = np.roll(precip, shift=(dy_cells, dx_cells), axis=(0, 1))

        # cut rolled edges to NaN (no wrap)
        if dy_cells > 0:   shifted[:dy_cells, :] = np.nan
        elif dy_cells < 0: shifted[dy_cells:, :] = np.nan
        if dx_cells > 0:   shifted[:, :dx_cells] = np.nan
        elif dx_cells < 0: shifted[:, dx_cells:] = np.nan

        vals = shifted[self.watershed_mask]
        precip_avg = np.nan if np.isnan(vals).any() else (float(vals.mean()) if vals.size else np.nan)

        out = {
            "event_id": str(int(row["event_id"])) if not isinstance(row["event_id"], str) else row["event_id"],
            "storm_path": sp,
            "x": x_new, "y": y_new,
            "weight": float(row["weight"]),
            "precip_avg_mm": precip_avg,
            "realization": row.get("realization", 1),
        }
        if "realization_seed" in row:
            out["realization_seed"] = row["realization_seed"]
        return out

    def _add_exc_prb(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["exc_prb"] = np.nan
        for r, group in df.groupby("realization", dropna=False):
            valid = group["precip_avg_mm"].notna()
            d = group.loc[valid].sort_values("precip_avg_mm", ascending=False).copy()
            d["exc_prb"] = d["weight"].cumsum()
            df.loc[d.index, "exc_prb"] = d["exc_prb"]
        return df
