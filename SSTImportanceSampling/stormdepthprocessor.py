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
    Single-runner processor:
      - init with `data` (expects: data.cumulative_precip, data.storm_centers, data.watershed_gdf)
      - call .run(df_storms, n_jobs=-1) to get depths + exceedance probabilities

    df_storms must include: ['storm_path','newx','newy','weight','event_id'] (+ optional 'rep')
    If ANY cell inside the watershed is NaN after transposition → precip_avg_mm = NaN.
    """

    def __init__(self, data):
        self.precip_cube: xr.DataArray = data.cumulative_precip  # dims: ('storm_path','y','x')
        self.storm_centers: pd.DataFrame = data.storm_centers.set_index("storm_path")
        watershed_gdf: gpd.GeoDataFrame = data.watershed_gdf

        self.x_coords = self.precip_cube.coords["x"].values
        self.y_coords = self.precip_cube.coords["y"].values
        self.dx = float(np.mean(np.diff(self.x_coords)))
        self.dy = float(np.mean(np.diff(self.y_coords)))

        self.transform = (
            Affine.translation(self.x_coords[0] - self.dx/2.0,
                               self.y_coords[0] - self.dy/2.0)
            * Affine.scale(self.dx, self.dy)
        )
        self.watershed_mask = geometry_mask(
            geometries=[mapping(geom) for geom in watershed_gdf.geometry],
            out_shape=(len(self.y_coords), len(self.x_coords)),
            transform=self.transform,
            invert=True,
        )

        # storm_path -> index into cube
        cube_paths = pd.Index(self.precip_cube.coords["storm_path"].values)
        self._path_to_idx = {sp: i for i, sp in enumerate(cube_paths)}

    # -------- public: single runner --------
    def run(self, df_storms: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
        """
        Returns DataFrame with:
        ['event_id','storm_path','x','y','weight','precip_avg_mm','exc_prb','rep']
        """
        if df_storms.empty:
            return pd.DataFrame()

        needed = {"storm_path","newx","newy","weight","event_id"}
        missing = needed - set(df_storms.columns)
        if missing:
            raise ValueError(f"`df_storms` missing required columns: {missing}")

        reps = df_storms["rep"] if "rep" in df_storms.columns else 1
        rows = df_storms.assign(rep=reps).to_dict(orient="records")

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(self._process_single)(pd.Series(r)) for r in rows
        )
        out = pd.DataFrame([r for r in results if r is not None])
        if out.empty:
            return out

        out["rep"] = [r["rep"] if "rep" in r else 1 for r in rows if r is not None][: len(out)]
        return self._add_exc_prb(out)

    # -------- internals --------
    def _process_single(self, row: pd.Series) -> Union[dict, None]:
        sp = row["storm_path"]
        if sp not in self.storm_centers.index or sp not in self._path_to_idx:
            return None

        x_orig, y_orig = self.storm_centers.loc[sp, ["x", "y"]]
        x_new, y_new = float(row["newx"]), float(row["newy"])

        dx_cells = int(round((x_new - x_orig) / self.dx))
        dy_cells = int(round((y_new - y_orig) / self.dy))

        precip = self.precip_cube.isel(storm_path=self._path_to_idx[sp]).values
        shifted = np.roll(precip, shift=(dy_cells, dx_cells), axis=(0, 1))

        if dy_cells > 0:   shifted[:dy_cells, :] = np.nan
        elif dy_cells < 0: shifted[dy_cells:, :] = np.nan
        if dx_cells > 0:   shifted[:, :dx_cells] = np.nan
        elif dx_cells < 0: shifted[:, dx_cells:] = np.nan

        vals = shifted[self.watershed_mask]
        precip_avg = np.nan if np.isnan(vals).any() else (float(vals.mean()) if vals.size else np.nan)

        return {
            "event_id": str(int(row["event_id"])) if not isinstance(row["event_id"], str) else row["event_id"],
            "storm_path": sp,
            "x": x_new, "y": y_new,
            "weight": float(row["weight"]),
            "precip_avg_mm": precip_avg,
            "rep": row.get("rep", 1),
        }

    def _add_exc_prb(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        valid = df["precip_avg_mm"].notna()
        d = df.loc[valid].sort_values("precip_avg_mm", ascending=False)
        d["exc_prb"] = d["weight"].cumsum()
        df["exc_prb"] = np.nan
        df.loc[d.index, "exc_prb"] = d["exc_prb"]
        return df
