"""
Module to preprocess watershed and storm catalog data for use in SST Importance Sampling.

Use :class:`Preprocessor` to preprocess your data.
Call :meth:`Preprocessor.run` after initialization.
You can reuse a processed dataset using :meth:`Preprocessor.load`.

This module handles:
    - Reading and projecting watershed/domain geometries
    - Computing basic spatial stats (bounds, centroid, ranges) for watershed & domain
    - Reading DSS storm files and computing cumulative precipitation
    - Computing storm centers (max cumulative precip cell)
    - Building a single master grid definition (meta)
    - Computing per-storm VALID storm-center masks
    - Saving preprocessed outputs (NetCDF + Parquet + JSON meta)
"""

import os
import json
import glob
import contextlib
from typing import Dict, Union, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from affine import Affine
from pyproj import CRS
from shapely.geometry import Point, LineString
from shapely.ops import unary_union
from shapely.affinity import translate
from rasterio.features import rasterize
from tqdm import tqdm
from hecdss import HecDss


# --------------------- CRS (SHG) ---------------------
SHG_WKT = (
    'PROJCS["USA_Contiguous_Albers_Equal_Area_Conic_USGS_version",'
    'GEOGCS["GCS_North_American_1983",'
    'DATUM["D_North_American_1983",'
    'SPHEROID["GRS_1980",6378137.0,298.257222101]],'
    'PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Albers"],'
    'PARAMETER["False_Easting",0.0],'
    'PARAMETER["Central_Meridian",-96.0],'
    'PARAMETER["Standard_Parallel_1",29.5],'
    'PARAMETER["Standard_Parallel_2",45.5],'
    'PARAMETER["Latitude_Of_Origin",23.0],'
    'UNIT["Meter",1.0]]'
)


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Silence stdout/stderr (used when calling HEC-DSS)."""
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


# --------------------- helpers ---------------------
def _grid_transform_from_coords(x_coords: np.ndarray, y_coords: np.ndarray) -> Affine:
    """Affine for a regular grid when x,y are cell CENTERS."""
    dx = float(np.mean(np.diff(x_coords)))
    dy = float(np.mean(np.diff(y_coords)))
    return Affine.translation(x_coords[0] - dx/2.0, y_coords[0] - dy/2.0) * Affine.scale(dx, dy)

def _sp_stats(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Bounds, centroid, ranges for polygons."""
    minx, miny, maxx, maxy = gdf.total_bounds
    cen = gdf.geometry.unary_union.centroid
    return pd.Series({
        "minx": float(minx), "miny": float(miny),
        "maxx": float(maxx), "maxy": float(maxy),
        "x": float(cen.x), "y": float(cen.y),
        "range_x": float(maxx - minx), "range_y": float(maxy - miny),
    })

def _rasterize_bool(geom, shape: Tuple[int,int], transform: Affine) -> np.ndarray:
    """Rasterize geometry to a boolean mask (True inside)."""
    return rasterize([(geom, 1)], out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)


# --------------------- main class ---------------------
class Preprocessor:
    """
    Preprocess storm catalogs and watershed/domain for SST-IS.

    Outputs (persisted):
        - watershed_projected.gpkg      (GeoPackage)
        - domain_projected.gpkg         (GeoPackage)
        - spatial_stats.json            (JSON)
        - cumulative_precip.nc          (NetCDF DataArray: cumulative_precip[storm_path,y,x])
        - storm_centers.pq              (Parquet: storm_path, x, y, pmax_mm)
        - grid_meta.json                (JSON: CRS/shape/affine/cell_size/x0/y0)
        - valid_mask.nc                 (NetCDF DataArray: valid_mask[storm,y,x] as uint8)
    """

    # --------------- init ---------------
    def __init__(self, config_path: str, output_folder: str):
        self.config_path = str(config_path)
        self.output_folder = str(output_folder)
        os.makedirs(self.output_folder, exist_ok=True)

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        self.shg_crs = CRS.from_wkt(SHG_WKT)

        # geoms
        self.watershed_gdf: Optional[gpd.GeoDataFrame] = None
        self.domain_gdf: Optional[gpd.GeoDataFrame] = None

        # data products (in-memory)
        self.cumulative_precip: Dict[str, xr.DataArray] = {}
        self.storm_centers = pd.DataFrame(columns=["storm_path", "x", "y", "pmax_mm"])
        self.valid_mask_nc = None

        # stats
        self.watershed_stats: Optional[pd.Series] = None
        self.domain_stats: Optional[pd.Series] = None

        # internal valid centroid region (not persisted in config)
        self.valid_centroid_region_gdf: Optional[gpd.GeoDataFrame] = None

        # paths
        self.watershed_path_out = os.path.join(self.output_folder, "watershed_projected.gpkg")
        self.domain_path_out = os.path.join(self.output_folder, "domain_projected.gpkg")
        self.spatial_stats_path = os.path.join(self.output_folder, "spatial_stats.json")
        self.nc_path = os.path.join(self.output_folder, "cumulative_precip.nc")
        self.storm_center_path = os.path.join(self.output_folder, "storm_centers.pq")
        self.grid_meta_path = os.path.join(self.output_folder, "grid_meta.json")
        self.valid_mask_nc_path = os.path.join(self.output_folder, "valid_mask.nc")

        # options
        opts = self.config.get("preprocess_opts", {})
        self.vcr_max_seg_len: float = float(opts.get("valid_region_max_seg_len", 100.0))
        self.vcr_simplify_tol: float = float(opts.get("valid_region_simplify_tol", 0.0))

    # --------------- driver ---------------
    def run(self):
        """
        Workflow:
          1) Read & project geometries
          2) Stats → JSON
          3) Read DSS storms → cumulative precip + storm center per storm
          4) Build master grid meta from domain bbox aligned to DSS grid
          5) Compute the valid centroid region polygon
          6) Build per-storm valid masks using the valid centroid region
          7) Save outputs & update config
        """
        self._process_geometries()
        self._compute_and_save_spatial_stats()
        self._process_dss_files()
        grid_meta = self._build_master_grid_meta_from_dss_and_domain()
        self._compute_valid_centroid_region_internal()          
        self._compute_and_save_valid_masks_via_vcr(grid_meta)   # writes valid_mask.nc
        self._save_outputs()
        self._update_config(grid_meta)

    # --------------- steps ---------------
    def _process_geometries(self):
        """Read watershed/domain, project to SHG, save GPKG."""
        watershed_path = self.config["watershed"]["geometry_file"]
        domain_path = self.config["transposition_region"]["geometry_file"]
        self.watershed_gdf = gpd.read_file(watershed_path).to_crs(self.shg_crs)
        self.domain_gdf = gpd.read_file(domain_path).to_crs(self.shg_crs)
        self.watershed_gdf.to_file(self.watershed_path_out, driver="GPKG")
        self.domain_gdf.to_file(self.domain_path_out, driver="GPKG")

    def _compute_and_save_spatial_stats(self) -> None:
        if self.watershed_gdf is None or self.domain_gdf is None:
            raise ValueError("Geometries must be processed first.")
        self.watershed_stats = _sp_stats(self.watershed_gdf)
        self.domain_stats = _sp_stats(self.domain_gdf)
        stats = {
            "watershed_stats": self.watershed_stats.to_dict(),
            "domain_stats": self.domain_stats.to_dict(),
        }
        with open(self.spatial_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    # ---- DSS reading ----
    def _process_dss_files(self):
        """Read DSS files, compute cumulative precip, pick center (max cell)."""
        dss_folder = self.config["catalog"]["catalog_folder"]
        dss_files = sorted(glob.glob(os.path.join(dss_folder, "*.dss")))
        if len(dss_files) == 0:
            raise FileNotFoundError(f"No .dss files found under: {dss_folder}")

        for dss_file in tqdm(dss_files, desc="Processing DSS files"):
            name = os.path.splitext(os.path.basename(dss_file))[0]
            with _suppress_stdout_stderr():
                da = self._read_dss_cumulative(dss_file)
            self.cumulative_precip[name] = da

            center = self._max_cell_center(da)
            pmax = float(np.nanmax(da.values))
            self.storm_centers.loc[len(self.storm_centers)] = {
                "storm_path": name, "x": center.x, "y": center.y, "pmax_mm": pmax
            }

    def _read_dss_cumulative(self, dss_path: str) -> xr.DataArray:
        """Return cumulative precipitation for a DSS file as 2D DataArray (y,x)."""
        dss = HecDss(dss_path)
        catalog = dss.get_catalog()
        paths = [p for p in catalog.uncondensed_paths if "PRECIPITATION" in p and "SHG" in p]
        data_list = []
        cell_size = ll_x = ll_y = None

        for path in paths:
            record = dss.get(path)
            grid = record.data.astype(np.float32)
            grid[grid == -3.4028235e38] = np.nan
            data_list.append(grid)
            if cell_size is None:
                cell_size = record.cellSize
                ll_x = record.lowerLeftCellX * cell_size
                ll_y = record.lowerLeftCellY * cell_size
        dss.close()

        if len(data_list) == 0:
            raise ValueError(f"No PRECIPITATION/SHG grids found in {dss_path}")

        data_stack = np.stack(data_list, axis=0)
        cumulative = np.nansum(data_stack, axis=0)

        rows, cols = cumulative.shape
        x_coords = ll_x + (0.5 + np.arange(cols)) * cell_size
        y_coords = ll_y + (0.5 + np.arange(rows)) * cell_size

        return xr.DataArray(
            cumulative,
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords},
            attrs={"units": "mm", "cell_size": float(cell_size), "crs": str(self.shg_crs)},
        )

    @staticmethod
    def _max_cell_center(da: xr.DataArray) -> Point:
        arr = da.values
        if np.isnan(arr).all():
            return Point(float(da["x"].values[0]), float(da["y"].values[0]))
        flat = np.nanargmax(arr)
        i, j = np.unravel_index(flat, arr.shape)
        return Point(float(da["x"].values[j]), float(da["y"].values[i]))

    # ---- master grid meta ----
    def _build_master_grid_meta_from_dss_and_domain(self) -> Dict:
        """
        Build a master grid aligned to DSS grids covering the domain bbox.
        Returns dict with rows, cols, cell_size, transform (a,b,c,d,e,f), x0,y0, crs_wkt.
        """
        if not self.cumulative_precip:
            raise RuntimeError("DSS cumulative precip not available.")

        any_da: xr.DataArray = next(iter(self.cumulative_precip.values()))
        cell_size = float(any_da.attrs["cell_size"])

        D = self.domain_gdf.geometry.unary_union
        minx, miny, maxx, maxy = D.bounds

        def _aligned(min_c, max_c, cell):
            n = int(np.ceil((max_c - min_c) / cell))
            centers = min_c + (0.5 + np.arange(n)) * cell
            if centers.size == 0 or centers[-1] < (max_c - 0.5*cell):
                centers = np.append(centers, (centers[-1] + cell) if centers.size else (min_c + 0.5*cell))
            return centers

        xs = _aligned(minx, maxx, cell_size)
        ys = _aligned(miny, maxy, cell_size)
        rows, cols = len(ys), len(xs)
        transform = _grid_transform_from_coords(xs, ys)

        meta = {
            "crs_wkt": self.shg_crs.to_wkt(),
            "rows": int(rows),
            "cols": int(cols),
            "cell_size": float(cell_size),
            "transform": [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
            "x0": float(xs[0]),
            "y0": float(ys[0]),
        }
        with open(self.grid_meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return meta

    # ---- internal: valid centroid region ----
    @staticmethod
    def _valid_centroid_region(domain_gdf: gpd.GeoDataFrame,
                               watershed_gdf: gpd.GeoDataFrame,
                               max_seg_len: float = 100.0,
                               simplify_tol: float = 0.0) -> gpd.GeoDataFrame:
        """
        Polygon of all centroid positions s such that translating watershed by (s - c)
        keeps it entirely inside the domain. (c = watershed centroid)
        """
        D = unary_union(domain_gdf.geometry).buffer(0)
        B = unary_union(watershed_gdf.geometry).buffer(0)
        if simplify_tol > 0:
            D = D.simplify(simplify_tol, preserve_topology=True)
        c = B.centroid
        cx, cy = c.x, c.y

        def _densify_ring(coords):
            if max_seg_len <= 0:
                yield from coords; return
            coords = list(coords)
            for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:]):
                seg_len = LineString([(x1, y1), (x2, y2)]).length
                n = max(1, int(np.ceil(seg_len / max_seg_len)))
                for t in np.linspace(0.0, 1.0, n, endpoint=False):
                    yield (x1 + t*(x2 - x1), y1 + t*(y2 - y1))
            yield coords[-1]

        def _boundary_points(geom):
            if geom.geom_type == "Polygon":
                yield from _densify_ring(geom.exterior.coords)
                for ring in geom.interiors: yield from _densify_ring(ring.coords)
            elif geom.geom_type == "MultiPolygon":
                for g in geom.geoms: yield from _boundary_points(g)
            else:
                raise ValueError(f"Unexpected geometry type: {geom.geom_type}")

        feasible = D
        seen = set()
        for px, py in _boundary_points(B):
            dx, dy = -(px - cx), -(py - cy)
            key = (round(dx, 9), round(dy, 9))
            if key in seen: continue
            seen.add(key)
            feasible = feasible.intersection(translate(D, xoff=dx, yoff=dy))
            if feasible.is_empty: break

        return gpd.GeoDataFrame(geometry=[feasible], crs=domain_gdf.crs or watershed_gdf.crs)

    def _compute_valid_centroid_region_internal(self):
        """Compute & keep the valid centroid region in-memory only."""
        if self.watershed_gdf is None or self.domain_gdf is None:
            raise ValueError("Geometries must be processed before VCR.")
        self.valid_centroid_region_gdf = self._valid_centroid_region(
            self.domain_gdf, self.watershed_gdf,
            max_seg_len=self.vcr_max_seg_len,
            simplify_tol=self.vcr_simplify_tol
        )

    # ---- per-storm valid masks ----
    def _compute_and_save_valid_masks_via_vcr(self, grid_meta: Dict):
        """
        Build per-storm VALID masks using the valid-centroid region (VCR).

        P is valid for storm j  <=>  s := c + (Cj - P) ∈ VCR,
        where c is watershed centroid and Cj is the original storm center.
        Implemented as a reflect+shift lookup over a rasterized VCR.
        """
        import numpy as np
        import xarray as xr
        from affine import Affine

        if self.valid_centroid_region_gdf is None:
            raise RuntimeError("Valid centroid region not computed.")

        # --- unpack grid ---
        rows = int(grid_meta["rows"]); cols = int(grid_meta["cols"])
        cell = float(grid_meta["cell_size"])
        a, b, c_, d, e, f = grid_meta["transform"]
        transform = Affine(a, b, c_, d, e, f)
        x0 = float(grid_meta["x0"]); y0 = float(grid_meta["y0"])
        xs = x0 + np.arange(cols) * cell
        ys = y0 + np.arange(rows) * cell

        # Enforce candidate P must lie inside the domain polygon
        D = self.domain_gdf.geometry.unary_union
        in_domain = _rasterize_bool(D, (rows, cols), transform)  # (rows, cols) bool

        # --- rasterize ---
        VCR_geom = self.valid_centroid_region_gdf.geometry.unary_union
        if VCR_geom.is_empty:
            # nothing is feasible: all-zero masks
            vcr_mask = np.zeros((rows, cols), dtype=bool)
        else:
            vcr_mask = _rasterize_bool(VCR_geom, (rows, cols), transform)  # (rows, cols) bool

        # watershed centroid c
        if self.watershed_stats is None or "x" not in self.watershed_stats or "y" not in self.watershed_stats:
            raise RuntimeError(
                "watershed_stats missing. Ensure _compute_and_save_spatial_stats() ran before building valid masks."
            )
        cx_w = float(self.watershed_stats["x"])
        cy_w = float(self.watershed_stats["y"])

        # helper: map 1-D centers to integer indices of s = K - centers
        def idx_map(centers: np.ndarray, K: float) -> np.ndarray:
            step = centers[1] - centers[0]
            # i_s ≈ round((K - centers - centers[0]) / step)
            return np.rint((K - centers - centers[0]) / step).astype(np.int64)

        storm_ids = self.storm_centers["storm_path"].tolist()
        nS = len(storm_ids)
        out = np.zeros((nS, rows, cols), dtype="uint8")

        # Precompute per-axis in-bounds tests (broadcasted)
        ones_r = np.ones((rows, 1), dtype=np.int64)
        ones_c = np.ones((1, cols), dtype=np.int64)

        for s_idx, sid in enumerate(tqdm(storm_ids, desc="Valid mask via VCR")):
            row = self.storm_centers.loc[self.storm_centers["storm_path"] == sid]
            if row.empty:
                continue
            Cx = float(row["x"].values[0])
            Cy = float(row["y"].values[0])

            # K = c + Cj
            Kx = cx_w + Cx
            Ky = cy_w + Cy

            # Per-axis target indices for s = (Ky - ys[i], Kx - xs[j])
            is_ = idx_map(ys, Ky)   # shape: (rows,)
            js  = idx_map(xs, Kx)   # shape: (cols,)

            # Broadcast to 2-D index grids (rows x cols)
            IS = is_[:, None] * ones_c      # (rows, cols)
            JS = ones_r * js[None, :]       # (rows, cols)

            # In-bounds mask
            in_i = (IS >= 0) & (IS < rows)
            in_j = (JS >= 0) & (JS < cols)
            inb  = in_i & in_j

            # Clip to bounds for safe gather, then gather with advanced indexing
            ISc = np.clip(IS, 0, rows - 1)
            JSc = np.clip(JS, 0, cols - 1)
            vm_full = vcr_mask[ISc, JSc]    # (rows, cols), broadcasted advanced indexing

            # Zero where out-of-bounds; 1 where in-bounds and VCR true
            vm = np.zeros((rows, cols), dtype="uint8")
            vm[inb] = vm_full[inb].astype("uint8")
            vm &= in_domain  # disallow centers outside irregular domain boundary

            out[s_idx] = vm

        # Save NetCDF
        da_valid = xr.DataArray(
            out, name="valid_mask",
            dims=("storm", "y", "x"),
            coords={"storm": storm_ids, "y": ys, "x": xs},
            attrs={
                "description": (
                    "Valid storm-center mask (1=valid, 0=invalid). A cell P is valid for storm j "
                    "iff c + (Cj - P) lies in the valid centroid region (c = watershed centroid)."
                ),
                "cell_size": float(cell),
                "crs": grid_meta.get("crs_wkt", str(self.shg_crs)),
            },
        )
        encoding = {
            "valid_mask": {
                "zlib": True, "complevel": 5,
                "dtype": "uint8",
                # "_FillValue": np.uint8(0),   # <-- REMOVE THIS LINE
                "chunksizes": (1, min(1024, rows), min(1024, cols)),
            }
        }
        da_valid = da_valid.astype("uint8")
        da_valid.to_netcdf(self.valid_mask_nc_path, encoding=encoding)
        self.valid_mask_nc = da_valid

    # ---- save & config ----
    def _save_outputs(self):
        """Save cumulative precip (NetCDF) and storm centers (Parquet)."""
        storm_names = list(self.cumulative_precip.keys())
        da_stack = xr.concat(
            [self.cumulative_precip[name] for name in storm_names],
            dim=xr.DataArray(storm_names, dims="storm_path", name="storm_path")
        )
        da_stack.name = "cumulative_precip"
        da_stack.to_netcdf(self.nc_path)
        self.storm_centers.to_parquet(self.storm_center_path, index=False)

    def _update_config(self, grid_meta: Dict):
        """Persist paths to the on-disk artifacts (no valid-centroid region)."""
        self.config["preprocessed"] = {
            "watershed_projected": self.watershed_path_out,
            "domain_projected": self.domain_path_out,
            "cumulative_precip": self.nc_path,
            "storm_centers": self.storm_center_path,
            "spatial_stats": self.spatial_stats_path,
            "master_grid_meta": self.grid_meta_path,
            "valid_mask_nc": self.valid_mask_nc_path,
        }
        out_config_path = os.path.join(self.output_folder, "config.json")
        with open(out_config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    # --------------- loader ---------------
    @classmethod
    def load(cls, config_path: Union[str, Path]) -> "Preprocessor":
        """
        Reload a preprocessed dataset from config.json.
        Loads:
          - projected watershed/domain (GPKG)
          - cumulative_precip (NetCDF)
          - storm_centers (Parquet)
          - spatial_stats.json
          - master_grid_meta.json 
          - valid_mask.nc
        """
        config_path = Path(config_path)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.json at {config_path}")

        with open(config_path, "r") as f:
            cfg = json.load(f)
        pre = cfg.get("preprocessed", {})
        if not pre:
            raise ValueError("Missing 'preprocessed' section in config.json")

        obj = cls(config_path=str(config_path), output_folder=str(config_path.parent))
        obj.config = cfg

        obj.watershed_gdf = gpd.read_file(pre["watershed_projected"])
        obj.domain_gdf = gpd.read_file(pre["domain_projected"])

        ds_cum = xr.open_dataset(pre["cumulative_precip"])
        if "cumulative_precip" in ds_cum:
            da = ds_cum["cumulative_precip"]
            obj.cumulative_precip = {
                str(sid): da.sel(storm_path=sid).drop_vars("storm_path")
                for sid in da["storm_path"].values.tolist()
            }
        else:
            obj.cumulative_precip = {"_stack": ds_cum}

        obj.storm_centers = pd.read_parquet(pre["storm_centers"])

        stats_path = pre.get("spatial_stats")
        if stats_path and os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            obj.watershed_stats = pd.Series(stats.get("watershed_stats", {}))
            obj.domain_stats = pd.Series(stats.get("domain_stats", {}))
            obj.spatial_stats_path = stats_path

        obj.grid_meta_path = pre.get("master_grid_meta")

        obj.valid_mask_nc_path = pre.get("valid_mask_nc")
        obj.valid_mask_nc = None
        if obj.valid_mask_nc_path and os.path.exists(obj.valid_mask_nc_path):
            try:
                obj.valid_mask_nc = xr.open_dataarray(obj.valid_mask_nc_path)
            except Exception:
                ds_valid = xr.open_dataset(obj.valid_mask_nc_path)
                obj.valid_mask_nc = ds_valid.get("valid_mask")
        
        if obj.valid_mask_nc is not None:
            if obj.valid_mask_nc.dtype != np.uint8:
                obj.valid_mask_nc = obj.valid_mask_nc.fillna(0).astype("uint8")

        return obj
