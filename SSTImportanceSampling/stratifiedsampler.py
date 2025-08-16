from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
from joblib import Parallel, delayed
from shapely.geometry import Point, Polygon, box, mapping
from shapely.ops import triangulate
from rasterio.features import geometry_mask


# ---------------------------
# 1) Geometry & grid helpers
# ---------------------------

def _cell_size_from_target_cells(domain: gpd.GeoDataFrame, target_cells: int) -> float:
    """Return square cell edge-length (meters) so total cells ≈ target_cells after clipping."""
    if domain.crs is None:
        raise ValueError("domain GeoDataFrame must have a projected (equal-area) CRS.")
    A = float(domain.geometry.area.sum())
    A_cell = A / max(target_cells, 1)
    return math.sqrt(A_cell)


def _square_grid_over_bbox(bounds: Tuple[float, float, float, float], cell: float) -> Iterable[Polygon]:
    """Generate axis-aligned square polygons that tile the bounding box."""
    minx, miny, maxx, maxy = bounds
    xs = np.arange(minx, maxx + cell, cell)
    ys = np.arange(miny, maxy + cell, cell)
    for x in xs[:-1]:
        for y in ys[:-1]:
            yield box(x, y, x + cell, y + cell)


def make_initial_grid(
    domain: gpd.GeoDataFrame,
    target_cells: int = 96,
    sliver_fraction: float = 0.20,
) -> gpd.GeoDataFrame:
    """
    Create a clipped square grid over an irregular domain polygon.

    Returns a GeoDataFrame with columns:
      - grid_id (int)
      - level (int)
      - area_m2 (float)
      - pi (float)   # cell probability for iteration 1 (area-proportional)
      - geometry (Polygon)
    """
    dom = domain[["geometry"]].copy()
    if len(dom) > 1:
        dom = dom.dissolve().reset_index(drop=True)

    cell = _cell_size_from_target_cells(dom, target_cells)
    raw = gpd.GeoDataFrame(geometry=list(_square_grid_over_bbox(dom.total_bounds, cell)), crs=dom.crs)
    grid = gpd.overlay(raw, dom, how="intersection", keep_geom_type=True)
    grid = grid[~grid.is_empty].reset_index(drop=True)

    A_total = float(dom.geometry.area.sum())
    A_cell_mean_preclip = cell * cell
    min_area = sliver_fraction * A_cell_mean_preclip

    grid["area_m2"] = grid.geometry.area
    grid = grid.loc[grid["area_m2"] > min_area].reset_index(drop=True)

    grid.insert(0, "grid_id", np.arange(1, len(grid) + 1, dtype=int))
    grid["level"] = 1
    grid["pi"] = grid["area_m2"] / A_total

    # stable ordering
    grid = grid.sort_values("area_m2", ascending=False).reset_index(drop=True)
    grid["grid_id"] = np.arange(1, len(grid) + 1, dtype=int)
    return grid


def _sample_point_in_triangle(tri: Polygon) -> Point:
    """Uniform sample inside a triangle via barycentric coords."""
    (x1, y1), (x2, y2), (x3, y3) = np.asarray(tri.exterior.coords)[:3]
    r1, r2 = np.random.rand(2)
    sr1 = math.sqrt(r1)
    u = 1.0 - sr1
    v = sr1 * (1.0 - r2)
    w = sr1 * r2
    return Point(u * x1 + v * x2 + w * x3, u * y1 + v * y2 + w * y3)


def sample_uniform_in_polygon(poly: Polygon, n: int = 1) -> List[Point]:
    """Uniform samples from an arbitrary polygon by area-weighted triangle sampling."""
    tris = triangulate(poly)
    areas = np.array([t.area for t in tris], dtype=float)
    probs = areas / areas.sum()
    idx = np.random.choice(len(tris), size=n, p=probs)
    return [_sample_point_in_triangle(tris[i]) for i in idx]


# --------------------------------------------
# 2) Simple precipitation-depths processor
# --------------------------------------------

class StormDepthProcessorSimple:
    """
    Compute watershed-averaged precipitation depth for sampled (x, y) + chosen storm.
    Expects a preprocessed cube with dims (storm_path, y, x) and a watershed polygon mask.
    """

    def __init__(
        self,
        precip_cube: xr.DataArray,
        storm_centers: pd.DataFrame,  # columns: ['storm_path','x','y'] in same CRS
        watershed_gdf: gpd.GeoDataFrame,
    ) -> None:
        self.cube = precip_cube
        self.storm_centers = storm_centers.set_index("storm_path")
        self.x_coords = precip_cube.x.values
        self.y_coords = precip_cube.y.values
        self.dx = float(np.mean(np.diff(self.x_coords)))
        self.dy = float(np.mean(np.diff(self.y_coords)))
        self.transform = Affine.translation(self.x_coords[0], self.y_coords[0]) * Affine.scale(self.dx, self.dy)

        self.mask = geometry_mask(
            geometries=[mapping(geom) for geom in watershed_gdf.geometry],
            out_shape=(len(self.y_coords), len(self.x_coords)),
            transform=self.transform,
            invert=True,
        )

        if "storm_path" not in precip_cube.dims:
            raise ValueError("precip_cube must have a 'storm_path' dimension.")

    def _shift_and_avg(self, storm_path: str, x_new: float, y_new: float) -> float:
        # original storm center
        try:
            x0, y0 = self.storm_centers.loc[storm_path, ["x", "y"]]
        except KeyError:
            return np.nan

        dx_cells = int(round((x_new - x0) / self.dx))
        dy_cells = int(round((y_new - y0) / self.dy))

        sp_idx = int(np.where(self.cube.storm_path.values == storm_path)[0][0])
        precip = self.cube.isel(storm_path=sp_idx).values  # 2D (y,x)

        shifted = np.roll(precip, shift=(dy_cells, dx_cells), axis=(0, 1))

        # zero-fill wrapped edges
        if dy_cells > 0:
            shifted[:dy_cells, :] = 0
        elif dy_cells < 0:
            shifted[dy_cells:, :] = 0
        if dx_cells > 0:
            shifted[:, :dx_cells] = 0
        elif dx_cells < 0:
            shifted[:, dx_cells:] = 0

        # mask watershed & average
        masked = np.where(self.mask, shifted, 0.0)
        masked = np.where(np.isnan(masked), 0.0, masked)
        return float(masked.sum() / self.mask.sum())

    def compute_depths(
        self,
        samples: pd.DataFrame,  # columns: ['event_id','x','y','storm_path']
        n_jobs: int = -1,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns: event_id, x, y, storm_path, precip_avg_mm."""
        tasks = list(
            zip(samples["event_id"].tolist(), samples["storm_path"].tolist(), samples["x"].tolist(), samples["y"].tolist())
        )

        def _one(ev_id: str, sp: str, x: float, y: float) -> Dict[str, object]:
            pavg = self._shift_and_avg(sp, x, y)
            return {"event_id": ev_id, "x": x, "y": y, "storm_path": sp, "precip_avg_mm": pavg}

        if n_jobs == 1:
            out = [_one(*t) for t in tasks]
        else:
            out = Parallel(n_jobs=n_jobs, prefer="threads")([delayed(_one)(*t) for t in tasks])

        return pd.DataFrame(out)


# ---------------------------------------------------
# 3) Adaptive stratified IS for precipitation depth
# ---------------------------------------------------

@dataclass
class Partition:
    """Holds a grid (GeoDataFrame), with lookup helpers."""
    grid: gpd.GeoDataFrame  # columns: grid_id, area_m2, pi, geometry
    sindex: object          # spatial index for point→cell lookup
    areas: Dict[int, float] # grid_id → area
    pis: Dict[int, float]   # grid_id → pi (cell prob)

    @classmethod
    def from_grid(cls, grid_gdf: gpd.GeoDataFrame) -> "Partition":
        grid = grid_gdf[["grid_id", "area_m2", "pi", "geometry"]].copy()
        sindex = grid.sindex
        areas = dict(zip(grid["grid_id"].tolist(), grid["area_m2"].astype(float).tolist()))
        pis = dict(zip(grid["grid_id"].tolist(), grid["pi"].astype(float).tolist()))
        return cls(grid=grid, sindex=sindex, areas=areas, pis=pis)

    def locate_cell_id(self, x: float, y: float) -> Optional[int]:
        """Return grid_id for the polygon containing (x,y), or None if outside."""
        pt = Point(x, y)
        # quick bbox filter then exact check
        candidates = list(self.sindex.query(pt))
        for idx in candidates:
            if self.grid.iloc[idx].geometry.contains(pt):
                return int(self.grid.iloc[idx]["grid_id"])
        return None


class AdaptiveDepthSIS:
    """
    Adaptive stratified importance sampling for precipitation depths (uniform target in space × storms).
    Uses AMIS to reuse all evaluations across iterations.
    """

    def __init__(
        self,
        domain_gdf: gpd.GeoDataFrame,
        processor: StormDepthProcessorSimple,
        storm_paths: np.ndarray,          # 1D array of storm_path labels
        target_cells: int = 96,
        pi_floor: float = 1e-3,           # minimum per-cell probability
        storm_floor: float = 0.0,         # we keep storm choice uniform; floor unused
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.domain = domain_gdf[["geometry"]].copy()
        self.grid0 = make_initial_grid(domain_gdf, target_cells=target_cells)
        self.partitions: List[Partition] = [Partition.from_grid(self.grid0)]
        self.processor = processor
        self.storm_paths = storm_paths
        self.n_storms = int(storm_paths.size)
        self.pi_floor = float(pi_floor)
        self.rng = rng or np.random.default_rng(1234)

        # constants for the target density p(x,s) = 1/A * 1/|S|
        self.A_total = float(self.domain.geometry.area.sum())
        self.p_storm = 1.0 / self.n_storms
        self.p_target_const = (1.0 / self.A_total) * self.p_storm

        # storage
        self.samples_all: pd.DataFrame = pd.DataFrame()  # all iterations
        self.iter_meta: List[Dict[str, float]] = []      # batch sizes, etc.

    # ---------- sampling helpers ----------

    def _allocate_counts(
        self,
        grid: gpd.GeoDataFrame,
        n_total: int,
        weighted_stats: Optional[pd.DataFrame] = None,
        blend_area: float = 0.3,
    ) -> pd.DataFrame:
        """
        Return grid with a new column 'n_alloc' (int draws per cell).
        Uses Neyman allocation n_c ∝ A_c * sigma_c, blended with area-proportional.
        """
        g = grid[["grid_id", "area_m2", "pi", "geometry"]].copy()

        if weighted_stats is None or weighted_stats.empty or "sigma" not in weighted_stats.columns:
            # first iteration: area-proportional allocation
            probs = g["area_m2"] / g["area_m2"].sum()
        else:
            stats = weighted_stats.set_index("grid_id")
            # use parent's sigma for any missing cell
            sig = g["grid_id"].map(stats["sigma"]).fillna(stats["sigma"].median())
            neyman = g["area_m2"] * sig
            neyman = neyman / neyman.sum()
            area = g["area_m2"] / g["area_m2"].sum()
            probs = blend_area * area + (1.0 - blend_area) * neyman

        # enforce a probability floor via pi_floor
        probs = np.maximum(probs.values, self.pi_floor)
        probs = probs / probs.sum()

        n_alloc = np.floor(probs * n_total).astype(int)
        # adjust rounding
        short = n_total - n_alloc.sum()
        if short > 0:
            extra_idx = np.argsort(-(probs - n_alloc / n_total))[:short]
            n_alloc[extra_idx] += 1

        g["n_alloc"] = n_alloc
        return g

    def _sample_points_in_cells(self, grid_alloc: gpd.GeoDataFrame) -> pd.DataFrame:
        """Sample uniform points inside each cell according to 'n_alloc'."""
        rows = []
        for _, row in grid_alloc.iterrows():
            n = int(row["n_alloc"])
            if n <= 0:
                continue
            pts = sample_uniform_in_polygon(row.geometry, n=n)
            for p in pts:
                rows.append(
                    {
                        "grid_id": int(row["grid_id"]),
                        "x": float(p.x),
                        "y": float(p.y),
                        # proposal density piece for this iteration (space part only)
                        "A_c": float(row["area_m2"]),
                        "pi_c": float(row["pi"]),
                    }
                )
        return pd.DataFrame(rows)

    # ---------- AMIS weighting ----------

    def _q_iter(self, part: Partition, x: float, y: float) -> float:
        """q_t(x,s) for a specific iteration t (uniform storms)."""
        gid = part.locate_cell_id(x, y)
        if gid is None:
            return 0.0
        pi_c = part.pis[gid]
        A_c = part.areas[gid]
        return pi_c * (1.0 / A_c) * self.p_storm  # piecewise uniform × uniform storm

    def _qbar(self, coords: np.ndarray) -> np.ndarray:
        """
        Mixture proposal density over all iterations used so far.
        coords: array of shape (N, 2) with columns [x, y].
        """
        N = coords.shape[0]
        # equal-weight mixture by total draws (deterministic mixture)
        n_per_iter = np.array([m["n_total"] for m in self.iter_meta], dtype=float)
        Ntot = float(n_per_iter.sum())
        weights = n_per_iter / Ntot

        qsum = np.zeros(N, dtype=float)
        for w, part in zip(weights, self.partitions):
            # evaluate per-sample q_t
            q_t = np.array([self._q_iter(part, coords[i, 0], coords[i, 1]) for i in range(N)], dtype=float)
            qsum += w * q_t
        return qsum

    # ---------- public API ----------

    def run(
        self,
        n_iters: int,
        batch_schedule: List[int],
        blend_area: float = 0.3,
        n_jobs: int = -1,
        seed: Optional[int] = 1234,
    ) -> pd.DataFrame:
        """
        Run n_iters of adaptive stratified sampling for depths.
        Returns a DataFrame with all samples and IS weights.
        """
        if len(batch_schedule) != n_iters:
            raise ValueError("batch_schedule length must equal n_iters.")

        if seed is not None:
            np.random.seed(seed)
            self.rng = np.random.default_rng(seed)

        # iteration 1 uses initial grid
        grid_t = self.grid0.copy()

        for t in range(1, n_iters + 1):
            n_total = int(batch_schedule[t - 1])

            # 1) Allocation (Neyman after iter 1)
            weighted_stats = None
            if t > 1:
                weighted_stats = self._weighted_cell_stats_current_partition(grid_t)

            grid_alloc = self._allocate_counts(grid=grid_t, n_total=n_total, weighted_stats=weighted_stats, blend_area=blend_area)

            # 2) Sample points in allocated cells
            pts_df = self._sample_points_in_cells(grid_alloc)
            if pts_df.empty:
                continue

            # choose storms uniformly (you can change this later if you want cell-conditional)
            storms = self.rng.choice(self.storm_paths, size=len(pts_df), replace=True)

            # build samples dataframe for the processor
            pts_df = pts_df.assign(
                event_id=[f"iter{t}_k{k}" for k in range(len(pts_df))],
                storm_path=storms,
                iter_id=t,
            )

            # 3) Compute depths for this batch
            depths_df = self.processor.compute_depths(
                samples=pts_df[["event_id", "x", "y", "storm_path"]],
                n_jobs=n_jobs,
            )

            # 4) Merge metadata
            out_t = pts_df.merge(depths_df, on=["event_id", "x", "y", "storm_path"], how="left")

            # 5) Record current proposal density q_t used at draw time (space × storm)
            out_t["q_draw"] = (out_t["pi_c"] / out_t["A_c"]) * self.p_storm

            # 6) Append to global table
            self.iter_meta.append({"iter": t, "n_total": n_total})
            self.samples_all = pd.concat([self.samples_all, out_t], ignore_index=True)

            # 7) AMIS weights for all samples so far
            coords = self.samples_all[["x", "y"]].to_numpy()
            qbar = self._qbar(coords)
            # guardrail
            eps = 1e-300
            qbar = np.clip(qbar, eps, None)
            w = self.p_target_const / qbar
            self.samples_all["is_weight"] = w

            # 8) Diagnostics (optional): ESS
            W = self.samples_all["is_weight"].values
            ess = (W.sum() ** 2) / (np.square(W).sum() + eps)

            # 9) (Optional) update grid_t for refinement later; here we keep fixed grid
            # If you want refinement later, replace grid_t and add a new Partition.

            # update displayed pi in grid_t (not required for math, purely to inspect)
            grid_t = grid_t.copy()
            grid_t["pi"] = self._reconstruct_cell_pi_from_samples(grid_t)

            print(f"[Iter {t}] batch={n_total}, N_total={len(self.samples_all)}, ESS≈{ess:.1f}")

        return self.samples_all

    # ---------- stats for allocation ----------

    def _assign_cells_current(self, grid: gpd.GeoDataFrame) -> pd.Series:
        """Assign each sample in samples_all to a cell of the *current* grid."""
        part = Partition.from_grid(grid)
        gids = []
        for _, r in self.samples_all.iterrows():
            gid = part.locate_cell_id(float(r["x"]), float(r["y"]))
            gids.append(gid)
        return pd.Series(gids, index=self.samples_all.index, name="grid_id_current")

    def _weighted_cell_stats_current_partition(self, grid: gpd.GeoDataFrame) -> pd.DataFrame:
        """Weighted mean/std of precip per cell using IS weights."""
        if self.samples_all.empty:
            return pd.DataFrame()

        gids = self._assign_cells_current(grid)
        df = self.samples_all.copy()
        df["grid_id_current"] = gids
        df = df.dropna(subset=["grid_id_current"])
        df["grid_id_current"] = df["grid_id_current"].astype(int)

        def _wstats(sub: pd.DataFrame) -> Tuple[float, float, int]:
            w = sub["is_weight"].to_numpy()
            y = sub["precip_avg_mm"].to_numpy()
            if len(w) == 0:
                return (np.nan, np.nan, 0)
            W = w.sum()
            mu = float(np.sum(w * y) / W)
            var = float(np.sum(w * (y - mu) ** 2) / W)
            return (mu, math.sqrt(max(var, 0.0)), len(sub))

        stats = (
            df.groupby("grid_id_current")
            .apply(_wstats)
            .apply(pd.Series)
            .reset_index()
            .rename(columns={"grid_id_current": "grid_id", 0: "mean", 1: "sigma", 2: "n"})
        )
        return stats

    def _reconstruct_cell_pi_from_samples(self, grid: gpd.GeoDataFrame) -> np.ndarray:
        """
        Rough visualization helper: estimate the *effective* cell visitation
        probabilities from the latest mixture (not used for AMIS math).
        """
        if self.samples_all.empty:
            return grid["pi"].to_numpy()

        part = Partition.from_grid(grid)
        counts = np.zeros(len(grid), dtype=float)
        total = 0.0
        for _, r in self.samples_all.iterrows():
            gid = part.locate_cell_id(float(r["x"]), float(r["y"]))
            if gid is None:
                continue
            idx = int(np.where(grid["grid_id"].values == gid)[0][0])
            counts[idx] += 1.0
            total += 1.0
        if total <= 0:
            return grid["pi"].to_numpy()
        est = counts / total
        # floor for display
        est = 0.9 * est + 0.1 * (grid["area_m2"] / grid["area_m2"].sum()).to_numpy()
        return est


# ---------------------------------------------------
# 4) Weighted ECDF & return periods from depths
# ---------------------------------------------------

def weighted_ecdf_return_periods(
    df: pd.DataFrame,
    arrival_rate: float = 10.0,
) -> pd.DataFrame:
    """
    Build a weighted exceedance curve and return periods from importance weights.

    Returns a DataFrame sorted by precip_avg_mm (desc) with columns:
      precip_avg_mm, mass (normalized weight), exc_prob_event, RP_years
    """
    if df.empty:
        return df.copy()

    # Sort by depth descending
    d = df[["precip_avg_mm", "is_weight"]].dropna().sort_values("precip_avg_mm", ascending=False).reset_index(drop=True)

    # Normalize weights to sum to 1 (event-level probability mass under target)
    mass = d["is_weight"].to_numpy()
    mass = mass / mass.sum()

    # For each threshold (each row), event exceedance probability = cumulative mass above threshold
    exc_prob_event = np.cumsum(mass)

    # Annual exceedance probability with Poisson arrival rate λ
    P_annual = 1.0 - np.exp(-arrival_rate * exc_prob_event)
    RP_years = 1.0 / np.clip(P_annual, 1e-12, None)

    out = pd.DataFrame(
        {
            "precip_avg_mm": d["precip_avg_mm"].to_numpy(),
            "mass": mass,
            "exc_prob_event": exc_prob_event,
            "RP_years": RP_years,
        }
    )
    return out
