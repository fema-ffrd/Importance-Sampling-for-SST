import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box, Point
from shapely.ops import unary_union
from affine import Affine
from rasterio.features import geometry_mask
from typing import List, Optional, Tuple


class AdaptiveStratifiedSampler:
    """
    Adaptive stratified (MISER-like) sampler with **per-repetition** global
    importance weights to a **uniform** target, plus a *distance-to-watershed-
    centroid* prior that biases allocation and splitting toward the watershed.

    Prior (per leaf j)
    ------------------
    Let d_j be the Euclidean distance (in CRS units) between the leaf centroid
    and the watershed centroid. Define a prior weight in [prior_floor, 1]:
        prior_j = prior_floor + (1 - prior_floor) * exp(- d_j / prior_sigma)
    If prior_sigma <= 0 (or None), prior_j = 1 for all leaves (no bias).

    Influence
    ---------
    At iteration t in a repetition, the prior is applied with an *effective*
    exponent α_eff(t). You control:
        - prior_alpha ∈ [0,1]: max strength at t=1
        - prior_anneal_iters ≥ 0: number of iters over which it fades to 0
      α_eff(t) = prior_alpha                               if prior_anneal_iters == 0
               = prior_alpha * max(0, 1 - (t-1)/prior_anneal_iters) otherwise

    We modify the base allocation/split scores (which already use variance×area):
        base_score_j = ((var_j + eps_var) * area_j)
        score_j      = base_score_j * (prior_j ** α_eff(t))   # allocation
        split_score  = (var_j * area_j) * (prior_j ** α_eff(t))  # splitting

    Target & weights
    ----------------
    Target density is uniform over the domain: p(s)=1/A_domain.
    After each repetition r, with leaf j area A_j, count n_{j,r}, total N_r,
    we assign to every sample from leaf j in rep r the global weight:
        w_{j,r} = (A_j / A_domain) * (N_r / n_{j,r})
    (This remains valid because sampling is still uniform *within* each leaf.)

    Parameters
    ----------
    data : object with attributes
        cumulative_precip : xr.DataArray (storm_path, y, x), coords x,y,storm_path
        storm_centers     : pd.DataFrame with ['storm_path','x','y'] (same CRS/units)
        domain_gdf        : GeoDataFrame domain polygon(s)
        watershed_gdf     : GeoDataFrame watershed polygon(s) (same CRS/grid as cube)
    nx0, ny0 : int
        Initial grid columns × rows.
    B : int
        Samples per iteration (budget).
    eps_var : float
        Small stabilizer added to per-leaf variance in allocation score.
    n_min_to_split : int
        Minimum per-leaf sample count before a leaf can be split.
    max_depth : int
        Maximum split depth (root=0).
    min_leaf_area_frac : float
        Minimum in-domain area fraction (of A_domain) to keep a child leaf.
    K_split : int
        Number of parents to split per iteration (quad-split each).
    temper_alpha : float in (0,1]
        Tempering exponent for scores (alpha<1 prevents single-leaf domination).
    floor_min1 : bool
        If True and B ≥ #active leaves, ensure each active leaf gets ≥1 planned sample.

    Prior controls
    --------------
    prior_sigma : float | None
        Decay length for exp(-d/sigma). Same units as CRS (e.g., meters). None/<=0 disables prior.
    prior_floor : float in (0,1]
        Lower bound of the prior weight (avoids zeroing far cells).
    prior_alpha : float in [0,1]
        Strength of prior (as an exponent) at t=1.
    prior_anneal_iters : int ≥ 0
        If >0, prior influence decays linearly to zero by this many iterations.

    Attributes
    ----------
    samples : pd.DataFrame
        All samples from all reps:
        ['sample_id','rep','iter','leaf_id','x','y','storm_path','g','w_uniform'].
    grids_by_rep : list[list[gpd.GeoDataFrame]]
        grids_by_rep[r-1] is a list of snapshots for repetition r (with ['rep','iter']).
    leaves_by_rep : list[pd.DataFrame]
        Final leaves table for each repetition (active+inactive) with a 'rep' column.
    """

    def __init__(self,
                 data,
                 nx0: int = 10,
                 ny0: int = 10,
                 B: int = 500,
                 eps_var: float = 1e-6,
                 n_min_to_split: int = 3,
                 max_depth: int = 10,
                 min_leaf_area_frac: float = 1e-5,
                 K_split: int = 5,
                 temper_alpha: float = 0.8,
                 floor_min1: bool = True,
                 # prior controls:
                 prior_sigma: Optional[float] = None,
                 prior_floor: float = 0.2,
                 prior_alpha: float = 1.0,
                 prior_anneal_iters: int = 0):
        # Data
        self.cube: xr.DataArray = data.cumulative_precip
        self.centers: pd.DataFrame = data.storm_centers.set_index("storm_path")
        self.domain_gdf: gpd.GeoDataFrame = data.domain_gdf
        self.watershed_gdf: gpd.GeoDataFrame = data.watershed_gdf

        # Domain & mask
        self.domain_geom = unary_union(self.domain_gdf.geometry)
        self.A_domain = float(self.domain_geom.area)

        x = self.cube.coords["x"].values
        y = self.cube.coords["y"].values
        self.dx = float(np.mean(np.diff(x)))
        self.dy = float(np.mean(np.diff(y)))
        self.transform = Affine.translation(x[0] - self.dx/2.0, y[0] - self.dy/2.0) * Affine.scale(self.dx, self.dy)
        self.ws_mask = geometry_mask(
            geometries=[geom.__geo_interface__ for geom in self.watershed_gdf.geometry],
            out_shape=(len(y), len(x)),
            transform=self.transform,
            invert=True,
        )
        self.storm_paths = self.cube.coords["storm_path"].values

        # Watershed centroid for the distance prior
        self.ws_centroid: Point = unary_union(self.watershed_gdf.geometry).centroid

        # Controls
        self.nx0, self.ny0 = int(nx0), int(ny0)
        self.B = int(B)
        self.eps_var = float(eps_var)
        self.n_min_to_split = int(n_min_to_split)
        self.max_depth = int(max_depth)
        self.min_leaf_area_frac = float(min_leaf_area_frac)
        self.K_split = int(K_split)
        self.temper_alpha = float(temper_alpha)
        self.floor_min1 = bool(floor_min1)

        # Prior controls
        self.prior_sigma = None if (prior_sigma is None or prior_sigma <= 0) else float(prior_sigma)
        self.prior_floor = float(prior_floor)
        self.prior_alpha = float(np.clip(prior_alpha, 0.0, 1.0))
        self.prior_anneal_iters = int(max(0, prior_anneal_iters))

        # Global state across all reps
        self.samples = pd.DataFrame(columns=[
            "sample_id","rep","iter","leaf_id","x","y","storm_path","g","w_uniform"
        ])
        self._next_sample_id = 0
        self.grids_by_rep: List[List[gpd.GeoDataFrame]] = []
        self.leaves_by_rep: List[pd.DataFrame] = []

        # Working state (set at start of each rep)
        self.rep = 0           # current repetition index (1..n_rep)
        self.t = 0             # current iteration within rep
        self.leaves = None     # current leaves table
        self._grids_this_rep: List[gpd.GeoDataFrame] = []

    # ---------------------- public API ----------------------

    def run(self, n_iters: int, n_rep: int = 1,
            rng: Optional[np.random.Generator] = None,
            compute_weights: bool = True) -> Tuple[pd.DataFrame, List[List[gpd.GeoDataFrame]]]:
        """
        Run `n_rep` repetitions, each for `n_iters` iterations.
        Returns (samples, grids_by_rep). Weights are computed per-rep at the end by default.
        """
        rng = rng or np.random.default_rng()

        for r in range(1, n_rep + 1):
            # reset repetition state
            self.rep = r
            self.t = 0
            self.leaves = self._make_initial_grid()
            self._apply_distance_prior(self.leaves)  # initialize 'prior' column
            self._grids_this_rep = []

            # iterate
            for _ in range(n_iters):
                self.step(rng=rng)

            # store final leaves (with rep tag)
            lf = self.leaves.copy()
            lf["rep"] = r
            self.leaves_by_rep.append(lf)

            # store grids for this rep
            self.grids_by_rep.append(self._grids_this_rep)

        if compute_weights:
            self.recompute_uniform_weights_per_rep()

        return self.samples.copy(), [list(lst) for lst in self.grids_by_rep]

    def step(self, rng: Optional[np.random.Generator] = None) -> None:
        """One iteration inside the current repetition."""
        rng = rng or np.random.default_rng()
        self.t += 1

        # 1) Allocate to active leaves
        alloc = self._allocate_active()

        # 2) Sample uniformly inside leaf ∩ domain
        new_pts = self._sample_points(alloc, rng)

        # 3) Evaluate g via random storm transposition
        if not new_pts.empty:
            g_vals, storm_used = self._evaluate_g(new_pts, rng)
            new_pts["g"] = g_vals
            new_pts["storm_path"] = storm_used

            # Append to master and update stats
            self.samples = pd.concat([self.samples, new_pts], ignore_index=True)
            self._update_leaf_stats(new_pts)

        # 4) Split parents (quad) and 5) snapshot grid
        self._split_topK()
        self._snapshot_grid()  # stores this iter's grid with rep/iter

    def recompute_uniform_weights_per_rep(self) -> None:
        """
        Compute **per-repetition** global uniform-target weights and attach to samples:
            w_{j,r} = (A_j / A_domain) * (N_r / n_{j,r}).
        """
        if self.samples.empty:
            return

        # Build area lookups for every (rep, leaf_id)
        if not self.leaves_by_rep:
            raise RuntimeError("No per-rep leaves stored; run() must complete to compute weights.")

        areas = pd.concat(
            [df.loc[:, ["rep","leaf_id","area_in_domain"]] for df in self.leaves_by_rep],
            ignore_index=True
        )

        # Counts per (rep, leaf_id) and totals per rep
        counts = self.samples.groupby(["rep","leaf_id"]).size().rename("n_j").reset_index()
        totals = self.samples.groupby("rep").size().rename("N_rep").reset_index()

        w = (counts
             .merge(areas, on=["rep","leaf_id"], how="left")
             .merge(totals, on="rep", how="left"))
        w["w_uniform"] = (w["area_in_domain"] / self.A_domain) * (w["N_rep"] / w["n_j"])

        # Map back to each sample row
        key = w.set_index(["rep","leaf_id"])["w_uniform"].to_dict()
        self.samples["w_uniform"] = [
            key.get((int(r), int(l)), np.nan) for r, l in zip(self.samples["rep"], self.samples["leaf_id"])
        ]

    # ---------------------- internals ----------------------

    def _effective_prior_alpha(self) -> float:
        """Return α_eff(t) for this iteration within the current repetition."""
        if self.prior_sigma is None or self.prior_alpha <= 0.0:
            return 0.0
        if self.prior_anneal_iters <= 0:
            return self.prior_alpha
        frac = max(0.0, 1.0 - (self.t - 1) / float(self.prior_anneal_iters))
        return self.prior_alpha * frac

    def _apply_distance_prior(self, df: pd.DataFrame) -> None:
        """Compute/attach 'prior' ∈ [prior_floor, 1] for each row of leaves DataFrame."""
        if self.prior_sigma is None:
            df["prior"] = 1.0
            return
        px = 0.5 * (df["x_min"].to_numpy(float) + df["x_max"].to_numpy(float))
        py = 0.5 * (df["y_min"].to_numpy(float) + df["y_max"].to_numpy(float))
        cx, cy = self.ws_centroid.x, self.ws_centroid.y
        d = np.hypot(px - cx, py - cy)  # Euclidean distance in CRS units
        prior = self.prior_floor + (1.0 - self.prior_floor) * np.exp(-d / self.prior_sigma)
        df["prior"] = prior.astype(float)

    def _make_initial_grid(self) -> pd.DataFrame:
        xmin, ymin, xmax, ymax = self.domain_geom.bounds
        dx0 = (xmax - xmin) / self.nx0
        dy0 = (ymax - ymin) / self.ny0
        rows = []
        leaf_id = 0
        for i in range(self.nx0):
            for j in range(self.ny0):
                b = box(xmin + i*dx0, ymin + j*dy0, xmin + (i+1)*dx0, ymin + (j+1)*dy0)
                a_in = b.intersection(self.domain_geom).area
                if a_in > 0:
                    rows.append(dict(
                        leaf_id=leaf_id,
                        x_min=b.bounds[0], x_max=b.bounds[2],
                        y_min=b.bounds[1], y_max=b.bounds[3],
                        area_in_domain=a_in,
                        n=0, mean=0.0, m2=0.0, var=np.nan,
                        depth=0, active=True, alloc=0,
                        prior=1.0  # filled later by _apply_distance_prior
                    ))
                    leaf_id += 1
        return pd.DataFrame(rows)

    def _allocate_active(self) -> pd.Series:
        act = self.leaves.index[self.leaves["active"]].to_list()
        if not act:
            raise RuntimeError("No active leaves to allocate to.")

        area = self.leaves.loc[act, "area_in_domain"].astype(float)
        var = self.leaves.loc[act, "var"].fillna(0.0).clip(lower=0.0)
        base_scores = (var + self.eps_var) * area

        # first-iter fallback to area-only
        if (self.leaves.loc[act, "n"] <= 0).all() or base_scores.sum() <= 0:
            base_scores = area.copy()

        # temper to prevent domination
        if self.temper_alpha < 1.0:
            base_scores = np.power(base_scores, self.temper_alpha)

        # apply prior with effective alpha
        alpha_eff = self._effective_prior_alpha()
        if alpha_eff > 0.0:
            prior_vals = self.leaves.loc[act, "prior"].astype(float).clip(lower=1e-12)
            scores = base_scores * np.power(prior_vals, alpha_eff)
        else:
            scores = base_scores

        alloc = pd.Series(0, index=act, dtype=int)
        B = self.B
        if self.floor_min1 and B >= len(act):
            alloc += 1
            B -= len(act)

        weights = scores / scores.sum()
        alloc_float = B * weights
        alloc += np.floor(alloc_float).astype(int)

        R = self.B - int(alloc.sum())
        if R > 0:
            frac = (alloc_float - np.floor(alloc_float)).sort_values(ascending=False)
            alloc.loc[frac.index[:R]] += 1

        self.leaves.loc[:, "alloc"] = 0
        self.leaves.loc[alloc.index, "alloc"] = alloc.values
        return alloc

    def _sample_points(self, alloc: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
        rows = []
        for _, r in self.leaves[self.leaves["active"]].iterrows():
            K = int(r["alloc"])
            if K <= 0:
                continue
            x0, x1, y0, y1 = r["x_min"], r["x_max"], r["y_min"], r["y_max"]
            leaf_clip = box(x0, y0, x1, y1).intersection(self.domain_geom)
            if leaf_clip.is_empty:
                continue

            kept, attempts = 0, 0
            max_attempts = max(K * 200, 5000)  # try harder for skinny boundary cells
            while kept < K and attempts < max_attempts:
                attempts += 1
                xr = rng.uniform(x0, x1)
                yr = rng.uniform(y0, y1)
                if leaf_clip.covers(Point(xr, yr)):  # includes boundary
                    rows.append(dict(
                        sample_id=self._next_sample_id,
                        rep=self.rep,
                        iter=self.t,
                        leaf_id=int(r["leaf_id"]),
                        x=float(xr), y=float(yr),
                        storm_path="", g=np.nan, w_uniform=np.nan
                    ))
                    self._next_sample_id += 1
                    kept += 1

        return pd.DataFrame(rows)

    def _evaluate_g(self, pts: pd.DataFrame, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        g_vals = np.empty(len(pts), dtype=float)
        storm_used = np.empty(len(pts), dtype=object)
        for i, (_, row) in enumerate(pts.iterrows()):
            sp = rng.choice(self.storm_paths)
            sidx = int(np.where(self.storm_paths == sp)[0][0])

            x_orig, y_orig = self.centers.loc[sp, ["x","y"]]
            dx_cells = int(round((row["x"] - x_orig) / self.dx))
            dy_cells = int(round((row["y"] - y_orig) / self.dy))

            arr = self.cube.isel(storm_path=sidx).values.astype(float)
            shifted = np.roll(arr, shift=(dy_cells, dx_cells), axis=(0,1))
            if dy_cells > 0: shifted[:dy_cells, :] = 0
            elif dy_cells < 0: shifted[dy_cells:, :] = 0
            if dx_cells > 0: shifted[:, :dx_cells] = 0
            elif dx_cells < 0: shifted[:, dx_cells:] = 0
            shifted = np.where(np.isnan(shifted), 0.0, shifted)

            ws_sum = float((shifted * self.ws_mask).sum())
            ws_n = int(self.ws_mask.sum())
            g_vals[i] = ws_sum / ws_n if ws_n > 0 else 0.0
            storm_used[i] = str(sp)
        return g_vals, storm_used

    def _update_leaf_stats(self, new_pts: pd.DataFrame) -> None:
        for leaf_id, df_leaf in new_pts.groupby("leaf_id"):
            vals = df_leaf["g"].to_numpy()
            j = self.leaves.index[self.leaves["leaf_id"] == leaf_id][0]
            n = int(self.leaves.at[j, "n"])
            mean = float(self.leaves.at[j, "mean"])
            m2 = float(self.leaves.at[j, "m2"])
            for v in vals:
                n += 1
                d = v - mean
                mean += d / n
                m2 += d * (v - mean)
            self.leaves.at[j, "n"] = n
            self.leaves.at[j, "mean"] = mean
            self.leaves.at[j, "m2"] = m2
            self.leaves.at[j, "var"] = (m2 / (n - 1)) if n > 1 else np.nan

    def _split_topK(self) -> None:
        cand = self.leaves[
            (self.leaves["active"]) &
            (self.leaves["n"] >= self.n_min_to_split) &
            (self.leaves["depth"] < self.max_depth)
        ].copy()
        if cand.empty:
            return

        # split score with prior influence
        alpha_eff = self._effective_prior_alpha()
        base = cand["var"].fillna(0.0) * cand["area_in_domain"]
        if alpha_eff > 0.0:
            prior_vals = cand["prior"].astype(float).clip(lower=1e-12)
            cand["score"] = base * np.power(prior_vals, alpha_eff)
        else:
            cand["score"] = base

        cand = cand[cand["score"] > 0]
        if cand.empty:
            return

        to_split = cand.sort_values("score", ascending=False).index[:self.K_split]
        new_rows, parents = [], []

        next_leaf_id_base = int(self.leaves["leaf_id"].max()) + 1
        new_count = 0

        for j in to_split:
            r = self.leaves.loc[j]
            xL, xU, yL, yU = r["x_min"], r["x_max"], r["y_min"], r["y_max"]
            x_mid, y_mid = 0.5*(xL+xU), 0.5*(yL+yU)
            children = [
                box(xL, yL, x_mid, y_mid),
                box(x_mid, yL, xU, y_mid),
                box(xL, y_mid, x_mid, yU),
                box(x_mid, y_mid, xU, yU),
            ]
            created = 0
            for b in children:
                a_in = b.intersection(self.domain_geom).area
                if a_in >= self.min_leaf_area_frac * self.A_domain:
                    new_rows.append(dict(
                        leaf_id=next_leaf_id_base + new_count,
                        x_min=b.bounds[0], x_max=b.bounds[2],
                        y_min=b.bounds[1], y_max=b.bounds[3],
                        area_in_domain=a_in,
                        n=0, mean=0.0, m2=0.0, var=np.nan,
                        depth=int(r["depth"]) + 1,
                        active=True, alloc=0,
                        prior=1.0  # fill next
                    ))
                    new_count += 1
                    created += 1
            if created > 0:
                parents.append(j)

        if parents:
            self.leaves.loc[parents, "active"] = False
            self.leaves.loc[parents, "alloc"] = 0

        if new_rows:
            # add children, then compute their priors
            self.leaves = pd.concat([self.leaves, pd.DataFrame(new_rows)], ignore_index=True)
            self._apply_distance_prior(self.leaves)

    def _snapshot_grid(self) -> None:
        polys = [box(r["x_min"], r["y_min"], r["x_max"], r["y_max"]) for _, r in self.leaves.iterrows()]
        gdf = gpd.GeoDataFrame(self.leaves.copy(), geometry=polys, crs=self.domain_gdf.crs)
        gdf["geometry"] = gdf.geometry.intersection(self.domain_geom)
        gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notnull()].copy()
        gdf["rep"] = self.rep
        gdf["iter"] = self.t
        self._grids_this_rep.append(gdf)
