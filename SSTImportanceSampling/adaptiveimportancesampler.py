from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from scipy.stats import truncnorm, norm
from shapely.geometry import mapping
from rasterio.features import geometry_mask
from affine import Affine

# ----------------------------- params -----------------------------
@dataclass
class AdaptParams:
    # Narrow (ADAPTED)
    mu_x_n: Optional[float] = None
    mu_y_n: Optional[float] = None
    sd_x_n: float = 1.0
    sd_y_n: float = 1.0
    rho_n: float = 0.0

    # Wide (FIXED)
    mu_x_w: Optional[float] = None
    mu_y_w: Optional[float] = None
    sd_x_w: float = 1.0
    sd_y_w: float = 1.0
    rho_w: float = 0.0

    # Mixture & smoothing
    mix: float = 0.30
    lambda_mix: float = 0.30
    beta_floor: float = 0.05
    beta_ceil: float = 0.95

    alpha: float = 0.25
    eps_var: float = 1e-6

    depth_threshold: float = 0.0
        
    # Numeric stability
    eps_floor: float = 1e-300

# -------------------------- helpers (q_base) --------------------------
def _truncnorm_objs(bounds: Tuple[float,float,float,float], mx, my, sx, sy):
    xmin, ymin, xmax, ymax = bounds
    ax, bx = (xmin - mx)/sx, (xmax - mx)/sx
    ay, by = (ymin - my)/sy, (ymax - my)/sy
    return truncnorm(ax, bx, loc=mx, scale=sx), truncnorm(ay, by, loc=my, scale=sy)

def _qbase_gaussian_copula(xs, ys, bounds, mx, my, sx, sy, rho, eps_floor) -> np.ndarray:
    tx, ty = _truncnorm_objs(bounds, mx, my, sx, sy)
    fx = np.maximum(tx.pdf(xs), eps_floor)  # (nx,)
    fy = np.maximum(ty.pdf(ys), eps_floor)  # (ny,)
    Fx = np.clip(tx.cdf(xs), 1e-15, 1-1e-15)
    Fy = np.clip(ty.cdf(ys), 1e-15, 1-1e-15)
    zx = norm.ppf(Fx)                       # (nx,)
    zy = norm.ppf(Fy)                       # (ny,)

    one_minus_r2 = 1.0 - rho*rho
    const = -0.5*np.log1p(-rho*rho)
    Zx = zx[None, :]
    Zy = zy[:, None]
    A = 0.5*(Zx**2 + Zy**2)
    B = 0.5*((Zx**2 - 2.0*rho*Zx*Zy + Zy**2) / one_minus_r2)
    qcop = np.exp(const + (A - B))
    q2d = qcop * (fy[:, None] * fx[None, :])
    return np.maximum(q2d, eps_floor)

def _qbase_mixture(xs, ys, bounds, p: AdaptParams, eps_floor) -> np.ndarray:
    q_n = _qbase_gaussian_copula(xs, ys, bounds,
                                 mx=p.mu_x_n, my=p.mu_y_n,
                                 sx=p.sd_x_n, sy=p.sd_y_n,
                                 rho=p.rho_n, eps_floor=eps_floor)
    q_w = _qbase_gaussian_copula(xs, ys, bounds,
                                 mx=p.mu_x_w, my=p.mu_y_w,
                                 sx=p.sd_x_w, sy=p.sd_y_w,
                                 rho=p.rho_w, eps_floor=eps_floor)
    return np.maximum(p.mix * q_n + (1.0 - p.mix) * q_w, eps_floor)

# ------------------------ main adaptive class ------------------------
class AdaptiveMixtureSampler:
    """
    Adaptive mixture sampler on valid cells,

    Required on `data`:
      - valid_mask_nc : xr.DataArray ('storm','y','x'), values in {0,1}
      - storm_centers : pd.DataFrame with ['storm_path','x','y']
      - watershed_stats : dict-like or Series with ['x','y'] (for default μ)
      - cumulative_precip : xr.DataArray ('storm','y','x') or dict[str->xr.DataArray(y,x)]
    """

    def __init__(self,
                 data,
                 params: AdaptParams,
                 precip_cube: xr.DataArray | Dict[str, xr.DataArray] | None = None,
                 seed: Optional[int] = None,
                 align_precip: bool = True,
                 align_method: str = "nearest"):
        self.params = params
        self.rng = np.random.default_rng(seed)

        # ---- validate inputs ----
        if not hasattr(data, "valid_mask_nc"):
            raise ValueError("`data` must have attribute `valid_mask_nc` (xarray DataArray).")
        if not hasattr(data, "storm_centers"):
            raise ValueError("`data` must have attribute `storm_centers` (DataFrame).")

        self.mask_da: xr.DataArray = data.valid_mask_nc
        if not {"storm","y","x"} <= set(self.mask_da.dims):
            raise ValueError("valid_mask_nc must have dims ('storm','y','x').")

        # Grid and bounds from mask
        self.xs = self.mask_da["x"].values
        self.ys = self.mask_da["y"].values
        self.colsN = len(self.xs)
        self.rowsN = len(self.ys)
        self.bounds = (float(self.xs[0]), float(self.ys[0]), float(self.xs[-1]), float(self.ys[-1]))
        # --- grid spacing & affine (centers -> upper-left origin)
        self.dx = float(np.mean(np.diff(self.xs)))
        self.dy = float(np.mean(np.diff(self.ys)))
        self.transform = (
            Affine.translation(self.xs[0] - self.dx/2.0, self.ys[0] - self.dy/2.0)
            * Affine.scale(self.dx, self.dy)
        )

        # --- watershed mask
        if not hasattr(data, "watershed_gdf"):
            raise ValueError("`data` must have attribute `watershed_gdf` to compute watershed-mean depths.")
        ws_gdf: gpd.GeoDataFrame = data.watershed_gdf
        self.watershed_mask = geometry_mask(
            geometries=[mapping(geom) for geom in ws_gdf.geometry],
            out_shape=(self.rowsN, self.colsN),
            transform=self.transform,
            invert=True,
        )

        # configurable edge handling
        self.zero_pad_edges = True  # or make this a __init__ kwarg if you want to toggle
        # Centers table filtered to mask storms
        centers_df: pd.DataFrame = data.storm_centers.copy()
        need_cols = {"storm_path","x","y"}
        if not need_cols <= set(centers_df.columns):
            raise ValueError("storm_centers must contain columns ['storm_path','x','y'].")
        storms_mask = self.mask_da["storm"].values.astype(str).tolist()
        centers_df["storm_path"] = centers_df["storm_path"].astype(str)
        centers_df = centers_df[centers_df["storm_path"].isin(storms_mask)].reset_index(drop=True)
        if centers_df.empty:
            raise ValueError("No overlapping storms between valid_mask_nc and storm_centers.")
        self.centers_df = centers_df
        self.storms = centers_df["storm_path"].values.astype(str)
        self.s2i = {s: i for i, s in enumerate(self.storms)}
        self.orig_xy = centers_df.set_index("storm_path").loc[self.storms, ["x","y"]].to_numpy(float)

        # Default μ from watershed centroid
        ws_stats = getattr(data, "watershed_stats", None)
        if ws_stats is None:
            ws_stats = {}
        elif isinstance(ws_stats, pd.Series):
            ws_stats = ws_stats.to_dict()
        elif not isinstance(ws_stats, dict):
            try:
                ws_stats = {"x": float(ws_stats.x), "y": float(ws_stats.y)}
            except Exception:
                ws_stats = {}
        if self.params.mu_x_n is None: self.params.mu_x_n = float(ws_stats.get("x", np.nan))
        if self.params.mu_y_n is None: self.params.mu_y_n = float(ws_stats.get("y", np.nan))
        if self.params.mu_x_w is None: self.params.mu_x_w = float(ws_stats.get("x", np.nan))
        if self.params.mu_y_w is None: self.params.mu_y_w = float(ws_stats.get("y", np.nan))
        if not np.isfinite(self.params.mu_x_n) or not np.isfinite(self.params.mu_y_n):
            raise ValueError("Narrow means not set and watershed_stats[x,y] unavailable.")
        if not np.isfinite(self.params.mu_x_w) or not np.isfinite(self.params.mu_y_w):
            raise ValueError("Wide means not set and watershed_stats[x,y] unavailable.")

        # ---- per-storm valid sets from mask ----
        self.per_storm: Dict[str, Dict[str, np.ndarray | float | int]] = {}
        qb_dummy = np.ones((self.rowsN, self.colsN))  # placeholder; real qb in _rebuild_q_base
        flat_all = qb_dummy.ravel(order="C")
        for name in self.storms:
            m2d = self.mask_da.sel(storm=name).values.astype(bool)
            flat_idx = np.flatnonzero(m2d.ravel(order="C"))
            if flat_idx.size == 0:
                continue
            self.per_storm[name] = {"flat_idx": flat_idx, "count": flat_idx.size}
        if not self.per_storm:
            raise ValueError("All storms have empty valid sets; cannot sample.")

        # ---- build q_base on mask grid ----
        self._rebuild_q_base()

        self.precip_cube = None
        raw_precip = precip_cube if precip_cube is not None else getattr(data, "cumulative_precip", None)
        if raw_precip is not None:
            self.precip_cube = self._normalize_and_align_precip(raw_precip, align=align_precip, method=align_method)

        self.history: List[dict] = []

    # ---------------------- q_base & Z_J -----------------------
    def _rebuild_q_base(self) -> None:
        """
        Build q_base and per-storm Z_J, and also cache component grids q_n, q_w
        so we can compute responsibilities quickly at sampled cells.
        """
        p = self.params
        # component grids
        self.qn2d = _qbase_gaussian_copula(
            self.xs, self.ys, self.bounds,
            mx=p.mu_x_n, my=p.mu_y_n, sx=p.sd_x_n, sy=p.sd_y_n, rho=p.rho_n,
            eps_floor=p.eps_floor
        )
        self.qw2d = _qbase_gaussian_copula(
            self.xs, self.ys, self.bounds,
            mx=p.mu_x_w, my=p.mu_y_w, sx=p.sd_x_w, sy=p.sd_y_w, rho=p.rho_w,
            eps_floor=p.eps_floor
        )
        # mixture on grid
        self.q2d = np.maximum(p.mix * self.qn2d + (1.0 - p.mix) * self.qw2d, p.eps_floor)

        # flatten caches for fast indexing at sampled cells
        self.q_flat  = self.q2d.ravel(order="C")
        self.qn_flat = self.qn2d.ravel(order="C")
        self.qw_flat = self.qw2d.ravel(order="C")

        # Z_J per storm = sum of q_base over that storm's valid set V_J
        for name in self.per_storm.keys():
            ps = self.per_storm[name]
            qb_vals = self.q_flat[ps["flat_idx"]]
            Zj = float(qb_vals.sum())
            ps["qb_vals"] = qb_vals
            ps["Z"] = Zj

    # ------------------- precip normalization/alignment -------------------
    def _normalize_and_align_precip(self, raw, align: bool, method: str) -> xr.DataArray:
        # Normalize to DataArray('storm','y','x') with ascending x/y
        def _ensure_da(da: xr.DataArray) -> xr.DataArray:
            dims = list(da.dims)
            if "storm" not in dims and "storm_path" in dims:
                da = da.rename({"storm_path": "storm"})
            if not {"storm","y","x"} <= set(da.dims):
                raise ValueError("precip DataArray must have dims ('storm','y','x') or pass a dict[str->2D].")
            if da["y"].values[0] > da["y"].values[-1]:
                da = da.sortby("y")
            if da["x"].values[0] > da["x"].values[-1]:
                da = da.sortby("x")
            return da.transpose("storm","y","x")

        if isinstance(raw, xr.Dataset):
            if "cumulative_precip" not in raw:
                raise ValueError("Dataset must contain variable 'cumulative_precip'.")
            cube = _ensure_da(raw["cumulative_precip"])
        elif isinstance(raw, xr.DataArray):
            cube = _ensure_da(raw)
        elif isinstance(raw, dict):
            names, arrs, xs_ref, ys_ref = [], [], None, None
            for k, da in raw.items():
                if not isinstance(da, xr.DataArray) or ("x" not in da.dims or "y" not in da.dims):
                    raise ValueError("Each dict value must be xr.DataArray with dims ('y','x').")
                if da["y"].values[0] > da["y"].values[-1]:
                    da = da.sortby("y")
                if da["x"].values[0] > da["x"].values[-1]:
                    da = da.sortby("x")
                if xs_ref is None:
                    xs_ref, ys_ref = da["x"].values, da["y"].values
                else:
                    if not (np.array_equal(xs_ref, da["x"].values) and np.array_equal(ys_ref, da["y"].values)):
                        raise ValueError("All dict precip arrays must share identical x/y coords.")
                names.append(str(k)); arrs.append(da)
            cube = xr.concat(arrs, dim="storm").assign_coords(storm=np.array(names, dtype=str)).transpose("storm","y","x")
        else:
            raise TypeError("Unsupported precip type.")

        # Filter to storms we actually use
        cube = cube.sel(storm=cube["storm"].astype(str).isin(self.storms))
        # Align to MASK grid if requested or required
        xs_p = cube["x"].values; ys_p = cube["y"].values
        same = np.array_equal(xs_p, self.xs) and np.array_equal(ys_p, self.ys)
        if not same and not align:
            raise ValueError("precip x/y must match mask x/y; set align_precip=True to auto-align.")
        if not same:
            # nearest (default) or linear interpolation onto mask grid
            cube = cube.interp(x=("x", self.xs), y=("y", self.ys), method=method)
        return cube.transpose("storm","y","x")

    # ---------------------------- sampling core ----------------------------
    def _sample_valid_cells(self, n: int, rng: np.random.Generator):
        # 1) storms uniformly among those with ≥1 valid cell
        storms = self.storms
        S = storms.size
        chosen_idx = rng.integers(0, S, size=n)
        chosen_storms = storms[chosen_idx]

        # 2) pick a valid cell per storm, ∝ q_base on that storm's V_J
        chosen_flat = np.empty(n, dtype=int)
        w_raw = np.empty(n, dtype=float)

        uniq, inv = np.unique(chosen_storms, return_inverse=True)
        for k, sname in enumerate(uniq):
            take = np.where(inv == k)[0]
            ps = self.per_storm[sname]
            flat_idx = ps["flat_idx"]
            qb_vals = ps["qb_vals"]; Zj = ps["Z"]; cnt = ps["count"]

            p = qb_vals / (Zj if Zj > 0.0 else np.finfo(float).tiny)
            idx_local = rng.choice(qb_vals.size, size=take.size, replace=True, p=p)
            chosen_flat[take] = flat_idx[idx_local]
            q_at_p = qb_vals[idx_local]
            w_raw[take] = Zj / (float(cnt) * q_at_p)

        return chosen_storms, chosen_flat, w_raw

    def _flat_to_xy(self, flat_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ii = flat_idx // self.colsN
        jj = flat_idx % self.colsN
        return self.xs[jj], self.ys[ii]

    # ------------------------- depths / hits ---------------------
    def _depths_for(self, chosen_storms: np.ndarray, chosen_flat: np.ndarray) -> np.ndarray:
        """
        Compute watershed-mean cumulative precip exactly like StormDepthProcessor:
        - integer cell shifts from physical offsets using dx,dy
        - zero-pad rolled edges
        - mean over watershed mask pixels
        """
        if self.precip_cube is None:
            return np.zeros(chosen_storms.size, dtype=float)

        depths = np.empty(chosen_storms.size, dtype=float)

        # flatten -> (i,j) -> (newx,newy)
        ii = chosen_flat // self.colsN
        jj = chosen_flat % self.colsN
        newx = self.xs[jj]
        newy = self.ys[ii]

        # original centers per storm
        x0 = self.orig_xy[:, 0]
        y0 = self.orig_xy[:, 1]

        for n, s in enumerate(chosen_storms):
            k = self.s2i[str(s)]
            # integer cell offsets computed from physical deltas
            dx_cells = int(round((float(newx[n]) - float(x0[k])) / self.dx))
            dy_cells = int(round((float(newy[n]) - float(y0[k])) / self.dy))

            # take storm precip field and roll
            arr = self.precip_cube.isel(storm=k).values  # (rows, cols)
            shifted = np.roll(arr, shift=(dy_cells, dx_cells), axis=(0, 1))

            # cut rolled edges exactly like processor
            if self.zero_pad_edges:
                if dy_cells > 0:   shifted[:dy_cells, :] = 0.0
                elif dy_cells < 0: shifted[dy_cells:, :] = 0.0
                if dx_cells > 0:   shifted[:, :dx_cells] = 0.0
                elif dx_cells < 0: shifted[:, dx_cells:] = 0.0
            else:
                if dy_cells > 0:   shifted[:dy_cells, :] = np.nan
                elif dy_cells < 0: shifted[dy_cells:, :] = np.nan
                if dx_cells > 0:   shifted[:, :dx_cells] = np.nan
                elif dx_cells < 0: shifted[:, dx_cells:] = np.nan

            # select watershed pixels (the key difference)
            vals = shifted[self.watershed_mask]

            if self.zero_pad_edges:
                # treat residual NaNs as 0.0 and take the mean over the mask
                depths[n] = float(np.nan_to_num(vals, nan=0.0).mean()) if vals.size else np.nan
            else:
                depths[n] = np.nan if np.isnan(vals).any() else (float(vals.mean()) if vals.size else np.nan)

        return depths


    # -------------------------------- adapt --------------------------------
    def adapt(
        self,
        num_iterations: int,
        samples_per_iter: int,
        depth_threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        Adaptive sampling loop with reward-weighted EM (optional depth threshold).

        Each iteration:
            1. Draw ``samples_per_iter`` valid cells across storms.  
            2. Compute self-normalized importance weights.  
            3. Evaluate watershed-average depths at shifted storm centers.  
            4. Update mixture responsibilities and mixture weight using rewards.  
            5. Moment-match narrow distribution parameters.  
            6. Rebuild proposal for the next iteration.  
            7. Append a minimal history snapshot.  

        Parameters
        ----------
        num_iterations : int
            Number of adaptation iterations to run.
        samples_per_iter : int
            Number of samples drawn per iteration.
        depth_threshold : float, default 0.0
            Samples with depth below this threshold contribute zero reward.

        Returns
        -------
        pandas.DataFrame
            History with columns:  
            ``['iter','n','updated','mix','mu_x_n','mu_y_n','sd_x_n','sd_y_n','rho_n','beta_hat_reward']``.
        """
        if self.precip_cube is None:
            raise ValueError("adapt() requires precip_cube to compute depths/reward.")

        p = self.params
        self.history = []

        def _push(iter_idx: int, n: int, updated: int, beta_hat_reward: float | None):
            """Append a minimal history snapshot."""
            self.history.append(dict(
                iter=iter_idx, n=int(n), updated=int(updated),
                mix=float(p.mix),
                mu_x_n=float(p.mu_x_n), mu_y_n=float(p.mu_y_n),
                sd_x_n=float(p.sd_x_n), sd_y_n=float(p.sd_y_n),
                rho_n=float(p.rho_n),
                beta_hat_reward=(float(beta_hat_reward) if beta_hat_reward is not None else None),
            ))

        # Initial snapshot (pre-iteration)
        _push(iter_idx=0, n=0, updated=0, beta_hat_reward=None)

        gamma = 1e-6  # small denominator smoothing

        for t in range(1, num_iterations + 1):
            # 1) Sample valid cells & raw importance weights
            storms, flat_idx, w_raw = self._sample_valid_cells(samples_per_iter, self.rng)

            # 2) Self-normalized weights
            sumW = float(np.sum(w_raw))
            w_tilde = (w_raw / sumW) if (np.isfinite(sumW) and sumW > 0.0) else np.full_like(w_raw, 1.0 / len(w_raw))

            # 3) Coordinates & watershed-average depths
            x_new, y_new = self._flat_to_xy(flat_idx)
            depths = self._depths_for(storms, flat_idx)

            # 4) Responsibilities under current mixture
            qn_at = self.qn_flat[flat_idx]
            qw_at = self.qw_flat[flat_idx]
            denom = np.maximum(p.mix * qn_at + (1.0 - p.mix) * qw_at, p.eps_floor)
            r_n = (p.mix * qn_at) / denom  # responsibility of narrow

            # 5) Reward with threshold
            reward = np.where(depths >= depth_threshold, depths, 0.0)

            # 6) Reward-weighted EM update for mix
            num = float(np.sum(w_tilde * r_n * reward))
            den = float(np.sum(w_tilde * reward)) + gamma
            beta_hat = num / den  # fraction of reward mass explained by narrow
            beta_new = (1.0 - p.lambda_mix) * p.mix + p.lambda_mix * beta_hat
            p.mix = float(np.clip(beta_new, p.beta_floor, p.beta_ceil))

            # 7) Moment-match narrow parameters using RB, reward-weighted samples
            rb_w = w_tilde * r_n * reward
            s_rb = float(np.sum(rb_w))
            updated = 0

            if s_rb > 0.0 and np.count_nonzero(reward) > 10:
                rb_w /= s_rb  # normalize

                # Means (x, y)
                mu_x_hat = float(np.sum(rb_w * x_new))
                mu_y_hat = float(np.sum(rb_w * y_new))

                # Variances (x, y)
                var_x_hat = float(np.sum(rb_w * (x_new - mu_x_hat) ** 2) + p.eps_var)
                var_y_hat = float(np.sum(rb_w * (y_new - mu_y_hat) ** 2) + p.eps_var)

                # Smooth means/variances
                p.mu_x_n = p.alpha * mu_x_hat + (1.0 - p.alpha) * p.mu_x_n
                p.mu_y_n = p.alpha * mu_y_hat + (1.0 - p.alpha) * p.mu_y_n
                vx = p.alpha * var_x_hat + (1.0 - p.alpha) * (p.sd_x_n ** 2)
                vy = p.alpha * var_y_hat + (1.0 - p.alpha) * (p.sd_y_n ** 2)
                p.sd_x_n = float(np.sqrt(max(vx, p.eps_var)))
                p.sd_y_n = float(np.sqrt(max(vy, p.eps_var)))

                # Correlation via latent normals (copula)
                eps = 1e-12
                tx_n, ty_n = _truncnorm_objs(self.bounds, p.mu_x_n, p.mu_y_n, p.sd_x_n, p.sd_y_n)
                u_x = np.clip(tx_n.cdf(x_new), eps, 1 - eps)
                u_y = np.clip(ty_n.cdf(y_new), eps, 1 - eps)
                z_x = norm.ppf(u_x)
                z_y = norm.ppf(u_y)

                mzx = float(np.sum(rb_w * z_x))
                mzy = float(np.sum(rb_w * z_y))
                vzx = float(np.sum(rb_w * (z_x - mzx) ** 2) + p.eps_var)
                vzy = float(np.sum(rb_w * (z_y - mzy) ** 2) + p.eps_var)
                cov_zy = float(np.sum(rb_w * (z_x - mzx) * (z_y - mzy)))
                corr_hat = cov_zy / (np.sqrt(vzx) * np.sqrt(vzy))
                p.rho_n = float(np.clip(p.alpha * corr_hat + (1.0 - p.alpha) * p.rho_n, -0.95, 0.95))

                updated = 1

            # 8) Rebuild proposal
            self._rebuild_q_base()

            # 9) Log snapshot
            _push(iter_idx=t, n=samples_per_iter, updated=updated, beta_hat_reward=beta_hat)

        return pd.DataFrame(self.history)


    # ------------------------------ final draw ------------------------------
    def sample_final(self, n: int, num_realizations: int = 1,
                     with_depths: bool = True,
                     seed: Optional[int] = None) -> pd.DataFrame:
        """
        Final draws from the adapted proposal.
        Uses the same valid-cell logic and weights as in the adapt loop.
        Distinct, stored seed per realization.
        """
        parent_ss = np.random.SeedSequence(seed)
        child_seeds = parent_ss.spawn(num_realizations)

        frames = []
        for r_idx, child_ss in enumerate(child_seeds, start=1):
            realization_seed = int(child_ss.generate_state(1, dtype=np.uint64)[0])
            rng = np.random.default_rng(child_ss)

            storms, flat_idx, w_raw = self._sample_valid_cells(n, rng)
            s = float(np.sum(w_raw))
            w = (w_raw / s) if (np.isfinite(s) and s > 0) else np.full_like(w_raw, 1.0 / len(w_raw))

            newx, newy = self._flat_to_xy(flat_idx)
            kidx = np.array([self.s2i[s] for s in storms], dtype=int)
            ox = self.orig_xy[kidx, 0]; oy = self.orig_xy[kidx, 1]

            depths = self._depths_for(storms, flat_idx) if with_depths else np.zeros(n, dtype=float)

            df = pd.DataFrame({
                "realization": r_idx,
                "realization_seed": realization_seed,
                "event_id": np.arange(1, n + 1, dtype=int),
                "storm_path": storms,
                "x": ox, "y": oy,
                "newx": newx, "newy": newy,
                "delx": newx - ox, "dely": newy - oy,
                "weight_raw": w_raw,
                "weight": w,
                "precip_avg_mm": depths,
            })
            frames.append(df)

        out = pd.concat(frames, ignore_index=True)
        if with_depths:
            out = self._add_exc_prb_per_rep(out)
        return out

    @staticmethod
    def _add_exc_prb_per_rep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["exc_prb"] = np.nan
        for realization, grp in df.groupby("realization", dropna=False):
            w = grp["weight"].to_numpy(float)
            w = w / np.sum(w) if np.sum(w) > 0 else np.full_like(w, 1.0/len(w))
            order = np.argsort(grp["precip_avg_mm"].to_numpy())[::-1]
            idx_sorted = grp.index.to_numpy()[order]
            df.loc[idx_sorted, "exc_prb"] = np.cumsum(w[order])
        return df
