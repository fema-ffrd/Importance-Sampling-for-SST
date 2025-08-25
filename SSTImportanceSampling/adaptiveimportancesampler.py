from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass
from typing import Optional, Tuple, List
from scipy.stats import truncnorm, norm, multivariate_normal
from shapely.geometry import mapping
from rasterio.features import geometry_mask
from affine import Affine

# ----------------------------- parameter bundle -----------------------------

@dataclass
class AdaptParams:
    """
    Parameters for the two-component adaptive mixture.

    Narrow component (adapted over iterations):
        mu_x_n, mu_y_n : initial narrow mean (domain units)
        sd_x_n, sd_y_n : initial narrow std dev
        rho_n          : narrow copula correlation (on latent normals)

    Wide component (fixed throughout adaptation):
        mu_x_w, mu_y_w : wide mean (e.g., domain centroid)
        sd_x_w, sd_y_w : wide std dev (e.g., domain ranges)
        rho_w          : wide copula correlation

    Mixture and adaptation knobs:
        mix            : initial narrow weight (clamped to [beta_floor, beta_ceil])
        alpha          : EMA smoothing for moment-matched mean/variance (narrow)
        eps_var        : variance floor to avoid collapse
        lambda_mix     : damping for mixture update (0=no update, 1=full)
        beta_floor     : min allowed narrow weight
        beta_ceil      : max allowed narrow weight
        K_temper       : nonnegative tempering added to (depth) for hit reweighting
    """
    # Narrow (adapted) — non-defaults
    mu_x_n: float
    mu_y_n: float
    sd_x_n: float
    sd_y_n: float
    # Wide (fixed) — non-defaults
    mu_x_w: float
    mu_y_w: float
    sd_x_w: float
    sd_y_w: float
    # Defaults
    rho_n: float = 0.0
    rho_w: float = 0.0
    mix: float = 0.25
    alpha: float = 0.75
    eps_var: float = 1e-6
    lambda_mix: float = 0.30
    beta_floor: float = 0.01
    beta_ceil: float = 0.80
    K_temper: float = 1.0


class AdaptiveMixtureSampler:
    """
    Adaptive importance sampler over storm centers with real transposition depths.

    Model
    -----
    Proposal q(x,y) is a two-component mixture of rectangle-truncated Normals.
    Each component uses truncated marginals and a Gaussian copula (ρ) on the latent
    standard normals. The **narrow** component parameters (μ, σ) adapt via
    moment matching on **hits** (depth > 0); the **wide** component stays fixed.
    The mixture weight `mix` updates with a damped EM-style step using a smoothed,
    weighted hit rate.

    Depth model (no toy simulator):
    For each sampled (x,y), pick **one storm** from `precip_cube` (with replacement),
    shift the storm grid so its original center moves to (x,y), zero-pad exposed edges,
    mask to the watershed polygon, and compute the basin-averaged depth (mm).

    Inputs
    ------
    precip_cube : xr.DataArray
        Dimensions ('storm_path','y','x'), same projected CRS as (x,y) coordinates.
    storm_centers : pd.DataFrame
        Columns ['storm_path','x','y'] with original catalog centers in the same CRS.
    watershed_gdf : gpd.GeoDataFrame
        One polygon row; defines basin for averaging (same CRS).
    domain_gdf : gpd.GeoDataFrame
        One polygon row; defines the sampling domain (same CRS).
    params : AdaptParams
        Mixture and adaptation hyperparameters.
    seed : Optional[int]
        RNG seed for reproducibility.

    Key Methods
    -----------
    adapt(num_iterations, samples_per_iter) -> pd.DataFrame
        Runs adaptation; returns iteration-by-iteration diagnostics (including iter=0 snapshot).
    sample_final(n, with_depths=True) -> pd.DataFrame
        Draws final samples and returns a **full table** with:
            rep, event_id, x, y, weight, storm_path, precip_avg_mm, exc_prb
        (matching your other pipelines’ outputs).
    weighted_frequency(depths, weights) -> (depths_sorted, exceedance_probs)
        Utility to compute weighted exceedance curves.
    """

    # ---------------------------- init & setup -----------------------------

    def __init__(
        self,
        precip_cube: xr.DataArray,
        storm_centers: pd.DataFrame,
        watershed_gdf: gpd.GeoDataFrame,
        domain_gdf: gpd.GeoDataFrame,
        params: AdaptParams,
        seed: Optional[int] = None,
    ):
        self.cube = precip_cube
        self.domain = domain_gdf.iloc[[0]].copy()
        self.watershed = watershed_gdf.iloc[[0]].copy()
        self.params = params
        self.rng = np.random.default_rng(seed)

        # Grid geometry
        self.x_coords = self.cube.coords["x"].values
        self.y_coords = self.cube.coords["y"].values
        self.dx = float(np.mean(np.diff(self.x_coords)))
        self.dy = float(np.mean(np.diff(self.y_coords)))
        self.transform = (
            Affine.translation(self.x_coords[0] - self.dx / 2.0,
                               self.y_coords[0] - self.dy / 2.0)
            * Affine.scale(self.dx, self.dy)
        )

        # Watershed mask (True = inside)
        self.ws_mask = geometry_mask(
            geometries=[mapping(g) for g in self.watershed.geometry],
            out_shape=(len(self.y_coords), len(self.x_coords)),
            transform=self.transform,
            invert=True,
        )
        self.ws_npix = int(self.ws_mask.sum())
        if self.ws_npix <= 0:
            raise ValueError("Watershed mask is empty.")

        # Storm index + centers aligned to cube order
        self.storm_paths: np.ndarray = self.cube.coords["storm_path"].values
        sc = storm_centers.set_index("storm_path")
        self.center_xy = sc.loc[self.storm_paths, ["x", "y"]].to_numpy()

        # Target density (uniform over domain polygon)
        self.xmin, self.ymin, self.xmax, self.ymax = self.domain.total_bounds
        self.domain_area = float(self.domain.geometry.iloc[0].area)
        if not np.isfinite(self.domain_area) or self.domain_area <= 0:
            raise ValueError("Domain polygon area must be positive and finite.")
        self.log_p = -np.log(self.domain_area)

        # Build distributions
        self._refresh_components()

        # Iteration log
        self.history: List[dict] = []

    # ------------------ internal: distributions & log q --------------------

    def _refresh_components(self):
        """(Re)build truncated marginals and copulas from current params."""
        p = self.params
        eps = 1e-12

        # Narrow
        a_xn = (self.xmin - p.mu_x_n) / p.sd_x_n
        b_xn = (self.xmax - p.mu_x_n) / p.sd_x_n
        a_yn = (self.ymin - p.mu_y_n) / p.sd_y_n
        b_yn = (self.ymax - p.mu_y_n) / p.sd_y_n
        self.tx_n = truncnorm(a_xn, b_xn, loc=p.mu_x_n, scale=p.sd_x_n)
        self.ty_n = truncnorm(a_yn, b_yn, loc=p.mu_y_n, scale=p.sd_y_n)

        # Wide (fixed)
        a_xw = (self.xmin - p.mu_x_w) / p.sd_x_w
        b_xw = (self.xmax - p.mu_x_w) / p.sd_x_w
        a_yw = (self.ymin - p.mu_y_w) / p.sd_y_w
        b_yw = (self.ymax - p.mu_y_w) / p.sd_y_w
        self.tx_w = truncnorm(a_xw, b_xw, loc=p.mu_x_w, scale=p.sd_x_w)
        self.ty_w = truncnorm(a_yw, b_yw, loc=p.mu_y_w, scale=p.sd_y_w)

        # Copulas (Gaussian on latent normals)
        self.cov_n = np.array([[1.0, p.rho_n], [p.rho_n, 1.0]], dtype=float)
        self.cov_w = np.array([[1.0, p.rho_w], [p.rho_w, 1.0]], dtype=float)
        self.L_n = np.linalg.cholesky(self.cov_n + eps * np.eye(2))
        self.L_w = np.linalg.cholesky(self.cov_w + eps * np.eye(2))
        self.phi_n = multivariate_normal(mean=[0.0, 0.0], cov=self.cov_n)
        self.phi_w = multivariate_normal(mean=[0.0, 0.0], cov=self.cov_w)

    def _log_q_mixture(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """log q(x,y) via log-sum-exp over the (narrow, wide) components."""
        eps = 1e-12
        mix = self.params.mix

        # Narrow terms
        Fx_n = np.clip(self.tx_n.cdf(x), eps, 1 - eps); Fy_n = np.clip(self.ty_n.cdf(y), eps, 1 - eps)
        zx_n = norm.ppf(Fx_n); zy_n = norm.ppf(Fy_n)
        log_phi_n = self.phi_n.logpdf(np.column_stack([zx_n, zy_n]))
        log_phi_std_n = norm.logpdf(zx_n) + norm.logpdf(zy_n)
        log_fx_n = np.log(np.clip(self.tx_n.pdf(x), eps, np.inf))
        log_fy_n = np.log(np.clip(self.ty_n.pdf(y), eps, np.inf))
        log_q_n = (log_phi_n - log_phi_std_n) + (log_fx_n + log_fy_n)

        # Wide terms
        Fx_w = np.clip(self.tx_w.cdf(x), eps, 1 - eps); Fy_w = np.clip(self.ty_w.cdf(y), eps, 1 - eps)
        zx_w = norm.ppf(Fx_w); zy_w = norm.ppf(Fy_w)
        log_phi_w = self.phi_w.logpdf(np.column_stack([zx_w, zy_w]))
        log_phi_std_w = norm.logpdf(zx_w) + norm.logpdf(zy_w)
        log_fx_w = np.log(np.clip(self.tx_w.pdf(x), eps, np.inf))
        log_fy_w = np.log(np.clip(self.ty_w.pdf(y), eps, np.inf))
        log_q_w = (log_phi_w - log_phi_std_w) + (log_fx_w + log_fy_w)

        a = np.log(mix + eps) + log_q_n
        b = np.log(1.0 - mix + eps) + log_q_w
        m = np.maximum(a, b)
        return m + np.log(np.exp(a - m) + np.exp(b - m))

    # -------------------- internal: sample (x,y) inside ---------------------

    def _sample_xy(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n points from the mixture and retain only those inside the domain polygon."""
        poly = self.domain.geometry.iloc[0]
        eps = 1e-12
        mix = self.params.mix

        xs, ys = [], []
        batch = max(256, n)
        tries = 0
        while len(xs) < n and tries < 200:
            tries += 1
            choose_n = self.rng.random(batch) < mix
            x = np.empty(batch, float); y = np.empty(batch, float)

            # narrow component
            if choose_n.any():
                u = np.clip(self.rng.random((choose_n.sum(), 2)), eps, 1 - eps)
                z = norm.ppf(u); zc = z @ self.L_n.T; uc = norm.cdf(zc)
                x[choose_n] = self.tx_n.ppf(np.clip(uc[:, 0], eps, 1 - eps))
                y[choose_n] = self.ty_n.ppf(np.clip(uc[:, 1], eps, 1 - eps))
            # wide component
            if (~choose_n).any():
                u = np.clip(self.rng.random(((~choose_n).sum(), 2)), eps, 1 - eps)
                z = norm.ppf(u); zc = z @ self.L_w.T; uc = norm.cdf(zc)
                x[~choose_n] = self.tx_w.ppf(np.clip(uc[:, 0], eps, 1 - eps))
                y[~choose_n] = self.ty_w.ppf(np.clip(uc[:, 1], eps, 1 - eps))

            gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y), crs=self.domain.crs)
            gdf = gdf[gdf.within(poly)]
            xs.extend(gdf.geometry.x.to_numpy())
            ys.extend(gdf.geometry.y.to_numpy())

        xs = np.asarray(xs[:n], float); ys = np.asarray(ys[:n], float)
        return xs, ys

    # --------------- internal: one-sample depth + storm_path ----------------

    def _depth_and_storm_for_sample(self, x_new: float, y_new: float) -> Tuple[float, str]:
        """
        Pick one storm, shift to (x_new, y_new), zero-pad wrap, mask to watershed,
        return (precip_avg_mm, storm_path).
        """
        k = int(self.rng.integers(low=0, high=len(self.storm_paths)))
        storm_path = str(self.storm_paths[k])
        x_orig, y_orig = self.center_xy[k, :]

        dx_cells = int(round((x_new - x_orig) / self.dx))
        dy_cells = int(round((y_new - y_orig) / self.dy))

        arr = self.cube.isel(storm_path=k).values  # (ny, nx)
        shifted = np.roll(arr, shift=(dy_cells, dx_cells), axis=(0, 1))

        if dy_cells > 0:   shifted[:dy_cells, :] = 0
        elif dy_cells < 0: shifted[dy_cells:, :] = 0
        if dx_cells > 0:   shifted[:, :dx_cells] = 0
        elif dx_cells < 0: shifted[:, dx_cells:] = 0

        masked = np.where(self.ws_mask, shifted, 0.0)
        masked = np.where(np.isnan(masked), 0.0, masked)
        precip_avg = float(masked.sum() / self.ws_npix)
        return precip_avg, storm_path

    def _depths_and_storms(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Loop over samples; one storm per sample. Returns (depths_mm, storm_paths)."""
        n = x.size
        depths = np.empty(n, float)
        storms: List[str] = [None] * n
        for i in range(n):
            depths[i], storms[i] = self._depth_and_storm_for_sample(x[i], y[i])
        return depths, storms

    # --------------------------- snapshot helper ----------------------------

    def _snapshot(
        self,
        it: int,
        ess: float = np.nan,
        hit_raw: float = np.nan,
        hit_w: float = np.nan,
        updated: int = 0,
        n: int = 0,
    ) -> dict:
        """Create a log record for the current parameters/diagnostics."""
        p = self.params
        return dict(
            iter=it,
            mix=p.mix,
            mu_x_n=p.mu_x_n, mu_y_n=p.mu_y_n,
            sd_x_n=p.sd_x_n, sd_y_n=p.sd_y_n,
            mu_x_w=p.mu_x_w, mu_y_w=p.mu_y_w,
            sd_x_w=p.sd_x_w, sd_y_w=p.sd_y_w,
            rho_n=p.rho_n, rho_w=p.rho_w,
            ess=ess, hit_rate_raw=hit_raw, hit_rate_weighted=hit_w,
            updated=updated, n=n,
        )

    # ------------------------- adaptation routine --------------------------

    def adapt(self, num_iterations: int, samples_per_iter: int) -> pd.DataFrame:
        """
        Run adaptation and return a history DataFrame with iteration diagnostics.

        Columns include:
            iter, mix, mu_x_n, mu_y_n, sd_x_n, sd_y_n,
            mu_x_w, mu_y_w, sd_x_w, sd_y_w, rho_n, rho_w,
            ess, hit_rate_raw, hit_rate_weighted, updated, n
        """
        # reset and log initial (iter=0) snapshot so plots show the starting guess
        self.history = []
        self.history.append(self._snapshot(it=0))

        for t in range(1, num_iterations + 1):
            # 1) sample xy
            x, y = self._sample_xy(samples_per_iter)

            # 2) weights & depths
            log_q = self._log_q_mixture(x, y)
            w = np.exp(self.log_p - log_q)  # unnormalized
            sumW = w.sum()
            w_norm = (w / sumW) if (np.isfinite(sumW) and sumW > 0) else np.full_like(w, 1.0 / len(w))

            depths, _ = self._depths_and_storms(x, y)
            hits = (depths > 0).astype(int)

            # diagnostics
            ess = float(1.0 / np.sum(w_norm ** 2))
            hit_rate_raw = float(hits.mean())
            gamma = 1e-6
            p_hit_w = float((np.sum(w_norm * hits) + gamma) / (1.0 + 2.0 * gamma))

            # 3) update mix (damped EM-style on weighted hit prob, clamped)
            p = self.params
            beta_new = (1.0 - p.lambda_mix) * p.mix + p.lambda_mix * p_hit_w
            p.mix = float(np.clip(beta_new, p.beta_floor, p.beta_ceil))

            # 4) moment matching on *hits only* (tempered), with EMA smoothing
            idx = np.nonzero(hits)[0]
            updated = 0
            if idx.size > 10:
                w_hit = w_norm[idx] * (depths[idx] + p.K_temper)
                s = w_hit.sum()
                w_hit = (w_hit / s) if (np.isfinite(s) and s > 0) else np.full_like(w_hit, 1.0 / len(w_hit))

                xh = x[idx]; yh = y[idx]

                # ----- (a) Mean/variance updates in (x,y) for the NARROW component -----
                mu_x_new = float(np.sum(w_hit * xh))
                mu_y_new = float(np.sum(w_hit * yh))
                var_x_new = float(np.sum(w_hit * (xh - mu_x_new) ** 2) + p.eps_var)
                var_y_new = float(np.sum(w_hit * (yh - mu_y_new) ** 2) + p.eps_var)

                p.mu_x_n = p.alpha * mu_x_new + (1 - p.alpha) * p.mu_x_n
                p.mu_y_n = p.alpha * mu_y_new + (1 - p.alpha) * p.mu_y_n
                vx = p.alpha * var_x_new + (1 - p.alpha) * (p.sd_x_n ** 2)
                vy = p.alpha * var_y_new + (1 - p.alpha) * (p.sd_y_n ** 2)
                p.sd_x_n = float(np.sqrt(max(vx, p.eps_var)))
                p.sd_y_n = float(np.sqrt(max(vy, p.eps_var)))

                # ----- (b) Copula correlation update on latent normals z -----
                eps = 1e-12
                u_x = np.clip(self.tx_n.cdf(xh), eps, 1 - eps)
                u_y = np.clip(self.ty_n.cdf(yh), eps, 1 - eps)
                z_x = norm.ppf(u_x)
                z_y = norm.ppf(u_y)

                mzx = float(np.sum(w_hit * z_x))
                mzy = float(np.sum(w_hit * z_y))
                vzx = float(np.sum(w_hit * (z_x - mzx) ** 2) + p.eps_var)
                vzy = float(np.sum(w_hit * (z_y - mzy) ** 2) + p.eps_var)
                cov_zy = float(np.sum(w_hit * (z_x - mzx) * (z_y - mzy)))
                corr_hat = cov_zy / (np.sqrt(vzx) * np.sqrt(vzy))

                rho_new = p.alpha * corr_hat + (1 - p.alpha) * p.rho_n
                p.rho_n = float(np.clip(rho_new, -0.95, 0.95))

                updated = 1

            # 5) refresh for next iter
            self._refresh_components()

            # 6) log snapshot for this iteration
            self.history.append(self._snapshot(
                it=t, ess=ess, hit_raw=hit_rate_raw, hit_w=p_hit_w, updated=updated, n=samples_per_iter
            ))

        return pd.DataFrame(self.history)

    # ----------------------------- final sampling ---------------------------

    @staticmethod
    def _add_exc_prb(df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort by precip_avg_mm (desc) and add cumulative exceedance probability
        using the provided weights (already normalized).
        """
        df_sorted = df.sort_values("precip_avg_mm", ascending=False).reset_index(drop=True).copy()
        w = df_sorted["weight"].to_numpy(dtype=float)
        w = w / np.sum(w) if np.sum(w) > 0 else np.full_like(w, 1.0 / len(w))
        df_sorted["exc_prb"] = np.cumsum(w)
        return df_sorted

    def sample_final(self, n: int, with_depths: bool = True) -> pd.DataFrame:
        """
        Draw n from the FINAL adapted proposal and return a full table that
        mirrors the output from your existing pipeline.

        Returns
        -------
        DataFrame with columns:
            rep, event_id, x, y, weight, storm_path, precip_avg_mm, exc_prb
        """
        # 1) sample centers
        x, y = self._sample_xy(n)

        # 2) importance weights under final proposal
        log_q = self._log_q_mixture(x, y)
        w = np.exp(self.log_p - log_q)
        s = w.sum()
        w = (w / s) if (np.isfinite(s) and s > 0) else np.full_like(w, 1.0 / len(w))

        # 3) compute depths and selected storm_path per sample
        if with_depths:
            depths, paths = self._depths_and_storms(x, y)
        else:
            depths = np.zeros(n, dtype=float)
            idx = self.rng.integers(low=0, high=len(self.storm_paths), size=n)
            paths = [str(self.storm_paths[k]) for k in idx]

        # 4) assemble final table (same schema as your other methods)
        df = pd.DataFrame({
            "rep": 1,
            "event_id": np.arange(1, n + 1, dtype=int),
            "x": x,
            "y": y,
            "weight": w,
            "storm_path": paths,
            "precip_avg_mm": depths,
        })

        # 5) add exceedance probability
        df = self._add_exc_prb(df)
        return df

    # ---------------------- helper: weighted frequency ----------------------

    @staticmethod
    def weighted_frequency(depths: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted exceedance curve: sort depths high→low and return (depths_sorted, cumsum(weights_sorted)).
        """
        order = np.argsort(depths)[::-1]
        d = depths[order]
        w = weights[order]
        w = w / np.sum(w) if np.sum(w) > 0 else np.full_like(w, 1.0 / len(w))
        pp = np.cumsum(w)
        return d, pp
