from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from scipy.stats import truncnorm, norm, multivariate_normal
from shapely.geometry import mapping
from rasterio.features import geometry_mask
from affine import Affine


# ----------------------------- parameter bundle -----------------------------

@dataclass
class AdaptParams:
    # ---- required (no defaults) ----
    # Narrow (ADAPTED)
    mu_x_n: float
    mu_y_n: float
    sd_x_n: float
    sd_y_n: float
    # Wide (FIXED)
    mu_x_w: float
    mu_y_w: float
    sd_x_w: float
    sd_y_w: float

    # ---- with defaults (place AFTER all required) ----
    rho_n: float = 0.0
    rho_w: float = 0.0
    mix: float = 0.25
    alpha: float = 0.75
    eps_var: float = 1e-6
    lambda_mix: float = 0.30
    beta_floor: float = 0.01
    beta_ceil: float = 0.80
    K_temper: float = 1.0


# ------------------------------ main sampler --------------------------------

class AdaptiveMixtureSampler:
    """
    """

    # ---------------------------- init & setup -----------------------------

    def __init__(
        self,
        precip_cube: xr.DataArray | xr.Dataset | Dict[str, xr.DataArray],
        storm_centers: pd.DataFrame,
        watershed_gdf: gpd.GeoDataFrame,
        domain_gdf: gpd.GeoDataFrame,
        params: AdaptParams,
        seed: Optional[int] = None,
    ):
        self.params = params
        self.rng = np.random.default_rng(seed)

        # --- normalize precip input to a DataArray cube ('storm_path','y','x') ---
        self.cube = self._normalize_precip(precip_cube, storm_centers)

        # domain / watershed
        self.domain = domain_gdf.iloc[[0]].copy()
        self.watershed = watershed_gdf.iloc[[0]].copy()

        # grid geometry
        self.x_coords = self.cube.coords["x"].values
        self.y_coords = self.cube.coords["y"].values
        self.dx = float(np.mean(np.diff(self.x_coords)))
        self.dy = float(np.mean(np.diff(self.y_coords)))
        self.transform = (
            Affine.translation(self.x_coords[0] - self.dx / 2.0,
                               self.y_coords[0] - self.dy / 2.0)
            * Affine.scale(self.dx, self.dy)
        )

        # watershed mask (True inside)
        self.ws_mask = geometry_mask(
            geometries=[mapping(g) for g in self.watershed.geometry],
            out_shape=(len(self.y_coords), len(self.x_coords)),
            transform=self.transform,
            invert=True,
        )
        self.ws_npix = int(self.ws_mask.sum())
        if self.ws_npix <= 0:
            raise ValueError("Watershed mask is empty.")

        # storm index + centers aligned to cube order
        self.storm_paths: np.ndarray = self.cube.coords["storm_path"].values.astype(str)
        sc = storm_centers.copy()
        sc["storm_path"] = sc["storm_path"].astype(str)
        sc = sc.set_index("storm_path").reindex(self.storm_paths)
        if sc[["x", "y"]].isna().any().any():
            missing = sc.index[sc[["x", "y"]].isna().any(axis=1)].tolist()
            raise KeyError(f"Missing centers for storm(s): {missing}")
        self.center_xy = sc[["x", "y"]].to_numpy(float)

        # target density (uniform over domain polygon)
        self.xmin, self.ymin, self.xmax, self.ymax = self.domain.total_bounds
        self.domain_area = float(self.domain.geometry.iloc[0].area)
        if not np.isfinite(self.domain_area) or self.domain_area <= 0:
            raise ValueError("Domain polygon area must be positive and finite.")
        self.log_p = -np.log(self.domain_area)

        # build proposal components
        self._refresh_components()

        # iteration log
        self.history: List[dict] = []

    # ---------------------- precip normalization ----------------------

    @staticmethod
    def _normalize_precip(
        raw: xr.DataArray | xr.Dataset | Dict[str, xr.DataArray],
        storm_centers: pd.DataFrame,
    ) -> xr.DataArray:
        """
        Return xr.DataArray with dims ('storm_path','y','x') and coords {storm_path,x,y}.
        """
        if isinstance(raw, xr.Dataset):
            if "cumulative_precip" not in raw:
                raise KeyError("Dataset must contain variable 'cumulative_precip'.")
            cube = raw["cumulative_precip"]
        elif isinstance(raw, xr.DataArray):
            cube = raw
        elif isinstance(raw, dict):
            order = storm_centers["storm_path"].astype(str).tolist()
            arrs, names = [], []
            x_ref = y_ref = None
            for sp in order:
                da = raw.get(str(sp))
                if da is None:
                    raise KeyError(f"Storm '{sp}' missing in precip dict.")
                # enforce same x/y across all storms
                if x_ref is None:
                    x_ref = da.coords["x"].values
                    y_ref = da.coords["y"].values
                else:
                    if not np.array_equal(x_ref, da.coords["x"].values) or not np.array_equal(y_ref, da.coords["y"].values):
                        raise ValueError("All precip fields must share identical x/y coordinates.")
                arrs.append(da)
                names.append(str(sp))
            cube = xr.concat(arrs, dim="storm_path").assign_coords(storm_path=np.array(names, dtype=str))
        else:
            raise TypeError("Unsupported type for precip_cube. Use DataArray, Dataset, or dict[str -> DataArray(y,x)].")

        # standardize dims order
        dims = tuple(cube.dims)
        if set(("storm_path", "y", "x")) - set(dims):
            raise ValueError("Precip cube must have dims including 'storm_path','y','x'.")
        cube = cube.transpose("storm_path", "y", "x")
        return cube

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

        # Wide
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
        """log q(x,y) via log-sum-exp over (narrow, wide)."""
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

    def _sample_xy(self, n: int, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Sample n points from the mixture and keep only those inside the domain polygon."""
        rng = self.rng if rng is None else rng
        poly = self.domain.geometry.iloc[0]
        eps = 1e-12
        mix = self.params.mix

        xs, ys = [], []
        batch = max(256, n)
        tries = 0
        while len(xs) < n and tries < 200:
            tries += 1
            choose_n = rng.random(batch) < mix
            x = np.empty(batch, float); y = np.empty(batch, float)

            if choose_n.any():
                u = np.clip(rng.random((choose_n.sum(), 2)), eps, 1 - eps)
                z = norm.ppf(u); zc = z @ self.L_n.T; uc = norm.cdf(zc)
                x[choose_n] = self.tx_n.ppf(np.clip(uc[:, 0], eps, 1 - eps))
                y[choose_n] = self.ty_n.ppf(np.clip(uc[:, 1], eps, 1 - eps))
            if (~choose_n).any():
                u = np.clip(rng.random(((~choose_n).sum(), 2)), eps, 1 - eps)
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

    def _depth_and_storm_for_sample(self, x_new: float, y_new: float, rng: np.random.Generator) -> Tuple[float, str]:
        k = int(rng.integers(low=0, high=len(self.storm_paths)))
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

    def _depths_and_storms(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, List[str]]:
        n = x.size
        depths = np.empty(n, float)
        storms: List[str] = [None] * n
        for i in range(n):
            depths[i], storms[i] = self._depth_and_storm_for_sample(x[i], y[i], rng)
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
        No num_realizations here. Single-stream adaptation.
        """
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

            depths, _ = self._depths_and_storms(x, y, self.rng)
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

                # (a) Means/vars in (x,y) for NARROW
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

                # (b) Copula correlation update on latent normals
                eps = 1e-12
                u_x = np.clip(self.tx_n.cdf(xh), eps, 1 - eps)
                u_y = np.clip(self.ty_n.cdf(yh), eps, 1 - eps)
                z_x = norm.ppf(u_x); z_y = norm.ppf(u_y)

                mzx = float(np.sum(w_hit * z_x)); mzy = float(np.sum(w_hit * z_y))
                vzx = float(np.sum(w_hit * (z_x - mzx) ** 2) + p.eps_var)
                vzy = float(np.sum(w_hit * (z_y - mzy) ** 2) + p.eps_var)
                cov_zy = float(np.sum(w_hit * (z_x - mzx) * (z_y - mzy)))
                corr_hat = cov_zy / (np.sqrt(vzx) * np.sqrt(vzy))

                rho_new = p.alpha * corr_hat + (1 - p.alpha) * p.rho_n
                p.rho_n = float(np.clip(rho_new, -0.95, 0.95))

                updated = 1

            # 5) refresh for next iter
            self._refresh_components()

            # 6) log snapshot
            self.history.append(self._snapshot(
                it=t, ess=ess, hit_raw=hit_rate_raw, hit_w=p_hit_w, updated=updated, n=samples_per_iter
            ))

        return pd.DataFrame(self.history)

    # ----------------------------- final sampling ---------------------------

    @staticmethod
    def _add_exc_prb_per_rep(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["exc_prb"] = np.nan
        for rep, grp in df.groupby("rep", dropna=False):
            w = grp["weight"].to_numpy(float)
            w = w / np.sum(w) if np.sum(w) > 0 else np.full_like(w, 1.0/len(w))
            order = np.argsort(grp["precip_avg_mm"].to_numpy())[::-1]
            idx_sorted = grp.index.to_numpy()[order]
            df.loc[idx_sorted, "exc_prb"] = np.cumsum(w[order])
        return df

    def sample_final(self, n: int, num_realizations: int = 1, with_depths: bool = True, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Draw `n` samples per realization from the FINAL adapted proposal.
        If num_realizations > 1, returns stacked reps with independent RNGs and per-rep weights/exceedance.
        """
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(num_realizations)

        frames = []
        for r_idx, cs in enumerate(child_seeds, start=1):
            rng = np.random.default_rng(cs.generate_state(1, dtype=np.uint64)[0])

            # 1) sample centers
            x, y = self._sample_xy(n, rng=rng)

            # 2) importance weights (normalize per rep)
            log_q = self._log_q_mixture(x, y)
            w = np.exp(self.log_p - log_q)
            s = w.sum()
            w = (w / s) if (np.isfinite(s) and s > 0) else np.full_like(w, 1.0 / len(w))

            # 3) depths + storms
            if with_depths:
                depths, paths = self._depths_and_storms(x, y, rng)
            else:
                depths = np.zeros(n, dtype=float)
                idx = rng.integers(low=0, high=len(self.storm_paths), size=n)
                paths = [str(self.storm_paths[k]) for k in idx]

            # 4) assemble
            df = pd.DataFrame({
                "rep": r_idx,
                "event_id": np.arange(1, n + 1, dtype=int),
                "x": x, "y": y,
                "weight": w,
                "storm_path": paths,
                "precip_avg_mm": depths,
            })
            frames.append(df)

        out = pd.concat(frames, ignore_index=True)

        # 5) per-rep exceedance probabilities
        if with_depths:
            out = self._add_exc_prb_per_rep(out)

        return out

    # ---------------------- helper: weighted frequency ----------------------

    @staticmethod
    def weighted_frequency(depths: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        order = np.argsort(depths)[::-1]
        d = depths[order]
        w = weights[order]
        w = w / np.sum(w) if np.sum(w) > 0 else np.full_like(w, 1.0 / len(w))
        pp = np.cumsum(w)
        return d, pp
